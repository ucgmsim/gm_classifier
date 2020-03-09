import copy
import os
import math
import glob
from typing import Union, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import scipy.signal as signal

from .geoNet_file import GeoNet_File, EmptyFile
from . import features

EVENT_YEARS = [str(ix) for ix in range(1950, 2050, 1)]


def get_event_id(record_ffp: str) -> Union[str, None]:
    event_id = None
    for part in record_ffp.split("/"):
        if "p" in part:
            split_part = part.split("p")
            if (
                len(split_part) == 2
                and split_part[0] in EVENT_YEARS
                and split_part[1].isdigit()
            ):
                return part
        elif part.isdigit() and event_id is None:
            event_id = part
    return event_id


def get_record_id(record_ffp: str) -> str:
    return str(os.path.basename(record_ffp).split(".")[0])


def get_station(record_ffp: str) -> Union[str, None]:
    filename = os.path.basename(record_ffp)
    split_fname = filename.split("_")
    if len(split_fname) == 3:
        return str(split_fname[-1].split(".")[0])
    elif len(split_fname) in [2, 4]:
        return str(split_fname[-2])
    else:
        return None


def get_record_ids_filter(record_list_ffp: str) -> np.ndarray:
    """Gets the record IDs from the specified record list"""
    with open(record_list_ffp, "r") as f:
        record_ids_filter = f.readlines()

    # Strip and drop empty lines
    return np.asarray(
        [
            record_id.strip()
            for record_id in record_ids_filter
            if len(record_id.strip()) > 0 and record_id.strip()[0] != "#"
        ],
        dtype=str,
    )


def process_record(
    record_ffp: str, konno_matrices: Union[str, Dict[int, np.ndarray]]
) -> Union[Tuple[None, None], Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Extracts the features from a GeoNet record file

    Parameters
    ----------
    record_ffp: string
        Path to the record file
    konno_matrices: string or dictionary
        Either a path to a directory containing the Konno matrices files
        or a dictionary of the Konno matrices in memory

    Returns
    -------
    features: dictionary
        Contains all 20 features (quality metrics)
    add_data: dictionary
        Additional data
    """
    # Load the file
    record_filename = os.path.basename(record_ffp)
    try:
        gf = GeoNet_File(record_ffp)
    except EmptyFile as ex:
        print(f"Record {record_filename} - File is empty")
        return None, None

    # Check that record is more than 5 seconds
    if gf.comp_1st.acc.size < 5.0 / gf.comp_1st.delta_t:
        print(f"Record {record_filename} - length is less than 5 seconds, ignored.")
        return None, None

    # Check that all the time-series of the record have the same length
    if not gf.comp_1st.acc.size == gf.comp_2nd.acc.size == gf.comp_up.acc.size:
        print(
            f"Record {record_filename} - The size of the acceleration time-series is "
            f"different between components"
        )
        return None, None

    # Ensure time delay adjusted timeseries still has more than 10 elements
    # Time delay < 0 when buffer start time is before event start time
    if gf.comp_1st.time_delay < 0:
        event_start_ix = math.floor(-1 * gf.comp_1st.time_delay / gf.comp_1st.delta_t)
        if gf.comp_1st.acc.size - event_start_ix < 10:
            print(
                f"Record {record_filename} - less than 10 elements between earthquake "
                f"rupture origin time and end of record"
            )
            return None, None

    # Quick and simple (not the best) baseline correction with demean and detrend
    gf.comp_1st.acc -= gf.comp_1st.acc.mean()
    gf.comp_2nd.acc -= gf.comp_2nd.acc.mean()
    gf.comp_up.acc -= gf.comp_up.acc.mean()

    gf.comp_1st.acc = signal.detrend(gf.comp_1st.acc, type="linear")
    gf.comp_2nd.acc = signal.detrend(gf.comp_2nd.acc, type="linear")
    gf.comp_up.acc = signal.detrend(gf.comp_up.acc, type="linear")

    input_data, add_data = features.get_features(gf, ko_matrices=konno_matrices)

    # Number of zero crossings per 10 seconds less than 10 equals
    # means malfunctioned record
    if add_data is None or add_data["zeroc"] < 10:
        print(
            f"Record {record_filename} - Number of zero crossings per 10 seconds "
            f"less than 10 -> malfunctioned record"
        )
        return None, None

    input_data["record_id"] = get_record_id(record_ffp)
    input_data["event_id"] = get_event_id(record_ffp)
    input_data["station"] = get_station(record_ffp)

    return input_data, add_data


def process_records(
    record_dir: str,
    event_list_ffp: str = None,
    record_list_ffp: str = None,
    ko_matrices_dir: str = None,
    low_mem_usage: bool = False,
    output_ffp: str = None,
) -> pd.DataFrame:
    """Processes a set of record files, allows filtering of which
    records to process

    Parameters
    ----------
    record_dir: string
        Base record directory, a recursive search for
        GeoNet record files is done from here
    record_list_ffp: string, optional
        Path to a text file which contains all records (one per line)
        to be processed. This should be a subset of records found
        from record_dir
    event_list_ffp: string, optional
        Path to a text file which contains all events (one per line)
        to be processed. This should be a subset of records found
        from record_dir
    ko_matrices_dir: str, optional
        Path to directory that contains the Konno
        matrices generated with gen_konno_matrices.py
    low_mem_usage: bool, optional
        If true, then Konno matrices are loaded on-the-fly
        versus loading all into memory initially. This will result in
        a performance drop when processing a large number of records
    output_ffp: string, optional
        Output path, file is regularly backed up to this file during processing,
        all records already existing in this file are skipped

    Returns
    -------
    pandas dataframe
        Dataframe with the features of all the processed records
    """

    def write(feature_df: pd.DataFrame, feature_rows: List):
        print("Writing results..")
        cur_feature_df = pd.DataFrame(feature_rows)
        cur_feature_df.set_index("record_id", drop=True, inplace=True)

        feature_df = (
            pd.concat([feature_df, cur_feature_df])
            if feature_df is not None
            else cur_feature_df
        )
        feature_df.to_csv(output_ffp)

        return feature_df

    record_files = np.asarray(
        glob.glob(os.path.join(record_dir, "**/*.V1A"), recursive=True), dtype=str
    )

    # Filter record files
    if event_list_ffp is not None:
        with open(event_list_ffp, "r") as f:
            events_filter = f.readlines()

        # Strip and drop empty lines
        events_filter = np.asarray(
            [
                event_id.strip()
                for event_id in events_filter
                if len(event_id.strip()) > 0 and event_id.strip()[0] != "#"
            ],
            dtype=str,
        )

        # Filter
        record_events = np.asarray(
            [get_event_id(record_ffp) for record_ffp in record_files], dtype=str
        )
        record_files = record_files[np.isin(record_events, events_filter)]

        missing_events_mask = ~np.isin(events_filter, record_events)
        if np.count_nonzero(missing_events_mask):
            print(
                "These events were specified in the event list, however no "
                "records were found:\n{}".format(
                    "\n".join(events_filter[missing_events_mask])
                )
            )
    elif record_list_ffp is not None:
        record_ids_filter = get_record_ids_filter(record_list_ffp)

        # Filter
        record_ids = np.asarray(
            [get_record_id(record_ffp) for record_ffp in record_files], dtype=str
        )
        record_files = record_files[np.isin(record_ids, record_ids_filter)]

        missing_records_mask = ~np.isin(record_ids_filter, record_ids)
        if np.count_nonzero(missing_records_mask):
            print(
                "No matching record files were found for these records: {}".format(
                    "\n".join(record_ids_filter[missing_records_mask])
                )
            )

    # Load the Konno matrices into memory
    if ko_matrices_dir is not None and not low_mem_usage:
        print(f"Loading Konno matrices into memory")
        konno_matrices = {
            matrix_id: np.load(os.path.join(ko_matrices_dir, f"konno_{matrix_id}.npy"))
            for matrix_id in [1024, 2048, 4096, 8192, 16384, 32768]
        }
    # Calculate the matrices and load into memory
    elif not low_mem_usage and ko_matrices_dir is None:
        print(f"Computing and loading Konno matrices into memory")
        konno_matrices = {
            matrix_id: features.get_konno_matrix(matrix_id * 2, dt=0.005)
            for matrix_id in [1024, 2048, 4096, 8192, 16384, 32768]
        }
    # Load them on the fly for each record
    elif low_mem_usage and ko_matrices_dir is not None:
        print(f"Loading Konno matrices as required")
        konno_matrices = ko_matrices_dir
    else:
        raise ValueError(
            "If the --low_mem_usage option is used, "
            "then the --ko_matrices_dir has to be specified"
        )

    print(f"Starting processing of {record_files.size} record files")

    feature_df = None
    if output_ffp is not None and os.path.isfile(output_ffp):
        print(
            "Output file already exists, filtering out record files that "
            "have already been processed"
        )
        feature_df = pd.read_csv(output_ffp, index_col="record_id")

        # Filter, as to not process already processed records
        record_ids = [get_record_id(record_ffp) for record_ffp in record_files]
        record_files = record_files[~np.isin(record_ids, feature_df.index.values)]

    feature_rows = []
    failed_records = []
    for ix, record_ffp in enumerate(record_files):
        record_name = os.path.basename(record_ffp)
        try:
            print(f"Processing record {record_name}, {ix + 1}/{record_files.size}")
            cur_features, cur_add_data = process_record(
                record_ffp, konno_matrices=konno_matrices
            )
        except Exception as ex:
            print(f"Record {record_name} failed with exception:\n{ex}")
            cur_features, cur_add_data = None, None
            failed_records.append(record_name)

        if cur_features is not None:
            feature_rows.append(cur_features)

        if output_ffp is not None and ix % 100 == 0 and ix > 0 and len(feature_rows) > 0:
            feature_df = write(feature_df, feature_rows)
            feature_rows = []

    # Save
    if feature_df is not None:
        feature_df = write(feature_df, feature_rows)
    # Just return the results
    else:
        feature_df = pd.DataFrame(feature_rows)
        feature_df.set_index("record_id", drop=True, inplace=True)

    if len(failed_records) > 0:
        print(
            "The following records failed processing:\n {}".format(
                "\n".join(failed_records)
            )
        )
    return feature_df
