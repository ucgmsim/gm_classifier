import os
import math
import glob
from typing import Union, Tuple, Dict

import numpy as np
import pandas as pd
import scipy.signal as signal

from .geoNet_file import GeoNet_File
from . import features


def process_record(
    record_ffp: str, konno_matrices: Union[str, Dict[int, np.ndarray]]
) -> Union[Tuple[None, None], Tuple[Dict[str, float], Dict[str, float]]]:
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

    # Get the record ID
    record_id = str(os.path.basename(record_ffp).split(".")[0])

    # Load the file
    gf = GeoNet_File(record_ffp)

    # Check that record is more than 5 seconds
    if gf.comp_1st.acc.size < 5.0 / gf.comp_1st.delta_t:
        print(f"Record {record_ffp} - length is less than 5 seconds, ignored.")
        return None, None

    # Check that all the time-series of the record have the same length
    if not gf.comp_1st.acc.size == gf.comp_2nd.acc.size == gf.comp_up.acc.size:
        print(f"The size of the acceleration time-series is "
              f"different between components for record {record_ffp}")
        return None, None

    # TODO: Pretty sure this has a bug in it?
    # When appended zeroes at the beginning of the record are removed, the
    # record might then be empty, skipp processing in such a case
    # agf=adjust_gf_for_time_delay(gf)
    # if agf.comp_1st.acc.size <= 10:
    #     print(f"Record {record_ffp} - less than 10 elements between earthquake rupture origin time and end of record")
    #     return None

    # Ensure time delay adjusted time-series still has more than 10 elements
    event_start_ix = math.floor(gf.comp_1st.time_delay / gf.comp_1st.delta_t)
    if gf.comp_1st.acc.size - event_start_ix < 10:
        print(
            f"Record {record_ffp} - less than 10 elements between earthquake "
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

    input_data["record_id"] = record_id

    return input_data, add_data


def process_records(
    record_dir: str,
    event_list_ffp: str = None,
    record_list_ffp: str = None,
    ko_matrices_dir: str = None,
    low_mem_usage: bool = False,
) -> pd.DataFrame:
    """Processes a set of record files, allows filtering of which
    records to process

    Parameters
    ----------
    record_dir: string
        Base record directory, a recursive search for
        GeoNet record files is done from here
    event_list_ffp: string
        Path to a text file which contains all records (one per line)
        to be processed. This should be a subset of records found
        from record_dir
    ko_matrices_dir: str
        Path to directory that contains the Konno
        matrices generated with gen_konno_matrices.py
    low_mem_usage: bool
        If true, then Konno matrices are loaded on-the-fly
        versus loading all into memory initially. This will result in
        a performance drop when processing a large number of records

    Returns
    -------
    pandas dataframe
        Dataframe with the features of all the processed records
    """
    record_files = np.asarray(
        glob.glob(os.path.join(record_dir, "**/*.V1A"), recursive=True), dtype=str
    )

    # Filter record files
    if event_list_ffp is not None:
        with open(event_list_ffp, "r") as f:
            events_filter = f.readlines()

        # Strip and drop empty lines
        events_filter = np.asarray(
            [event.strip() for event in events_filter if len(event.strip()) > 0],
            dtype=str,
        )

        # Filter
        record_events = np.asarray(
            [
                "_".join(os.path.basename(record_ffp).split("_")[0:-2])
                for record_ffp in record_files
            ],
            dtype=str,
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
        with open(record_list_ffp, "r") as f:
            record_ids_filter = f.readlines()

        # Strip and drop empty lines
        record_ids_filter = np.asarray(
            [event.strip() for event in record_ids_filter if len(event.strip()) > 0],
            dtype=str,
        )

        # Filter
        record_ids = np.asarray(
            [
                os.path.basename(record_ffp).split(".")[0]
                for record_ffp in record_files
            ],
            dtype=str,
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

    features_rows = []
    n_failed_records = 0
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
            n_failed_records += 1

        # Number of zero crossings per 10 seconds less than 10 equals
        # means malfunctioned record
        if cur_add_data is None or cur_add_data["zeroc"] < 10:
            continue

        features_rows.append(cur_features)

    feature_df = pd.DataFrame(features_rows)
    feature_df.set_index("record_id", drop=True, inplace=True)

    if n_failed_records > 0:
        print(
            f"{n_failed_records} records failed processing. "
            f"Check the log to see what went wrong."
        )
    return feature_df
