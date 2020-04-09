import time
import copy
import os
import math
import glob
from enum import Enum
from pathlib import Path
from typing import Union, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import scipy.signal as signal

from .geoNet_file import GeoNet_File, EmptyFile
from . import features

EVENT_YEARS = [str(ix) for ix in range(1950, 2050, 1)]


class RecordErrorType(Enum):
    # Record total length is less than 5 seconds
    TotalTime = 1

    # The acceleration timeseries have different lengths
    CompsNotMatching = 2

    # Time delay adjusted time series has less than 10 data points
    NQuakePoints = 3

    # Not enough zero crossings
    ZeroCrossings = 4


class RecordError(Exception):
    def __init__(self, message: str, error_type: RecordErrorType):
        super(Exception, self).__init__(message)

        self.error_type = error_type


def get_event_id(record_ffp: str) -> Union[str, None]:
    """Attempts to get the event ID from the record path.
    Might not always be correct"""
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
    """Gets the record id from the record file name"""
    return str(os.path.basename(record_ffp).split(".")[0])


def get_station(record_ffp: str) -> Union[str, None]:
    """Gets the station name from the record file name"""
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


def record_preprocesing(gf: GeoNet_File) -> GeoNet_File:
    """Performs some checks on the record and de-means and de-trends
    the acceleration time-series"""
    record_filename = os.path.basename(gf.record_ffp)

    # Check that record is more than 5 seconds
    if gf.comp_1st.acc.size < 5.0 / gf.comp_1st.delta_t:
        raise RecordError(
            f"Record {record_filename} - length is less than 5 seconds, ignored.",
            RecordErrorType.TotalTime,
        )

    # Check that all the time-series of the record have the same length
    if not gf.comp_1st.acc.size == gf.comp_2nd.acc.size == gf.comp_up.acc.size:
        raise RecordError(
            f"Record {record_filename} - The size of the acceleration time-series is "
            f"different between components",
            RecordErrorType.CompsNotMatching,
        )

    # Ensure time delay adjusted timeseries still has more than 10 elements
    # Time delay < 0 when buffer start time is before event start time
    if gf.comp_1st.time_delay < 0:
        event_start_ix = math.floor(-1 * gf.comp_1st.time_delay / gf.comp_1st.delta_t)
        if gf.comp_1st.acc.size - event_start_ix < 10:
            raise RecordError(
                f"Record {record_filename} - less than 10 elements between earthquake "
                f"rupture origin time and end of record",
                RecordErrorType.NQuakePoints,
            )

    # Quick and simple (not the best) baseline correction with demean and detrend
    gf.comp_1st.acc -= gf.comp_1st.acc.mean()
    gf.comp_2nd.acc -= gf.comp_2nd.acc.mean()
    gf.comp_up.acc -= gf.comp_up.acc.mean()

    gf.comp_1st.acc = signal.detrend(gf.comp_1st.acc, type="linear")
    gf.comp_2nd.acc = signal.detrend(gf.comp_2nd.acc, type="linear")
    gf.comp_up.acc = signal.detrend(gf.comp_up.acc, type="linear")

    # Check number of zero crossings by
    zeroc_1 = np.count_nonzero(
        np.multiply(gf.comp_1st.acc[0:-2], gf.comp_1st.acc[1:-1]) < 0
    )
    zeroc_2 = np.count_nonzero(
        np.multiply(gf.comp_2nd.acc[0:-2], gf.comp_2nd.acc[1:-1]) < 0
    )
    zeroc_3 = np.count_nonzero(
        np.multiply(gf.comp_up.acc[0:-2], gf.comp_up.acc[1:-1]) < 0
    )

    zeroc = (
        10
        * np.min([zeroc_1, zeroc_2, zeroc_3])
        / (gf.comp_1st.acc.size * gf.comp_1st.delta_t)
    )

    # Number of zero crossings per 10 seconds less than 10 equals
    # means malfunctioned record
    if zeroc < 10:
        raise RecordError(
            f"Record {record_filename} - Number of zero crossings per 10 seconds "
            f"less than 10 -> malfunctioned record",
            RecordErrorType.ZeroCrossings,
        )

    return gf


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
    gf = GeoNet_File(record_ffp)
    record_preprocesing(gf)

    input_data, add_data = features.get_features(gf, ko_matrices=konno_matrices)

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
    output_dir: str = None,
    output_prefix: str = "features",
) -> Tuple[pd.DataFrame, Dict]:
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
    output_dir: string, optional
        Output directory, features are regularly saved to this location
    output_prefix: string, optional
        Prefix for the output files

    Returns
    -------
    pandas dataframe
        Dataframe with the features of all the processed records
    dictionary
        The names of the failed records, where the keys are the different
        error types
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_comp_1_ffp = output_dir / f"{output_prefix}_comp_X.csv"
        output_comp_2_ffp = output_dir / f"{output_prefix}_comp_Y.csv"
        output_comp_v_ffp = output_dir / f"{output_prefix}_comp_Z.csv"
        output_gm_ffp = output_dir / f"{output_prefix}_gm.csv"
    else:
        print(f"No output directory, results are not saved and only returned")

    def write(
        write_data: List[Tuple[pd.DataFrame, List, str]],
        record_ids: List[str],
        event_ids: List[str],
        stations: List[str],
    ):
        print("Writing results..")
        results = []
        for cur_feature_df, cur_feature_rows, cur_output_ffp in write_data:
            new_feature_df = pd.DataFrame(cur_feature_rows)
            new_feature_df.index = record_ids
            new_feature_df["event_id"] = event_ids
            new_feature_df["station"] = stations

            cur_feature_df = (
                pd.concat([cur_feature_df, new_feature_df])
                if cur_feature_df is not None
                else new_feature_df
            )
            cur_feature_df.to_csv(cur_output_ffp, index_label="record_id")

            results.append(cur_feature_df)

        return results

    print(f"Searching for record files")
    record_files = np.asarray(
        glob.glob(os.path.join(record_dir, "**/*.V1A"), recursive=True), dtype=str
    )

    # Record files filtering
    if event_list_ffp is not None or record_list_ffp is not None:
        record_files = filter_record_files(
            record_files, event_list_ffp=event_list_ffp, record_list_ffp=record_list_ffp
        )

    # Hack that (partially) allows getting around obspy issue, when running this
    # function on a loop...
    np.random.seed(int(time.time()))
    np.random.shuffle(record_files)

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

    feature_df_1, feature_df_2 = None, None
    feature_df_v, feature_df_gm = None, None

    # Load existing results if they exists
    if (
        output_dir is not None
        and output_dir.is_dir()
        and all(
            [
                ffp.is_file()
                for ffp in [
                    output_comp_1_ffp,
                    output_comp_2_ffp,
                    output_comp_v_ffp,
                    output_gm_ffp,
                ]
            ]
        )
    ):
        print(
            "Output directory and files already exist, existing results will be used "
            "(and already processed records will be ignored)"
        )
        feature_df_1 = pd.read_csv(output_comp_1_ffp, index_col="record_id")
        feature_df_2 = pd.read_csv(output_comp_2_ffp, index_col="record_id")
        feature_df_v = pd.read_csv(output_comp_v_ffp, index_col="record_id")
        feature_df_gm = pd.read_csv(output_gm_ffp, index_col="record_id")

        # Ensure all the existing results are consistent
        assert (
            np.all(feature_df_1.index.values == feature_df_2.index.values)
            and np.all(feature_df_1.index.values == feature_df_v.index.values)
            and np.all(feature_df_1.index.values == feature_df_gm.index.values)
        )

        # Filter, as to not process already processed records
        record_ids = [get_record_id(record_ffp) for record_ffp in record_files]
        record_files = record_files[~np.isin(record_ids, feature_df_gm.index.values)]
    elif output_dir.is_dir() == False:
        output_dir.mkdir()

    feature_rows_1, feature_rows_2 = [], []
    feature_rows_v, feature_rows_gm = [], []
    failed_records = {
        RecordError: {err_type: [] for err_type in RecordErrorType},
        features.FeatureError: {err_type: [] for err_type in features.FeatureErrorType},
        "empty_file": [],
        "other": [],
    }
    record_ids, event_ids, stations = [], [], []
    for ix, record_ffp in enumerate(record_files):
        record_name = os.path.basename(record_ffp)
        try:
            print(f"Processing record {record_name}, {ix + 1}/{record_files.size}")
            cur_features, cur_add_data = process_record(
                record_ffp, konno_matrices=konno_matrices
            )
            record_ids.append(cur_features["record_id"])
            event_ids.append(cur_features["event_id"])
            stations.append(cur_features["station"])
        except RecordError as ex:
            failed_records[RecordError][ex.error_type].append(record_name)
            cur_features, cur_add_data = None, None
        except features.FeatureError as ex:
            failed_records[features.FeatureError][features.FeatureErrorType].append(
                record_name
            )
            cur_features, cur_add_data = None, None
        except EmptyFile as ex:
            failed_records["empty_file"].append(record_name)
            cur_features, cur_add_data = None, None
        except Exception as ex:
            failed_records["other"].append(record_name)
            cur_features, cur_add_data = None, None

        if cur_features is not None:
            feature_rows_1.append(cur_features["1"])
            feature_rows_2.append(cur_features["2"])
            feature_rows_v.append(cur_features["v"])
            feature_rows_gm.append(cur_features["gm"])
        if (
            output_dir is not None
            and ix % 100 == 0
            and ix > 0
            and len(feature_rows_gm) > 0
        ):
            feature_df_1, feature_df_2, feature_df_v, feature_df_gm = write(
                [
                    (feature_df_1, feature_rows_1, output_comp_1_ffp),
                    (feature_df_2, feature_rows_2, output_comp_2_ffp),
                    (feature_df_v, feature_rows_v, output_comp_v_ffp),
                    (feature_df_gm, feature_rows_gm, output_gm_ffp),
                ],
                record_ids, event_ids, stations
            )
            record_ids, event_ids, stations = [], [], []
            feature_rows_1, feature_rows_2 = [], []
            feature_rows_v, feature_rows_gm = [], []

    # Save
    if output_dir is not None:
        feature_df_1, feature_df_2, feature_df_v, feature_df_gm = write(
            [
                (feature_df_1, feature_rows_1, output_comp_1_ffp),
                (feature_df_2, feature_rows_2, output_comp_2_ffp),
                (feature_df_v, feature_rows_v, output_comp_v_ffp),
                (feature_df_gm, feature_rows_gm, output_gm_ffp),
            ],
            record_ids, event_ids, stations
        )
    # Just return the results
    else:
        feature_df_gm = pd.DataFrame(feature_rows_gm)
        feature_df_gm.index = record_ids

    return feature_df_gm, failed_records


def print_errors(failed_records: Dict[Any, Dict[Any, List]]):
    feature_erros = failed_records[features.FeatureError]
    record_erros = failed_records[RecordError]
    if len(record_erros[RecordErrorType.TotalTime]) > 0:
        print(
            "The following records failed processing due to "
            "the record length < 5 seconds:\n {}".format(
                "\n".join(record_erros[RecordErrorType.TotalTime])
            )
        )
    if len(record_erros[RecordErrorType.CompsNotMatching]) > 0:
        print(
            "The following records failed processing due to the "
            "acceleration timeseries of the components having"
            "different length:\n{}".format(
                "\n".join(record_erros[RecordErrorType.CompsNotMatching])
            )
        )
    if len(record_erros[RecordErrorType.NQuakePoints]) > 0:
        print(
            "The following records failed processing due to the "
            "time delay adjusted timeseries having less than "
            "10 datapoints:\n{}".format(
                "\n".join(record_erros[RecordErrorType.NQuakePoints])
            )
        )
    if len(record_erros[RecordErrorType.ZeroCrossings]) > 0:
        print(
            "The following records failed processing due to"
            "not meeting the required number of "
            "zero crossings:\n{}".format(
                "\n".join(record_erros[RecordErrorType.ZeroCrossings])
            )
        )
    if len(failed_records["empty_file"]) > 0:
        print(
            "The following records failed processing due to the "
            "geoNet file not containing any data:\n{}".format(
                "\n".join(failed_records["empty_file"])
            )
        )
    if len(failed_records["other"]):
        print(
            "The following records failed processing due to "
            "an unknown exception:\n{}".format("\n".join(failed_records["other"]))
        )
    if len(feature_erros[features.FeatureErrorType.PGA_zero]) > 0:
        print(
            "The following records failed processing due to "
            "one PGA being zero for one (or more) of the components:\n{}".format(
                "\n".join(feature_erros["other"])
            )
        )


def filter_record_files(
    record_files: np.ndarray, event_list_ffp: str = None, record_list_ffp: str = None
):
    """Filters the given record files by either event or record ID"""
    assert event_list_ffp is not None or record_list_ffp is not None

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

    return record_files
