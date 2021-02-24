import traceback
import os
import math
import glob
from enum import Enum
from pathlib import Path
from typing import Union, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import scipy.signal as signal
from obspy.clients.fdsn import Client as FDSN_Client
from obspy import read, Inventory, read_events

from .geoNet_file import GeoNet_File, EmptyFile
from . import features

EVENT_YEARS = [str(ix) for ix in range(1950, 2050, 1)]

G = 9.80665


class RecordErrorType(Enum):
    # Record total length is less than 5 seconds
    TotalTime = 1

    # The acceleration timeseries have different lengths
    CompsNotMatching = 2

    # Time delay adjusted time series has less than 10 data points
    NQuakePoints = 3

    # Not enough zero crossings
    ZeroCrossings = 4

    # Dt is different between components
    DtNotMatching = 5

class RecordFormat(Enum):
    V1A = "V1A"
    MiniSeed = "mseed"

class RecordError(Exception):
    def __init__(self, message: str, error_type: RecordErrorType):
        super(Exception, self).__init__(message)

        self.error_type = error_type


class Record:

    inventory = None

    def __init__(
        self,
        acc_1: np.ndarray,
        acc_2: np.ndarray,
        acc_v: np.ndarray,
        dt: float,
        record_id: str,
        time_delay: float = None,
    ):
        self.time_delay = time_delay
        self.id = record_id
        self.dt = dt
        self.acc_v = acc_v
        self.acc_2 = acc_2
        self.acc_1 = acc_1

        self.acc_arrays = [self.acc_1, self.acc_2, self.acc_v]
        self._ref_acc = [
            cur_acc for cur_acc in self.acc_arrays if cur_acc is not None
        ][0]

        self.has_horizontal = self.acc_2 is not None and self.acc_1 is not None

        self.size = self._ref_acc.size

        # Basics checks of the record
        self.__sanity_checking()

        self._is_preprocessed = False

    @property
    def is_preprocessed(self):
        return self._is_preprocessed


    def __sanity_checking(self):
        # Check that all components have the same number of data points
        if not np.all(
            [cur_acc.size == self._ref_acc.size for cur_acc in self.acc_arrays]
        ):
            raise RecordError(
                f"Record {self.id} - The acceleration timeseries have different lengths",
                RecordErrorType.CompsNotMatching,
            )

        # Check that record is more than 5 seconds
        if self._ref_acc.size < 5.0 / self.dt:
            raise RecordError(
                f"Record {self.id} - length is less than 5 seconds, ignored.",
                RecordErrorType.TotalTime,
            )

        # Ensure time delay adjusted time-series still has more than 10 elements
        # Time delay < 0 when buffer start time is before event start time
        if self.time_delay is not None and self.time_delay < 0:
            event_start_ix = math.floor(-1 * self.time_delay / self.dt)
            if self.size - event_start_ix < 10:
                raise RecordError(
                    f"Record {self.id} - less than 10 elements between earthquake "
                    f"rupture origin time and end of record",
                    RecordErrorType.NQuakePoints,
                )

    def record_preprocesing(self):
        """De-means and de-trends the acceleration time-series and
        checks if its a malfunctioned record using number of zero crossings
        """
        # Base line correct and de-trending, and compute the number
        # of zero crossing for sanity checking
        zero_crossings = []
        for cur_acc in self.acc_arrays:
            if cur_acc is None:
                zero_crossings.append(np.inf)
                continue

            cur_acc -= cur_acc.mean()
            cur_acc = signal.detrend(cur_acc, type="linear", overwrite_data=True)
            zero_crossings.append(np.count_nonzero(
                np.multiply(cur_acc[0:-2], cur_acc[1:-1]) < 0
            ))

        zeroc = (
                10
                * np.min(zero_crossings)
                / (self.size * self.dt)
        )

        # Number of zero crossings per 10 seconds less than 10 equals
        # means malfunctioned record
        if zeroc < 10:
            raise RecordError(
                f"Record {self.id} - Number of zero crossings per 10 seconds "
                f"less than 10 -> malfunctioned record",
                RecordErrorType.ZeroCrossings,
            )

    @classmethod
    def load_v1a(cls, v1a_ffp: str):
        gf = GeoNet_File(v1a_ffp)

        # Check that dt values are matching
        if not np.isclose(gf.comp_1st.delta_t, gf.comp_up.delta_t) or not np.isclose(
            gf.comp_2nd.delta_t, gf.comp_up.delta_t
        ):
            raise RecordError(
                f"Record {os.path.basename(v1a_ffp).split('.')[0]} - "
                f"The delta_t values are not matching across the components",
                RecordErrorType.DtNotMatching,
            )

        # Check that time_delay values are matching
        if not np.isclose(gf.comp_1st.time_delay, gf.comp_up.time_delay) or not np.isclose(
            gf.comp_2nd.time_delay, gf.comp_up.time_delay
        ):
            raise RecordError(
                f"Record {os.path.basename(v1a_ffp).split('.')[0]} - "
                f"The time_delay values are not matching across the components",
                RecordErrorType.DtNotMatching,
            )

        return cls(
            gf.comp_1st.acc,
            gf.comp_2nd.acc,
            gf.comp_up.acc,
            gf.comp_up.delta_t,
            os.path.basename(v1a_ffp).split(".")[0],
            time_delay=gf.comp_up.time_delay,
        )

    @classmethod
    def load_mseed(cls, mseed_ffp: str, inventory: Inventory = None):
        inventory = inventory if inventory is not None else cls.inventory
        st = read(mseed_ffp)
        if len(st) == 3:
            # Converts it to acceleration in m/s^2
            if st[0].stats.channel[1] == 'N':  # Checks whether data is strong motion
                st_acc = st.copy().remove_sensitivity(inventory=inventory)
            else:
                st_acc = st.copy().remove_sensitivity(inventory=inventory).differentiate()

            # This is a tad awkward, couldn't think of a better way
            # of doing this though.
            # Gets and converts the acceleration data to units g and
            # singles out the vertical component (order of the horizontal
            # ones does not matter)
            acc_data, dt = {}, st_acc[0].stats["delta"]
            for ix, cur_trace in enumerate(st_acc.traces):
                if not np.isclose(cur_trace.stats["delta"], dt):
                    raise RecordError(
                        f"Record {os.path.basename(mseed_ffp).split('.')[0]} - "
                        f"The delta_t values are not matching across the components",
                        RecordErrorType.DtNotMatching,
                    )

                # Vertical channel
                if "Z" in cur_trace.stats["channel"]:
                    acc_data["z"] = cur_trace.data / G
                else:
                    acc_data[ix + 1] = cur_trace.data / G

            return cls(
                acc_data.get(1),
                acc_data.get(2),
                acc_data.get("z"),
                dt,
                os.path.basename(mseed_ffp).split(".")[0],
            )
        else:
            raise ValueError(f"Record {os.path.basename(mseed_ffp).split('.')[0]} has less than 3 traces")

    @classmethod
    def load(cls, ffp: str):
        if os.path.basename(ffp).split(".")[-1].lower() == "v1a":
            return cls.load_v1a(ffp)
        elif os.path.basename(ffp).split(".")[-1].lower() == "mseed":
            if cls.inventory is None:
                print("Loading the station inventory (this may take a few seconds)")
                client_NZ = FDSN_Client("GEONET")
                inventory_NZ = client_NZ.get_stations(level='response')
                client_IU = FDSN_Client("IRIS")
                inventory_IU = client_IU.get_stations(network='IU',station='SNZO',level='response')
                cls.inventory = inventory_NZ+inventory_IU

            return cls.load_mseed(ffp, cls.inventory)

        else:
            raise ValueError(f"Record {ffp} has an invalid format, has to be one of ['V1A', 'mseed']")


def get_event_id(record_ffp: str) -> Union[str, None]:
    """Attempts to get the event ID from an event xml file, otherwise from the record path.
    Might not always work or be correct"""
    event_id = None
    try:
        event = read_events(os.path.abspath(os.path.join(os.path.dirname(record_ffp), '../..', '*.xml')))[0]
        event_id = str(event.resource_id).split("/")[-1]
    except:
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
    elif len(split_fname) == 5:
        return str(split_fname[-3])
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
        Path to the V1A record file
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
    record = Record.load(record_ffp)
    record.record_preprocesing()

    input_data, add_data = features.get_features(
        record,
        ko_matrices=konno_matrices,
    )

    input_data["record_id"] = get_record_id(record_ffp)
    input_data["event_id"] = get_event_id(record_ffp)
    input_data["station"] = get_station(record_ffp)

    return input_data, add_data


def process_records(
    record_dir: str,
    record_format: RecordFormat,
    event_list_ffp: str = None,
    record_list_ffp: str = None,
    ko_matrices_dir: str = None,
    low_mem_usage: bool = False,
    output_dir: str = None,
    output_prefix: str = "features",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Processes a set of record files, allows filtering of which
    records to process

    Parameters
    ----------
    record_dir: string
        Base record directory, a recursive search for
        V1A or mseed record files is done from here
    record_format: string
        Format of the records to look for, either V1A or mseed
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
        glob.glob(os.path.join(record_dir, f"**/*.{record_format.value}"), recursive=True),
        dtype=str,
    )

    # Record files filtering
    if event_list_ffp is not None or record_list_ffp is not None:
        record_files = filter_record_files(
            record_files, event_list_ffp=event_list_ffp, record_list_ffp=record_list_ffp
        )

    # Load the Konno matrices into memory
    if ko_matrices_dir is not None and not low_mem_usage:
        print(f"Loading Konno matrices into memory")
        konno_matrices = {
            matrix_id: np.load(os.path.join(ko_matrices_dir, f"konno_{matrix_id}.npy"))
            for matrix_id in [512, 1024, 2048, 4096, 8192, 16384, 32768]
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
    feature_df_v = None

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

        # Ensure all the existing results are consistent
        assert (
            np.all(feature_df_1.index.values == feature_df_2.index.values)
            and np.all(feature_df_1.index.values == feature_df_v.index.values)
        )

        # Filter, as to not process already processed records
        record_ids = [get_record_id(record_ffp) for record_ffp in record_files]
        record_files = record_files[~np.isin(record_ids, feature_df_1.index.values)]
    elif not output_dir.is_dir():
        output_dir.mkdir()

    feature_rows_1, feature_rows_2 = [], []
    feature_rows_v = []
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
            failed_records[features.FeatureError][ex.error_type].append(
                record_name
            )
            cur_features, cur_add_data = None, None
        except EmptyFile as ex:
            failed_records["empty_file"].append(record_name)
            cur_features, cur_add_data = None, None
        except Exception as ex:
            print(f"Record {record_name} failed due to the error: ")
            traceback.print_exc()
            failed_records["other"].append(record_name)
            cur_features, cur_add_data = None, None

        if cur_features is not None:
            feature_rows_1.append(cur_features["1"])
            feature_rows_2.append(cur_features["2"])
            feature_rows_v.append(cur_features["v"])
        if (
            output_dir is not None
            and ix % 100 == 0
            and ix > 0
            and len(feature_rows_1) > 0
        ):
            feature_df_1, feature_df_2, feature_df_v = write(
                [
                    (feature_df_1, feature_rows_1, output_comp_1_ffp),
                    (feature_df_2, feature_rows_2, output_comp_2_ffp),
                    (feature_df_v, feature_rows_v, output_comp_v_ffp),
                ],
                record_ids,
                event_ids,
                stations,
            )
            record_ids, event_ids, stations = [], [], []
            feature_rows_1, feature_rows_2 = [], []
            feature_rows_v = []

    # Save
    if output_dir is not None:
        feature_df_1, feature_df_2, feature_df_v = write(
            [
                (feature_df_1, feature_rows_1, output_comp_1_ffp),
                (feature_df_2, feature_rows_2, output_comp_2_ffp),
                (feature_df_v, feature_rows_v, output_comp_v_ffp),
            ],
            record_ids,
            event_ids,
            stations,
        )

    return feature_df_1, feature_df_2, feature_df_v, failed_records


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
                "\n".join(feature_erros[features.FeatureErrorType.PGA_zero])
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
