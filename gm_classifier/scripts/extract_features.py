import math
import copy
import os
import glob
import argparse
from typing import Dict, Union

import scipy.signal as signal
import numpy as np

import gm_classifier as gm


def adjust_for_time_delay(ts, dt, shift):
    """
        ts: time series data
    """
    t0_index = int(shift / dt)
    if t0_index == 0:
        num_pts = ts.size
    elif t0_index > 0:
        ts = np.concatenate((np.zeros(t0_index), ts))
        num_pts = ts.size
    elif t0_index < 0:
        ts = ts[np.abs(t0_index) :]
        num_pts = ts.size

    return ts, num_pts, dt


def adjust_gf_for_time_delay(gf):
    """
    Note:
        Only works with geoNet Vol1 data
    gf:
        is of type geoNet_file
    returns:
        a deep copy of gf, leaving the original untouched
    """
    gf = copy.deepcopy(gf)
    gf.comp_1st.acc, _, _ = adjust_for_time_delay(
        gf.comp_1st.acc, gf.comp_1st.delta_t, gf.comp_1st.time_delay
    )
    gf.comp_1st.time_delay = 0.0

    gf.comp_2nd.acc, _, _ = adjust_for_time_delay(
        gf.comp_2nd.acc, gf.comp_2nd.delta_t, gf.comp_2nd.time_delay
    )

    gf.comp_2nd.time_delay = 0.0

    gf.comp_up.acc, _, _ = adjust_for_time_delay(
        gf.comp_up.acc, gf.comp_up.delta_t, gf.comp_up.time_delay
    )
    gf.comp_up.time_delay = 0.0

    return gf


def process_rf(record_ffp: str, konno_matrices: Union[str, Dict[int, np.ndarray]]):
    """Process a single record file"""

    # Get the record ID
    record_id = str(os.path.basename(record_ffp).split(".")[0])

    # Load the file
    gf = gm.GeoNet_File(record_ffp)

    # Check that record is more than 5 seconds
    if gf.comp_1st.acc.size < 5.0 / gf.comp_1st.delta_t:
        print(f"Record {record_ffp} - length is less than 5 seconds, ignored.")
        return None

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
        return None

    # Quick and simple (not the best) baseline correction with demean and detrend
    gf.comp_1st.acc -= gf.comp_1st.acc.mean()
    gf.comp_2nd.acc -= gf.comp_2nd.acc.mean()
    gf.comp_up.acc -= gf.comp_up.acc.mean()

    gf.comp_1st.acc = signal.detrend(gf.comp_1st.acc, type="linear")
    gf.comp_2nd.acc = signal.detrend(gf.comp_2nd.acc, type="linear")
    gf.comp_up.acc = signal.detrend(gf.comp_up.acc, type="linear")

    features, add_data = gm.features.get_features(gf, ko_matrices=konno_matrices)

    return features, add_data


def main(
    record_dir: str,
    event_list_ffp: str = None,
    ko_matrices_dir: str = None,
    low_mem_usage: bool = False,
):
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
            [os.path.basename(record_ffp).split("_")[0] for record_ffp in record_files],
            dtype=str,
        )
        record_files = record_files[np.isin(record_events, events_filter)]

    konno_matrices = None
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
            matrix_id: gm.features.get_konno_matrix(matrix_id * 2, dt=0.005)
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

    process_rf(record_files[0], konno_matrices=konno_matrices)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "record_dir",
        type=str,
        help="Root directory for the records, will search for records recursively from here",
    )
    parser.add_argument(
        "--event_list_ffp",
        type=str,
        help="Path to file that list all events of interest (one per line), if None (default) all found records are used",
        default=None,
    )
    parser.add_argument(
        "--ko_matrices_dir",
        type=str,
        help="Path to the directory that contains the Konno matrices. "
        "Has to be specified if the --low_memory options is used",
        default=None,
    )
    parser.add_argument(
        "--low_memory",
        action="store_true",
        help="If specified will prioritise low memory usage over performance. "
        "Requires --ko_matrices_dir to be specified. ",
        default=False,
    )

    args = parser.parse_args()

    main(
        args.record_dir,
        args.event_list_ffp,
        ko_matrices_dir=args.ko_matrices_dir,
        low_mem_usage=args.low_memory,
    )
