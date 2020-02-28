import copy
import argparse

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


def main(
    output_ffp: str,
    record_dir: str,
    event_list_ffp: str = None,
    record_list_ffp: str = None,
    ko_matrices_dir: str = None,
    low_mem_usage: bool = False,
):
    feature_df = gm.records.process_records(
        record_dir,
        event_list_ffp=event_list_ffp,
        record_list_ffp=record_list_ffp,
        ko_matrices_dir=ko_matrices_dir,
        low_mem_usage=low_mem_usage,
    )
    feature_df.to_csv(output_ffp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "output_ffp", type=str, help="File path for the resulting csv file"
    )
    parser.add_argument(
        "record_dir",
        type=str,
        help="Root directory for the records, "
        "will search for records recursively from here",
    )
    parser.add_argument(
        "--event_list_ffp",
        type=str,
        help="Path to file that lists all events to use (one per line). "
        "Note: in order to be able to use event filtering, the path from the "
        "record_dir has to include a folder with the event id as its name. "
        "Formats of event ids: just a number or "
        "XXXXpYYYYYY (where XXXX is a valid year)",
        default=None,
    )
    parser.add_argument(
        "--record_list_ffp",
        type=str,
        help="Path to file that lists all records to use (one per line)",
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
        args.output_ffp,
        args.record_dir,
        event_list_ffp=args.event_list_ffp,
        record_list_ffp=args.record_list_ffp,
        ko_matrices_dir=args.ko_matrices_dir,
        low_mem_usage=args.low_memory,
    )
