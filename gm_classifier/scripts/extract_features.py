import argparse

import gm_classifier as gm


def main(
    output_dir: str,
    record_dir: str,
    output_prefix: str = "features",
    event_list_ffp: str = None,
    record_list_ffp: str = None,
    ko_matrices_dir: str = None,
    low_mem_usage: bool = False,
):
    feature_df, failed_records = gm.records.process_records(
        record_dir,
        event_list_ffp=event_list_ffp,
        record_list_ffp=record_list_ffp,
        ko_matrices_dir=ko_matrices_dir,
        low_mem_usage=low_mem_usage,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )

    gm.records.print_errors(failed_records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("output_dir", type=str, help="Path to the output directory")
    parser.add_argument(
        "record_dir",
        type=str,
        help="Root directory for the records, "
        "will search for V1A records recursively from here",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        help="Prefix for the output files",
        default="features",
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
        args.output_dir,
        args.record_dir,
        output_prefix=args.output_prefix,
        event_list_ffp=args.event_list_ffp,
        record_list_ffp=args.record_list_ffp,
        ko_matrices_dir=args.ko_matrices_dir,
        low_mem_usage=args.low_memory,
    )
