import os
import argparse

import gm_classifier as gm


def main(
    output_dir: str,
    record_dir: str,
    record_format: str,
    output_prefix: str = "features",
    event_list_ffp: str = None,
    record_list_ffp: str = None,
    ko_matrices_dir: str = None,
    low_mem_usage: bool = False,
):
    (
        feature_df_1,
        feature_df_2,
        feature_df_v,
        failed_records,
    ) = gm.records.process_records(
        gm.records.RecordFormat.V1A
        if record_format == "V1A"
        else gm.records.RecordFormat.MiniSeed,
        record_dir=record_dir,
        event_list_ffp=event_list_ffp,
        record_list_ffp=record_list_ffp,
        ko_matrices_dir=ko_matrices_dir,
        low_mem_usage=low_mem_usage,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )

    error_log = gm.records.get_records_error_log(failed_records)
    print(error_log)

    with open(os.path.join(output_dir, f"error_log_{gm.utils.create_run_id()}.txt"), "w") as f:
        f.write(error_log)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("output_dir", type=str, help="Path to the output directory")
    parser.add_argument(
        "record_dir",
        type=str,
        help="Root directory for the records, "
        "will search for V1A or mseed records recursively from here",
    )
    parser.add_argument(
        "record_format",
        type=str,
        choices=["V1A", "mseed"],
        help="Format of the records, either V1A or mseed",
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
        args.record_format,
        output_prefix=args.output_prefix,
        event_list_ffp=args.event_list_ffp,
        record_list_ffp=args.record_list_ffp,
        ko_matrices_dir=args.ko_matrices_dir,
        low_mem_usage=args.low_memory,
    )
