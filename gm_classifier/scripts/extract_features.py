import os
import argparse
import traceback
from typing import Dict, Any, List, Union

import gm_classifier as gmc


def main(
    output_dir: str,
    record_dir: str,
    record_format: str,
    output_prefix: str = "features",
    record_list_ffp: str = None,
    ko_matrices_dir: str = None,
    low_mem_usage: bool = False,
):
    (
        feature_df_1,
        feature_df_2,
        feature_df_v,
        failed_records,
    ) = gmc.records.process_records(
        gmc.records.RecordFormat(record_format),
        record_dir=record_dir,
        record_list_ffp=record_list_ffp,
        ko_matrices_dir=ko_matrices_dir,
        low_mem_usage=low_mem_usage,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )

    log_failed_records(output_dir, failed_records)
    return


def log_failed_records(
    output_dir: str,
    failed_records: Dict[Union[Exception, str], Dict[Union[Exception, str], List]],
):
    output_dir = os.path.join(output_dir, f"failed_records_{gmc.utils.create_run_id()}")
    os.mkdir(output_dir)

    feature_errors = failed_records[gmc.features.FeatureError]
    record_errors = failed_records[gmc.records.RecordError]

    if len(record_errors[gmc.records.RecordErrorType.Duration]) > 0:
        with open(os.path.join(output_dir, "record_duration.txt"), "w") as f:
            f.write(
                "The following records failed processing due to "
                "the record length < 5 seconds:\n {}".format(
                    "\n".join(record_errors[gmc.records.RecordErrorType.TotalTime])
                )
            )
    if len(record_errors[gmc.records.RecordErrorType.CompsNotMatching]) > 0:
        with open(os.path.join(output_dir, "components_lengths.txt"), "w") as f:
            f.write(
                "The following records failed processing due to the "
                "acceleration timeseries of the components having"
                "different length:\n{}".format(
                    "\n".join(
                        record_errors[gmc.records.RecordErrorType.CompsNotMatching]
                    )
                )
            )
    if len(record_errors[gmc.records.RecordErrorType.NQuakePoints]) > 0:
        with open(os.path.join(output_dir, "datapoints.txt"), "w") as f:
            f.write(
                "The following records failed processing due to the "
                "time delay adjusted timeseries having less than "
                "10 datapoints:\n{}".format(
                    "\n".join(record_errors[gmc.records.RecordErrorType.NQuakePoints])
                )
            )
    if len(record_errors[gmc.records.RecordErrorType.MissinResponseInfo]) > 0:
        with open(os.path.join(output_dir, "missing_response.txt"), "w") as f:
            f.write(
                "The following records failed due to missing response information:\n{}".format(
                    "\n".join(
                        record_errors[gmc.records.RecordErrorType.MissinResponseInfo]
                    )
                )
            )
    if len(failed_records["empty_file"]) > 0:
        with open(os.path.join(output_dir, "empty.txt"), "w") as f:
            f.write(
                "The following records failed processing due to the "
                "geoNet file not containing any data:\n{}".format(
                    "\n".join(failed_records["empty_file"])
                )
            )
    if len(failed_records["other"]):
        with open(os.path.join(output_dir, "unknown.txt"), "w") as f:
            f.write(
                "The following records failed processing due to an unknown exception:\n"
            )
            for cur_record, cur_ex, cur_tb in failed_records["other"]:
                f.write(f"{cur_record}, Traceback:\n")
                traceback.print_tb(cur_tb, file=f)
                f.write(cur_ex.__repr__())
                f.write("\n--------------------------------------------------------\n")
                f.write("\n")

    if len(feature_errors[gmc.features.FeatureErrorType.PGA_zero]) > 0:
        with open(os.path.join(output_dir, "pga_zero.txt"), "w") as f:
            f.write(
                "The following records failed processing due to "
                "one PGA being zero for one (or more) of the components:\n{}".format(
                    "\n".join(feature_errors[gmc.features.FeatureErrorType.PGA_zero])
                )
            )
    if len(feature_errors[gmc.features.FeatureErrorType.early_p_pick]) > 0:
        with open(os.path.join(output_dir, "early_p_pick.txt"), "w") as f:
            f.write(
                "The following records failed processing as the p-wave pick is < 2.5 from the"
                " start of the record, preventing accurate feature generation:\n{}".format(
                    "\n".join(
                        feature_errors[gmc.features.FeatureErrorType.early_p_pick]
                    )
                )
            )
    if len(feature_errors[gmc.features.FeatureErrorType.short_signal_duration]) > 0:
        with open(os.path.join(output_dir, "short_signal_duration.txt"), "w") as f:
            f.write(
                "The following records failed processing as the signal duration is less than 10.24s"
                ", preventing accurate feature generation:\n{}".format(
                    "\n".join(feature_errors[gmc.features.FeatureErrorType.short_signal_duration])
                )
            )
    if len(feature_errors[gmc.features.FeatureErrorType.missing_ko_matrix]) > 0:
        with open(os.path.join(output_dir, "missing_ko_matrix.txt"), "w") as f:
            f.write(
                "The following records failed processing as "
                "no Konno matrix of the requred size was available:\n{}".format(
                    "\n".join(
                        feature_errors[gmc.features.FeatureErrorType.missing_ko_matrix]
                    )
                )
            )

    return output_dir


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
        choices=["V1A", "mseed", "csv"],
        help="Format of the records, either V1A, csv or mseed",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        help="Prefix for the output files",
        default="features",
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
        record_list_ffp=args.record_list_ffp,
        ko_matrices_dir=args.ko_matrices_dir,
        low_mem_usage=args.low_memory,
    )
