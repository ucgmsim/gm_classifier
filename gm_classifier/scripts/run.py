"""Script for running end-to-end prediction from GeoNet record files"""
import os
import argparse

import gm_classifier as gm


def run_e2e(
    record_dir: str,
    model: str,
    output_ffp: str,
    event_list_ffp: str = None,
    ko_matrices_dir: str = None,
    low_memory: bool = None,
    model_name: str = None,
):
    input_df = gm.records.process_records(
        record_dir,
        event_list_ffp,
        ko_matrices_dir=ko_matrices_dir,
        low_mem_usage=low_memory,
    )

    if os.path.isdir(model):
        if model_name is None:
            raise ValueError(
                "If a directory for an original model is given, "
                "then the model name has to be specified"
            )

        result_df = gm.predict.run_original(model, model_name, input_df)
        result_df.to_csv(output_ffp)
    else:
        raise NotImplementedError

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "record_dir",
        type=str,
        help="Root directory for the records, "
        "will search for records recursively from here",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Either the path to a saved keras model "
        "or path to an original model directory",
    )
    parser.add_argument("output_ffp", type=str, help="Output csv path")
    parser.add_argument(
        "--event_list_ffp",
        type=str,
        help="Path to file that list all events of interest (one per line), "
        "if None (default) all found records are used",
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
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["canterbury", "canterbury_wellington"],
        default=None,
        help="If an original model is used, then the name has to be set"
        " here to allow for correct pre-processing to be used",
    )
    args = parser.parse_args()

    run_e2e(
        args.record_dir,
        args.model,
        args.output_ffp,
        event_list_ffp=args.event_list_ffp,
        ko_matrices_dir=args.ko_matrices_dir,
        low_memory=args.low_memory,
        model_name=args.model_name,
    )
