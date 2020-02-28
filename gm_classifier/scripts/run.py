"""Script for running end-to-end prediction from GeoNet record files"""
import argparse

import gm_classifier as gm


def run_e2e(
    record_dir: str,
    model: str,
    output_ffp: str,
    event_list_ffp: str = None,
    record_list_ffp: str = None,
    ko_matrices_dir: str = None,
    low_memory: bool = None,
):
    input_df = gm.records.process_records(
        record_dir,
        event_list_ffp=event_list_ffp,
        record_list_ffp=record_list_ffp,
        ko_matrices_dir=ko_matrices_dir,
        low_mem_usage=low_memory,
    )

    print("Running classification")
    if model.strip().lower() in ["canterbury", "canterbury_wellington"]:
        result_df = gm.predict.run_original(model, input_df)
        result_df.to_csv(
            output_ffp,
        )
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
        "or the name of an original model (i.e. ['canterbury', 'canterbury_wellington']",
    )
    parser.add_argument("output_ffp", type=str, help="Output csv path")
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

    run_e2e(
        args.record_dir,
        args.model,
        args.output_ffp,
        event_list_ffp=args.event_list_ffp,
        record_list_ffp=args.record_list_ffp,
        ko_matrices_dir=args.ko_matrices_dir,
        low_memory=args.low_memory,
    )
