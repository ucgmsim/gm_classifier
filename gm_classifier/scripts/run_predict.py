import os
import argparse

import pandas as pd

import gm_classifier as gm


def run(model: str, input_data_ffp: str, output_ffp: str) -> None:
    input_df = pd.read_csv(input_data_ffp)
    if model.strip().lower() in ["canterbury", "canterbury_wellington"]:
        result_df = gm.classify.classify_original(model, input_df)
        result_df.to_csv(output_ffp)
    elif os.path.isfile(model):
        gm.classify.classify(model, input_df)
    else:
        raise ValueError(
            "Argument models has to either be a path to "
            "a saved keras model or the name of an original "
            "model (i.e. one of ['canterbury', 'canterbury_wellington'])"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help="Either directory of saved model (and the pre-processing information) "
        "or the name of an original model (i.e. ['canterbury', 'canterbury_wellington']",
    )
    parser.add_argument(
        "input_data_ffp", type=str, help="Path to the csv with the input data"
    )
    parser.add_argument("output_ffp", type=str, help="Output csv path")
    args = parser.parse_args()

    run(args.model, args.input_data_ffp, args.output_ffp)
