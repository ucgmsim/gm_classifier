import os
import argparse

import numpy as np
import pandas as pd

import gm_classifier as gm


def run(
    model: str, input_data_ffp: str, output_ffp: str, model_name: str = None
) -> None:
    if os.path.isdir(model):
        if model_name is None:
            raise ValueError(
                "If a directory for an original model is given, "
                "then the model name has to be specified"
            )

        input_df = pd.read_csv(input_data_ffp)
        result_df = gm.predict.run_original(model, model_name, input_df)
        result_df.to_csv(output_ffp)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help="Either the path to a saved keras model "
        "or path to an original model directory",
    )
    parser.add_argument(
        "input_data_ffp", type=str, help="Path to the csv with the input data"
    )
    parser.add_argument("output_ffp", type=str, help="Output csv path")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["canterbury", "canterbury_wellington"],
        default=None,
        help="If an original model is used, then the name has to be set"
        " here to allow for correct pre-processing to be used",
    )
    args = parser.parse_args()

    run(args.model, args.input_data_ffp, args.output_ffp, model_name=args.model_name)
