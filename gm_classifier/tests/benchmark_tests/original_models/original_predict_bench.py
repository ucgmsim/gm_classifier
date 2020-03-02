import os

import pytest
import pandas as pd
import numpy as np

import gm_classifier as gm

file_dir = os.path.dirname(__file__)
original_models_dir = os.path.join(file_dir, "../../../original_models")


@pytest.mark.parametrize(
    ["model_name", "input_data_ffp"],
    [
        (
            "canterbury",
            os.path.join(file_dir, "bench_canterbury_data.csv"),
        ), (
            "canterbury_wellington",
            os.path.join(file_dir, "bench_canterbury_wellington_data.csv"),
        )
    ],
)
def test_orig_predict(model_name: str, input_data_ffp: str):
    # Contains both the input data and the expected output
    bench_df = pd.read_csv(input_data_ffp)
    result = gm.predict.classify_original(
        model_name, bench_df.loc[:, gm.features.FEATURE_NAMES]
    )

    assert np.all(
        np.isclose(
            bench_df.loc[:, ["yhat_low", "yhat_high"]].values,
            result.loc[:, ["y_low", "y_high"]],
        )
    )
