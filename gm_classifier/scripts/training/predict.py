import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

# Grow the GPU memory usage as needed
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import gm_classifier as gmc
from gm_classifier.src.console import console


def main(features_dir: Path, model_dir: Path, output_ffp: Path, n_preds: int = 25):
    console.print("Loading data")
    scalar_feature_config = gmc.utils.load_yaml(model_dir / "feature_config.yaml")
    snr_feature_names = [
        f"snr_value_{freq:.3f}"
        for freq in np.logspace(np.log(0.01), np.log(25), 100, base=np.e)
    ]

    feature_df = gmc.utils.load_features_from_dir(features_dir, concat=True)
    X_scalar = feature_df.loc[:, list(scalar_feature_config.keys())]
    X_snr = feature_df.loc[:, snr_feature_names]

    console.print("Pre-processing")
    pre_params = gmc.utils.load_picke(model_dir / "pre_params.pickle")
    gmc.pre.run_preprocessing(X_scalar, scalar_feature_config, params=pre_params)
    X_snr = X_snr.apply(np.log).values[..., None]

    console.print("Loading the model")
    gmc_model = keras.models.load_model(model_dir, compile=False)

    console.print("Running predictions")
    (
        y_score_est,
        y_score_est_std,
        y_fmin_est,
        y_fmin_est_std,
        y_multi_est,
        y_multi_est_std,
    ) = gmc.eval.get_combined_prediction(
        gmc_model,
        X_scalar,
        X_snr,
        n_preds,
        index=X_scalar.index.values.astype(str),
        multi_output=True,
    )

    result_df = pd.concat([y_score_est, y_score_est_std, y_fmin_est, y_fmin_est_std, y_multi_est, y_multi_est_std], axis=1)
    result_df["record"] = np.stack(np.char.rsplit(feature_df.index.values.astype(str), "_", 1))[:, 0]
    result_df["component"] = np.stack(np.char.rsplit(feature_df.index.values.astype(str), "_", 1))[:, -1]

    # Convert to record per row
    # z_df = results_df.loc[results_df.component == "Z"]
    # z_df.columns = np.char.add(z_df.columns.values.astype(str), "_Z")
    # results_df = pd.merge(pd.merge(results_df.loc[results_df.component == "X"], results_df.loc[results_df.component == "Y"], suffixes=("_X", "_Y"), left_on="record", right_on="record").set_index("record"),
    #          z_df, left_index=True, right_on="record_Z").set_index("record_Z")
    # results_df = results_df.drop(columns=["component_X", "component_Y", "component_Z"])

    result_df.to_csv(output_ffp, index_label="record_id")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "features_dir",
        type=Path,
        help="Path of directory that contains the feature files",
    )
    parser.add_argument(
        "model_dir", type=Path, help="Path of directory that contains the GMC model"
    )
    parser.add_argument("output_ffp", type=Path, help="File path for output csv")

    args = parser.parse_args()

    main(args.features_dir, args.model_dir, args.output_ffp)