"""Trains and evaluates a model that takes the features from all
three components for a specific record and estimates a score & f_min
for each of the components"""
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

import gm_classifier as gm

# ----- Config -----
# label_dir = "/Users/Clus/code/work/gm_classifier/data/records/training_data/iter"
label_dir = (
    "/Users/Clus/code/work/gm_classifier/data/records/local/training_data/labels"
)

# features_dir = "/Users/Clus/code/work/gm_classifier/data/records/training_data/all_records_features/200401"
features_dir = "/Users/Clus/code/work/gm_classifier/data/records/local/training_data/all_records_features/200401"

output_dir = "/Users/Clus/code/work/tmp/gm_classifier/record_based/tmp_1"

label_config = {
    "score_X": None,
    "f_min_X": None,
    "score_Y": None,
    "f_min_Y": None,
    "score_Z": None,
    "f_min_Z": None,
}
label_names = list(label_config.keys())

feature_config = {
    "signal_pe_ratio_max": ["standard", "whiten"],
    "signal_ratio_max": ["standard", "whiten"],
    "snr_min": ["standard", "whiten"],
    "snr_max": ["standard", "whiten"],
    "snr_average": ["standard", "whiten"],
    "max_tail_ratio": ["standard", "whiten"],
    "average_tail_ratio": ["standard", "whiten"],
    "max_head_ratio": ["standard", "whiten"],
    "snr_average_0.1_0.2": ["standard", "whiten"],
    "snr_average_0.2_0.5": ["standard", "whiten"],
    "snr_average_0.5_1.0": ["standard", "whiten"],
    "snr_average_1.0_2.0": ["standard", "whiten"],
    "snr_average_2.0_5.0": ["standard", "whiten"],
    "snr_average_5.0_10.0": ["standard", "whiten"],
    "fas_ratio_low": ["standard", "whiten"],
    "fas_ratio_high": ["standard", "whiten"],
    "pn_pga_ratio": ["standard", "whiten"],
    "is_vertical": None,
}
feature_config_X = {f"{key}_X": val for key, val in feature_config.items()}
feature_config_Y = {f"{key}_Y": val for key, val in feature_config.items()}
feature_config_Z = {f"{key}_Z": val for key, val in feature_config.items()}
feature_config = {**feature_config_X, **{**feature_config_Y, **feature_config_Z}}
feature_names = [key for key in feature_config.keys() if not "*" in key]

snr_features = [
    f"snr_value_{freq:.3f}"
    for freq in np.logspace(np.log(0.05), np.log(20), 50, base=np.e)
]

# snr_features = list(
#     np.concatenate(
#         (
#             np.char.add(snr_features, "_X"),
#             np.char.add(snr_features, "_Y"),
#             np.char.add(snr_features, "_Z"),
#         )
#     )
# )

# feature_names = feature_names + snr_features

# Model config
dropout_rate = 0.4
model_config = {
    "dense_layer_config": [
        (keras.layers.Dense, {"units": 32, "activation": "relu"}),
        (keras.layers.Dense, {"units": 16, "activation": "relu"}),
    ],
    "dense_input_name": "features",
    "cnn_layer_config": [
        (
            keras.layers.Conv1D,
            {
                "filters": 32,
                "kernel_size": 15,
                "strides": 1,
                "activation": "relu",
                "padding": "same",
            },
        ),
        (keras.layers.MaxPooling1D, {"pool_size": 2}),
        (keras.layers.Dropout, {"rate": dropout_rate}),
        (
            keras.layers.Conv1D,
            {
                "filters": 16,
                "kernel_size": 5,
                "strides": 1,
                "activation": "relu",
                "padding": "same",
            },
        ),
        # (keras.layers.MaxPooling1D, {"pool_size": 2}),
        (keras.layers.Dropout, {"rate": dropout_rate}),
    ],
    "cnn_input_name": "snr_series",
    "comb_layer_config": [
        (keras.layers.Dense, {"units": 128, "activation": "relu"}),
        (keras.layers.Dropout, {"rate": dropout_rate}),
        (keras.layers.Dense, {"units": 42, "activation": "relu"}),
        (keras.layers.Dropout, {"rate": dropout_rate}),
        (keras.layers.Dense, {"units": 16, "activation": "relu"}),
    ],
    "n_outputs": 6,
}

# Training details
optimizer = "Adam"
loss = "mse"
val_size = 0.1
n_epochs = 250
batch_size = 32

output_dir = Path(output_dir)

# ---- Training ----
label_df = gm.utils.load_labels_from_dir(label_dir)
feature_df = gm.utils.load_features_from_dir(features_dir)

train_df = pd.merge(
    feature_df, label_df, how="inner", left_index=True, right_index=True
)
train_df.to_csv(output_dir / "training_data.csv")

# Some sanity checks
assert np.all(
    np.isin(feature_names, feature_df.columns.values)
), "Not all features are in the feature dataframe"

# Data preparation
X_features = train_df.loc[:, feature_names]

# Bit of hack atm, need to update
X_snr = np.stack(
    (
        train_df.loc[:, np.char.add(snr_features, "_X")].values,
        train_df.loc[:, np.char.add(snr_features, "_Y")].values,
        train_df.loc[:, np.char.add(snr_features, "_Z")].values,
    ),
    axis=2,
)

ids = train_df.index.values
y = train_df.loc[:, label_names]

# Split into training and validation
X_features_train, X_features_val, X_snr_train, X_snr_val, y_train, y_val, ids_train, ids_val = train_test_split(
    X_features, X_snr, y, ids, test_size=val_size
)

# Pre-processing
X_features_train, X_features_val = gm.training.apply_pre(
    X_features_train.copy(),
    feature_config,
    output_dir,
    val_data=X_features_val.copy(),
    output_prefix="features",
)
X_snr_train, X_snr_val = np.log(X_snr_train), np.log(X_snr_val)

# Run training of the model
compile_kwargs = {"optimizer": optimizer, "loss": loss}
fit_kwargs = {"batch_size": batch_size, "epochs": n_epochs, "verbose": 2}
history, gm_model = gm.training.train(
    output_dir,
    gm.model.CnnSnrModel,
    model_config,
    (
        {"features": X_features_train.values, "snr_series": X_snr_train},
        y_train.values,
        ids_train,
    ),
    val_data=(
        {"features": X_features_val.values, "snr_series": X_snr_val},
        y_val.values,
        ids_val,
    ),
    compile_kwargs=compile_kwargs,
    fit_kwargs=fit_kwargs,
)

# Save the config
config = {
    "feature_config": str(feature_config),
    "model_config": str(model_config),
    "compiler_kwargs": str(compile_kwargs),
    "fit_kwargs": str(fit_kwargs),
}
with open(output_dir / "config.json", "w") as f:
    json.dump(config, f)

# ---- Evaluation ----
fig_size = (16, 10)

fig, ax = gm.plots.plot_loss(
    history,
    # output_ffp=str(output_dir / "loss_plot.png"),
    fig_kwargs={"figsize": fig_size},
)
ax.set_ylim((0, 1))
plt.savefig(str(output_dir / "loss_plot.png"))
plt.close()

# Load the best model
gm_model.load_weights(str(output_dir / "model.h5"))

# Predict train and validation
y_train_est = gm_model.predict(
    {"features": X_features_train.values, "snr_series": X_snr_train}
)
y_val_est = gm_model.predict(
    {"features": X_features_val.values, "snr_series": X_snr_val}
)

for cur_comp, (score_ix, f_min_ix) in zip(["X", "Y", "Z"], [(0, 1), (2, 3), (4, 5)]):
    # for cur_comp, score_ix in zip(["X", "Y", "Z"], [0, 1, 2]):
    # Plot true vs estimated
    score_min, score_max = (
        np.min(train_df[f"score_{cur_comp}"]),
        np.max(train_df[f"score_{cur_comp}"]),
    )
    fig, ax = gm.plots.plot_true_vs_est(
        y_train_est[:, score_ix],
        y_train.iloc[:, score_ix]
        + np.random.normal(0, 0.01, y_train.iloc[:, score_ix].size),
        y_val_est=y_val_est[:, score_ix],
        y_val_true=y_val.iloc[:, score_ix]
        + np.random.normal(0, 0.01, y_val.iloc[:, score_ix].size),
        title="Score",
        min_max=(score_min, score_max),
        scatter_kwargs={"s": 2.0},
        fig_kwargs={"figsize": fig_size},
        output_ffp=output_dir / f"score_true_vs_est_{cur_comp}.png",
    )

    score_res_train = y_train.iloc[:, score_ix] - y_train_est[:, score_ix]
    score_res_val = y_val.iloc[:, score_ix] - y_val_est[:, score_ix]
    fig, ax = gm.plots.plot_residual(
        score_res_train,
        y_train.iloc[:, score_ix]
        + +np.random.normal(0, 0.01, y_train.iloc[:, score_ix].size),
        score_res_val,
        y_val.iloc[:, score_ix]
        + np.random.normal(0, 0.01, y_val.iloc[:, score_ix].size),
        min_max=(score_min, score_max),
        title="Score residual",
        x_label="Score",
        scatter_kwargs={"s": 2.0},
        fig_kwargs={"figsize": fig_size},
        output_ffp=output_dir / f"score_res_{cur_comp}.png",
    )

    f_min_min, f_min_max = (
        np.min(train_df[f"f_min_{cur_comp}"]),
        np.max(train_df[f"f_min_{cur_comp}"]),
    )
    fig, ax = gm.plots.plot_true_vs_est(
        y_train_est[:, f_min_ix],
        y_train.iloc[:, f_min_ix]
        + np.random.normal(0, 0.025, y_train.iloc[:, f_min_ix].size),
        y_val_est=y_val_est[:, f_min_ix],
        y_val_true=y_val.iloc[:, f_min_ix]
        + np.random.normal(0, 0.025, y_val.iloc[:, f_min_ix].size),
        title="f_min",
        min_max=(f_min_min, f_min_max),
        scatter_kwargs={"s": 2.0},
        fig_kwargs={"figsize": fig_size},
        output_ffp=output_dir / f"f_min_true_vs_est_{cur_comp}.png",
    )

    fig, ax = gm.plots.plot_true_vs_est(
        y_train_est[:, f_min_ix],
        y_train.iloc[:, f_min_ix]
        + np.random.normal(0, 0.025, y_train.iloc[:, f_min_ix].size),
        y_val_est=y_val_est[:, f_min_ix],
        y_val_true=y_val.iloc[:, f_min_ix]
        + np.random.normal(0, 0.025, y_val.iloc[:, f_min_ix].size),
        title="f_min",
        min_max=(f_min_min, f_min_max),
        scatter_kwargs={"s": 2.0},
        fig_kwargs={"figsize": fig_size},
        # output_ffp=output_dir / "f_min_true_vs_est.png"
    )
    ax.set_xlim((0.0, 2.0))
    ax.set_ylim((0.0, 2.0))
    plt.savefig(output_dir / f"f_min_true_vs_est_zoomed_{cur_comp}.png")
    plt.close()

    f_min_res_train = y_train.iloc[:, f_min_ix] - y_train_est[:, f_min_ix]
    f_min_res_val = y_val.iloc[:, f_min_ix] - y_val_est[:, f_min_ix]
    fig, ax = gm.plots.plot_residual(
        f_min_res_train,
        y_train.iloc[:, f_min_ix]
        + np.random.normal(0, 0.025, y_train.iloc[:, f_min_ix].size),
        f_min_res_val,
        y_val.iloc[:, f_min_ix]
        + np.random.normal(0, 0.025, y_val.iloc[:, f_min_ix].size),
        min_max=(f_min_min, f_min_max),
        title="f_min residual",
        x_label="f_min",
        scatter_kwargs={"s": 2.0},
        fig_kwargs={"figsize": fig_size},
        output_ffp=output_dir / f"f_min_res_{cur_comp}.png",
    )
