"""Trains and evaluates a component based model that
 predicts a usability score and a f_min frequency

 Note: The input of the model is a single component of a
 record and the output a score & f_min for that
 component of the record
 """
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

import gm_classifier as gm

# ----- Config -----
label_dir = "/Users/Clus/code/work/gm_classifier/data/records/training_data/iter"
features_dir = "/Users/Clus/code/work/gm_classifier/data/records/training_data/all_records_features/200401"

output_dir = "/Users/Clus/code/work/tmp/gm_classifier/tmp_2"

label_names = ["score", "f_min"]

feature_names = [
    "signal_pe_ratio_max",
    "signal_ratio_max",
    "snr_min",
    "snr_max",
    "snr_average",
    "max_tail_ratio",
    "average_tail_ratio",
    "max_head_ratio",
    "snr_average_0.1_0.2",
    "snr_average_0.2_0.5",
    "snr_average_0.5_1.0",
    "snr_average_1.0_2.0",
    "snr_average_2.0_5.0",
    "snr_average_5.0_10.0",
    "fas_ratio_low",
    "fas_ratio_high",
    "pn_pga_ratio",
]
snr_features = [f"snr_value_{freq:.3f}" for freq in np.logspace(np.log(0.05), np.log(20), 50, base=np.e)]
feature_names = feature_names + snr_features

# Model details
model_config = {
    "units": [120, 60, 30],
    "act_funcs": "relu",
    "n_outputs": 2,
    "output_act_func": "linear",
    "output_names": ["score", "f_min"],
    "dropout": 0.5,
}

feature_pre_config = {"standardise": True, "whiten": True}
label_pre_config = {"shift": [1, 0]}

# Training details
optimizer = "Adam"
loss = "mse"
val_size = 0.2
n_epochs = 300
batch_size = 32

output_dir = Path(output_dir)

# ---- Training ----
label_df = gm.utils.load_comp_labels_from_dir(
    label_dir
)
feature_df = gm.utils.load_comp_features_from_dir(
    features_dir
)

train_df = pd.merge(
    feature_df, label_df, how="inner", left_index=True, right_index=True
)
train_df.to_csv(output_dir / "training_data.csv")

# Some sanity checks
assert np.all(
    np.isin(feature_names, feature_df.columns.values)
), "Not all features are in the feature dataframe"

# Get indices for splitting into training & validation set
train_ind, val_ind = train_test_split(
    np.arange(train_df.shape[0], dtype=int), test_size=val_size
)

# Create training and validation datasets
X_train = train_df.loc[:, feature_names].iloc[train_ind].values
X_val = train_df.loc[:, feature_names].iloc[val_ind].values

y_train = train_df.loc[:, label_names].iloc[train_ind].values
y_val = train_df.loc[:, label_names].iloc[val_ind].values

ids_train = train_df.index.values[train_ind]
ids_val = train_df.index.values[val_ind]

train_data = (X_train, y_train, ids_train)
val_data = (X_val, y_val, ids_val)

# Run training of the model
history, X_train, X_val, y_train_proc, y_val_proc = gm.training.fit(
    output_dir,
    feature_pre_config,
    model_config,
    train_data,
    val_data=val_data,
    label_config=label_pre_config,
    compile_kwargs={"optimizer": optimizer, "loss": loss},
    fit_kwargs={"batch_size": batch_size, "epochs": n_epochs, "verbose": 2},
)

# ---- Evaluation ----
fig_size = (16, 10)

fig, ax = gm.plots.plot_loss(
    history, output_ffp=str(output_dir / "loss_plot.png"), fig_kwargs={"figsize": fig_size}
)

# Load the best model
model = keras.models.load_model(output_dir / "model.h5")

# Predict (also does the reverse transform)
y_train_est = gm.predict.predict(output_dir, X_train, feature_config=None, label_config=label_pre_config)
y_val_est = gm.predict.predict(output_dir, X_val, feature_config=None, label_config=label_pre_config)

# Plot true vs estimated
score_min, score_max = np.min(train_df.score), np.max(train_df.score)
fig, ax = gm.plots.plot_true_vs_est(
    y_train_est[:, 0],
    y_train[:, 0] + np.random.normal(0, 0.01, y_train[:, 0].size),
    y_val_est=y_val_est[:, 0],
    y_val_true=y_val[:, 0] + np.random.normal(0, 0.01, y_val[:, 0].size),
    title="Score",
    min_max=(score_min, score_max),
    scatter_kwargs={"s": 2.0},
    fig_kwargs={"figsize": fig_size},
    output_ffp=output_dir / "score_true_vs_est.png"
)


score_res_train = y_train[:, 0] - y_train_est[:, 0]
score_res_val = y_val[:, 0] - y_val_est[:, 0]
fig, ax = gm.plots.plot_residual(
    score_res_train,
    y_train[:, 0] + + np.random.normal(0, 0.01, y_train[:, 0].size),
    score_res_val,
    y_val[:, 0] + np.random.normal(0, 0.01, y_val[:, 0].size),
    min_max=(score_min, score_max),
    title="Score residual",
    x_label="Score",
    scatter_kwargs={"s": 2.0},
    fig_kwargs={"figsize": fig_size},
    output_ffp=output_dir / "score_res.png"
)


f_min_min, f_min_max = np.min(train_df.f_min), np.max(train_df.f_min)
fig, ax = gm.plots.plot_true_vs_est(
    y_train_est[:, 1],
    y_train[:, 1] + np.random.normal(0, 0.025, y_train[:, 1].size),
    y_val_est=y_val_est[:, 1],
    y_val_true=y_val[:, 1] + np.random.normal(0, 0.025, y_val[:, 1].size),
    title="f_min",
    min_max=(f_min_min, f_min_max),
    scatter_kwargs={"s": 2.0},
    fig_kwargs={"figsize": fig_size},
    output_ffp=output_dir / "f_min_true_vs_est.png"
)

f_min_min, f_min_max = np.min(train_df.f_min), np.max(train_df.f_min)
fig, ax = gm.plots.plot_true_vs_est(
    y_train_est[:, 1],
    y_train[:, 1] + np.random.normal(0, 0.025, y_train[:, 1].size),
    y_val_est=y_val_est[:, 1],
    y_val_true=y_val[:, 1] + np.random.normal(0, 0.025, y_val[:, 1].size),
    title="f_min",
    min_max=(f_min_min, f_min_max),
    scatter_kwargs={"s": 2.0},
    fig_kwargs={"figsize": fig_size},
    # output_ffp=output_dir / "f_min_true_vs_est.png"
)
ax.set_xlim((0.0, 2.0))
ax.set_ylim((0.0, 2.0))
plt.savefig(output_dir / "f_min_true_vs_est_zoomed.png")
plt.close()

f_min_res_train = y_train[:, 1] - y_train_est[:, 1]
f_min_res_val = y_val[:, 1] - y_val_est[:, 1]
fig, ax = gm.plots.plot_residual(
    f_min_res_train,
    y_train[:, 1] + np.random.normal(0, 0.025, y_train[:, 1].size),
    f_min_res_val,
    y_val[:, 1] + np.random.normal(0, 0.025, y_val[:, 1].size),
    min_max=(f_min_min, f_min_max),
    title="f_min residual",
    x_label="f_min",
    scatter_kwargs={"s": 2.0},
    fig_kwargs={"figsize": fig_size},
    output_ffp=output_dir / "f_min_res.png"
)

exit()
