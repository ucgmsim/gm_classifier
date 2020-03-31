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
from tensorflow import keras
from sklearn.model_selection import train_test_split

import gm_classifier as gm

# ----- Config -----
label_dir = "/Users/Clus/code/work/gm_classifier/data/records/training_data/iter"
features_dir = "/Users/Clus/code/work/gm_classifier/data/records/training_data/all_records_features"

output_dir = "/Users/Clus/code/work/tmp/gm_classifier/tmp"

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

# Model details
model_config = {
    "units": [30, 30],
    "act_funcs": "relu",
    "n_outputs": 2,
    "output_act_func": "linear",
    "output_names": ["score", "f_min"],
    "dropout": 0.25,
}

pre_config = {"standardise": True, "whiten": True}

# Training details
optimizer = "Adam"
loss = {cur_label_name: "mean_squared_error" for cur_label_name in label_names}
loss_weights = [1.0, 1.0]

val_size = 0.2
n_epochs = 100
batch_size = 32

# ---- Training ----
output_dir = Path(output_dir)

label_df = gm.utils.load_labels_from_dir(
    "/Users/Clus/code/work/gm_classifier/data/records/training_data/iter"
)
feature_df = gm.utils.load_features_from_dir(
    "/Users/Clus/code/work/gm_classifier/data/records/training_data/all_records_features"
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

y_train = gm.training.get_multi_output_y(train_df, label_names, train_ind)
y_val = gm.training.get_multi_output_y(train_df, label_names, val_ind)

ids_train = train_df.index.values[train_ind]
ids_val = train_df.index.values[val_ind]

train_data = (X_train, y_train, ids_train)
val_data = (X_val, y_val, ids_val)

# Run training of the model
history, X_train, X_val = gm.training.train(
    output_dir,
    pre_config,
    model_config,
    train_data,
    val_data=val_data,
    compile_kwargs={"optimizer": optimizer, "loss": loss},
    fit_kwargs={"batch_size": batch_size, "epochs": n_epochs, "verbose": 2},
)

# ---- Evaluation ----
# Plot the loss
fig, ax = gm.plots.plot_multi_loss(history, ["score", "f_min"], output_ffp=str(output_dir / "loss_plot.png"))

# Load the best model
model = keras.models.load_model(output_dir / "model.h5")

y_train_est = model.predict(X_train)
y_val_est = model.predict(X_val)

# Plot true vs estimated
plt.figure()
plt.scatter(y_train_est[0], y_train["score"], c="b", s=0.5)
plt.scatter(y_val_est[0], y_val["score"], c="r", s=0.5)

plt.figure()
plt.scatter(y_train_est[1], y_train["f_min"], c="b", s=0.5)
plt.scatter(y_val_est[1], y_val["f_min"], c="r", s=0.5)

plt.show()

exit()
