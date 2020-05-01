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
from sklearn.utils.class_weight import compute_class_weight

import seaborn as sns

sns.set()
sns.set_style("whitegrid")

import gm_classifier as gm

# ----- Config -----
# label_dir = "/Users/Clus/code/work/gm_classifier/data/records/training_data/iter"
label_dir = (
    "/Users/Clus/code/work/gm_classifier/data/records/local/training_data/labels"
)

# features_dir = "/Users/Clus/code/work/gm_classifier/data/records/training_data/all_records_features/200401"
features_dir = "/Users/Clus/code/work/gm_classifier/data/records/local/training_data/all_records_features/200401"

# output_dir = "/Users/Clus/code/work/tmp/gm_classifier/record_based/tmp_3"
# output_dir = "/Users/Clus/code/work/tmp/gm_classifier/record_based/new_loss/test_1"
output_dir = "/Users/Clus/code/work/tmp/gm_classifier/record_based/new_loss/test_tmp"
# output_dir = "/Users/Clus/code/work/gm_classifier/results/200420_results"

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

snr_features = [
    f"snr_value_{freq:.3f}"
    for freq in np.logspace(np.log(0.05), np.log(20), 50, base=np.e)
]


# Model config
act_fn = "elu"
kernel_init = "glorot_uniform"
dropout_rate = 0.3
model_config = {
    "dense_layer_config": [
        (
            keras.layers.Dense,
            {"units": 32, "activation": act_fn, "kernel_initializer": kernel_init},
        ),
        (
            keras.layers.Dense,
            {"units": 16, "activation": act_fn, "kernel_initializer": kernel_init},
        ),
    ],
    "dense_input_name": "features",
    "cnn_layer_config": [
        (
            keras.layers.Conv1D,
            {
                "filters": 32,
                "kernel_size": 5,
                "strides": 1,
                "activation": act_fn,
                "kernel_initializer": kernel_init,
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
                "activation": act_fn,
                "kernel_initializer": kernel_init,
                "padding": "same",
            },
        ),
        (keras.layers.MaxPooling1D, {"pool_size": 2}),
        (keras.layers.Dropout, {"rate": dropout_rate}),
    ],
    "cnn_input_name": "snr_series",
    "comb_layer_config": [
        (
            keras.layers.Dense,
            {"units": 64, "activation": act_fn, "kernel_initializer": kernel_init},
        ),
        (keras.layers.Dropout, {"rate": dropout_rate}),
        (
            keras.layers.Dense,
            {"units": 32, "activation": act_fn, "kernel_initializer": kernel_init},
        ),
        (keras.layers.Dropout, {"rate": dropout_rate}),
        (
            keras.layers.Dense,
            {"units": 16, "activation": act_fn, "kernel_initializer": kernel_init},
        ),
    ],
    "output": keras.layers.Dense(6, activation="linear")
}

# Training details
optimizer = "Adam"
val_size = 0.1
n_epochs = 400
batch_size = 64

output_dir = Path(output_dir)

# ---- Training ----
label_df = gm.utils.load_labels_from_dir(label_dir, f_min_100_value=10)
feature_df = gm.utils.load_features_from_dir(features_dir)

train_df = pd.merge(
    feature_df, label_df, how="inner", left_index=True, right_index=True
)

gm_model = gm.RecordCompModel(output_dir, label_names=label_names, feature_config=feature_config,
                           snr_freq_values=np.logspace(np.log(0.05), np.log(20), 50, base=np.e),
                           model_config=model_config)

# Run training of the model
fit_kwargs = {"batch_size": 64, "epochs": 200, "verbose": 2}
gm_model.train(train_df, val_size=val_size, fit_kwargs=fit_kwargs)

# Create eval plots
gm_model.create_eval_plots()



