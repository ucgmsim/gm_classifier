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
label_dir = (
    "/Users/Clus/code/work/gm_classifier/data/records/local/training_data/labels"
)

features_dir = "/Users/Clus/code/work/gm_classifier/data/records/local/training_data/all_records_features/200429"

output_dir = Path("/Users/Clus/code/work/gm_classifier/results/200529_results/new_model")

# ---- Training ----
label_df = gm.utils.load_labels_from_dir(label_dir, f_min_100_value=10)
feature_df = gm.utils.load_features_from_dir(features_dir)

train_df = pd.merge(
    feature_df, label_df, how="inner", left_index=True, right_index=True
)

# gm_model = gm.RecordCompModel(
#     output_dir,
#     label_names=label_names,
#     feature_config=feature_config,
#     # snr_freq_values=np.logspace(np.log(0.05), np.log(20), 50, base=np.e),
#     snr_freq_values=np.logspace(np.log(0.01), np.log(25), 100, base=np.e),
#     model_config=model_config,
# )
gm_model = gm.RecordCompModel(output_dir)

# Run training of the model
fit_kwargs = {"batch_size": 32, "epochs": 300, "verbose": 2}
# gm_model.train(train_df, val_size=val_size, fit_kwargs=fit_kwargs, compile_kwargs={"run_eagerly": True})
gm_model.train(train_df, val_size=0.1, fit_kwargs=fit_kwargs)

# Create eval plots
gm_model.create_eval_plots()


# # Temporary -- hack..
# output_ffp = Path("/Users/Clus/code/work/gm_classifier/results/200529_results/new_model/results_20200529.csv")
# y_hat = gm_model.predict(feature_df)
# est_df = pd.DataFrame(index=feature_df.index.values, data=y_hat, columns=gm_model.label_names)
#
# if (output_dir / "est_df.csv").is_file():
#     label_est_df = pd.read_csv(output_dir / "est_df.csv", index_col="record_id")
#     assert np.all(np.isclose(est_df.loc[label_est_df.index.values], label_est_df.values, atol=1e-5))
#
# # Save the result
# result_df = pd.merge(feature_df, est_df, how="inner", left_index=True, right_index=True)
# result_df.to_csv(output_ffp, index_label="record_id")