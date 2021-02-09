"""Trains and evaluates a model that takes the features from all
three components for a specific record and estimates a score & f_min
for each of the components"""
import json
from pathlib import Path

import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
import pandas as pd
import seaborn as sns

sns.set()
sns.set_style("whitegrid")

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




# ----- Config -----
label_dir = (
    "/home/claudy/dev/work/data/gm_classifier/records/training_data/labels"
)

features_dir = "/home/claudy/dev/work/data/gm_classifier/records/training_data/features/210121"

base_output_dir = Path("/home/claudy/dev/work/data/gm_classifier/results/test")

# ---- Training ----
output_dir = base_output_dir / gmc.utils.create_run_id()
output_dir.mkdir(exist_ok=False, parents=False)

label_dfs = gmc.utils.load_labels_from_dir(label_dir, f_min_100_value=10, merge=False)
feature_dfs = gmc.utils.load_features_from_dir(features_dir, merge=False)

gm_model = gmc.RecordCompModel.from_config(output_dir)

# Run training of the model
fit_kwargs = {"batch_size": 32, "epochs": 300, "verbose": 1}
gm_model.train(feature_dfs, label_dfs, val_size=0.2, fit_kwargs=fit_kwargs)

# t = gm_model.predict((feature_df_X, feature_df_Y, feature_df_Z), n_preds=5)

# Create eval plots
gmc.plots.create_eval_plots(output_dir, gm_model, feature_dfs, label_dfs,
                            gm_model.train_ids, gm_model.val_ids, gm_model.train_history, n_preds=10)

exit()


