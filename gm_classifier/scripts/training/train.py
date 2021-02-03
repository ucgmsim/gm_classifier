"""Trains and evaluates a model that takes the features from all
three components for a specific record and estimates a score & f_min
for each of the components"""
import json
from pathlib import Path

import wandb
from wandb.keras import WandbCallback

import pandas as pd
import seaborn as sns

sns.set()
sns.set_style("whitegrid")

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

label_df = gmc.utils.load_labels_from_dir(label_dir, f_min_100_value=10)
feature_df = gmc.utils.load_features_from_dir(features_dir)

train_df = pd.merge(
    feature_df, label_df, how="inner", left_index=True, right_index=True
)

gm_model = gmc.RecordCompModel(output_dir)

# Run training of the model
fit_kwargs = {"batch_size": 32, "epochs": 300, "verbose": 2}
gm_model.train(train_df, val_size=0.1, fit_kwargs=fit_kwargs)

# Create eval plots
gm_model.create_eval_plots()


