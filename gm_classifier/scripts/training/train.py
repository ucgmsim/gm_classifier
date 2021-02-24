"""Trains and evaluates a model that takes the features from all
three components for a specific record and estimates a score & f_min
for each of the components"""
from pathlib import Path

import tensorflow as tf
import wandb
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

# features_dir = "/home/claudy/dev/work/data/gm_classifier/records/training_data/features/tmp"
features_dir = "/home/claudy/dev/work/data/gm_classifier/records/training_data/features/210223"

base_output_dir = Path("/home/claudy/dev/work/data/gm_classifier/results/test")

use_wandb = True

# ---- Training ----
run_id = gmc.utils.create_run_id()
output_dir = base_output_dir / run_id
output_dir.mkdir(exist_ok=False, parents=False)

label_dfs = gmc.utils.load_labels_from_dir(label_dir, f_min_100_value=10, drop_na=True, drop_f_min_101=True,
                                           multi_eq_value=0.0, malf_value=0.0, merge=False)
feature_dfs = gmc.utils.load_features_from_dir(features_dir, merge=False)


gm_model = gmc.RecordCompModel.from_config(output_dir, log_wandb=use_wandb)
print("Scalar features: ", gm_model.feature_names)

if use_wandb:
    tags = []
    if "jerk_detector" in gm_model.feature_names:
        tags.append("malf_features")
    if "numpeaks_detector" in gm_model.feature_names:
        tags.append("multi_eq_features")

    wandb.init(project="gmc", name=run_id, tags=tags)

# Run training of the model
fit_kwargs = {"batch_size": 32, "epochs": 300, "verbose": 1}
gm_model.train(feature_dfs, label_dfs, val_size=0.2, fit_kwargs=fit_kwargs)

# Create eval plots
gmc.plots.create_eval_plots(output_dir, gm_model, feature_dfs, label_dfs,
                            gm_model.train_ids, gm_model.val_ids, gm_model.train_history, n_preds=10)

exit()


