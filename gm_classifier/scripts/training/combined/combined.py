import time
from pathlib import Path

import yaml
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import wandb
from sklearn.model_selection import train_test_split
from wandb.keras import WandbCallback

import ml_tools

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

# --------------- Config ------------------

label_dir = Path(
    "/home/claudy/dev/work/data/gm_classifier/records/training_data/labels"
)

features_dir = Path(
    "/home/claudy/dev/work/data/gm_classifier/records/training_data/features/210226"
)
base_output_dir = Path("/home/claudy/dev/work/data/gm_classifier/results/test")
ignore_ids_ffp = label_dir / "ignore_ids.txt"

tags = ["simple_score"]

# --------------- Loading ------------------
feature_config = ml_tools.utils.load_yaml("./feature_config.yaml")
hyperparams = ml_tools.utils.load_yaml("./hyperparams.yaml")

X, label_df = gmc.data.load_dataset(
    features_dir, label_dir, ignore_ids_ffp, features=list(feature_config.keys()),
)
y = label_df["score"]

# --------------- Split & pre-processing ------------------
X_train, X_val, y_train, y_val, train_ids, val_ids = gmc.pre.train_test_split(X, y)

X_train, pre_params = gmc.pre.run_preprocessing(X_train, feature_config)
X_val, _ = gmc.pre.run_preprocessing(X_val, feature_config, params=pre_params)

# --------------- Setup & wandb ------------------
run_id = gmc.utils.create_run_id()
output_dir = base_output_dir / run_id
output_dir.mkdir(exist_ok=False, parents=False)

wandb.init(config=hyperparams, project="gmc", name=run_id, tags=tags)
hyperparams = wandb.config

# --------------- Build & compile model ------------------
model_config = gmc.model.get_score_model_config(hyperparams)
gmc_model = gmc.model.build_score_model(
    model_config,
    len(feature_config.keys()),
)

loss_fn = keras.losses.mse
gmc_model.compile(optimizer=hyperparams["optimizer"], loss=loss_fn)

# --------------- Save & print details ------------------
ml_tools.utils.save_print_data(
    output_dir,
    feature_config=(feature_config, True),
    hyperparams=(dict(hyperparams), True),
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    model=gmc_model,
)

# --------------- Train ------------------

history = gmc_model.fit(
    X_train.values,
    y_train.values,
    hyperparams["batch_size"],
    hyperparams["epochs"],
    verbose=2,
    validation_data=(X_val, y_val),
    callbacks=[
        keras.callbacks.EarlyStopping(
            min_delta=0.005, patience=100, verbose=1, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=50, verbose=1, min_lr=1e-6, min_delta=5e-3
        ),
        WandbCallback(),
    ],
).history


# --------------- Eval ------------------
gmc.eval.print_single_model_eval(gmc_model, X_train.values, y_train.values, loss_fn)
gmc.eval.print_single_model_eval(
    gmc_model, X_val.values, y_val.values, loss_fn, prefix="val"
)

gmc.plots.plot_loss(
    history, output_ffp=output_dir / "loss.png", fig_kwargs=dict(figsize=(16, 10))
)

y_est_train, _ = gmc.eval.get_mc_single_predictions(gmc_model, X_train)
y_est_val, _ = gmc.eval.get_mc_single_predictions(gmc_model, X_val)

gmc.plots.plot_score_true_vs_est(
    label_df.loc[train_ids], y_est_train, output_dir, title="training"
)
gmc.plots.plot_score_true_vs_est(
    label_df.loc[val_ids], y_est_val, output_dir, title="validation"
)

time.sleep(30)

exit()
