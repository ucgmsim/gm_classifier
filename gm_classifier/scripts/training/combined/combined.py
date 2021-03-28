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

tags = ["combined"]

# --------------- Loading ------------------
scalar_feature_config = ml_tools.utils.load_yaml("./feature_config.yaml")
snr_feature_names = [
    f"snr_value_{freq:.3f}"
    for freq in np.logspace(np.log(0.01), np.log(25), 100, base=np.e)
]
hyperparams = ml_tools.utils.load_yaml("./hyperparams.yaml")

X_scalar, label_df = gmc.data.load_dataset(
    features_dir,
    label_dir,
    ignore_ids_ffp,
    features=list(scalar_feature_config.keys()),
)
X_snr, _ = gmc.data.load_dataset(
    features_dir, label_dir, ignore_ids_ffp, features=list(snr_feature_names),
)

y_score, y_fmin = label_df["score"], label_df["f_min"]

# --------------- Split & pre-processing ------------------
(
    X_scalar_train,
    X_scalar_val,
    X_snr_train,
    X_snr_val,
    y_score_train,
    y_score_val,
    y_fmin_train,
    y_fmin_val,
    train_ids,
    val_ids,
) = gmc.pre.train_test_split(X_scalar, X_snr, y_score, y_fmin)

X_scalar_train, pre_params = gmc.pre.run_preprocessing(
    X_scalar_train, scalar_feature_config
)
X_scalar_val, _ = gmc.pre.run_preprocessing(
    X_scalar_val, scalar_feature_config, params=pre_params
)

X_snr_train, X_snr_val = (
    X_snr_train.apply(np.log).values[..., None],
    X_snr_val.apply(np.log).values[..., None],
)


# --------------- Setup & wandb ------------------
run_id = gmc.utils.create_run_id()
output_dir = base_output_dir / run_id
output_dir.mkdir(exist_ok=False, parents=False)

wandb.init(config=hyperparams, project="gmc", name=run_id, tags=tags)
hyperparams = wandb.config

# --------------- Build & compile model ------------------


model_config = gmc.model.get_combined_model_config(hyperparams)
gmc_model = gmc.model.build_combined_model(
    model_config, len(scalar_feature_config.keys()), X_snr_train.shape[1]
)

weight_lookup = {1.0: 1.0, 0.75: 0.75, 0.5: 0.1, 0.25: 0.0, 0.0: 0.0}
fmin_loss = gmc.training.FMinLoss(weight_lookup)

loss_weights = [1.0, 0.01]
gmc_model.compile(
    optimizer=hyperparams["optimizer"],
    loss={"score": keras.losses.mse, "fmin": fmin_loss},
    loss_weights=loss_weights,
    # run_eagerly=True
)

# --------------- Save & print details ------------------
ml_tools.utils.save_print_data(
    output_dir,
    feature_config=(scalar_feature_config, True),
    hyperparams=(dict(hyperparams), True),
    X_scalar_train=X_scalar_train,
    X_scalar_val=X_scalar_val,
    X_snr_train=X_snr_train,
    X_snr_val=X_snr_val,
    y_score_train=y_score_train,
    y_score_val=y_score_val,
    y_fmin_train=y_fmin_train,
    y_fmin_val=y_fmin_val,
    loss_weights={"score_loss": loss_weights[0], "fmin_loss": loss_weights[1]},
    model=gmc_model,
)

# --------------- Train ------------------

history = gmc_model.fit(
    {"scalar": X_scalar_train.values, "snr": X_snr_train},
    {
        "score": y_score_train.values,
        "fmin": np.stack((y_score_train.values, y_fmin_train.values), axis=1),
    },
    hyperparams["batch_size"],
    hyperparams["epochs"],
    verbose=2,
    validation_data=(
        {"scalar": X_scalar_val, "snr": X_snr_val},
        {
            "score": y_score_val.values,
            "fmin": np.stack((y_score_val.values, y_fmin_val.values), axis=1),
        },
    ),
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
gmc.eval.print_combined_model_eval(
    gmc_model,
    X_scalar_train.values,
    X_snr_train,
    y_score_train.values.astype(np.float32),
    y_fmin_train.values.astype(np.float32),
    keras.losses.mse,
    fmin_loss,
    score_loss_weight=loss_weights[0],
    fmin_loss_weight=loss_weights[1],
)
gmc.eval.print_combined_model_eval(
    gmc_model,
    X_scalar_val.values,
    X_snr_val,
    y_score_val.values.astype(np.float32),
    y_fmin_val.values.astype(np.float32),
    keras.losses.mse,
    fmin_loss,
    score_loss_weight=loss_weights[0],
    fmin_loss_weight=loss_weights[1],
    prefix="val",
)

gmc.plots.plot_loss(
    history, output_ffp=output_dir / "loss.png", fig_kwargs=dict(figsize=(16, 10))
)

y_score_est_train, _, y_fmin_est_train, _ = gmc.eval.get_combined_prediction(
    gmc_model, X_scalar_train, X_snr_train, n_preds=25, index=X_scalar_train.index.values.astype(str)
)
y_score_est_val, _, y_fmin_est_val, _ = gmc.eval.get_combined_prediction(
    gmc_model, X_scalar_val, X_snr_val, n_preds=25, index=X_scalar_val.index.values.astype(str)
)


gmc.plots.plot_score_true_vs_est(
    label_df.loc[train_ids], y_score_est_train, output_dir, title="training"
)
gmc.plots.plot_score_true_vs_est(
    label_df.loc[val_ids], y_score_est_val, output_dir, title="validation"
)

gmc.plots.plot_fmin_true_vs_est(
    label_df.loc[train_ids], y_fmin_est_train, output_dir, title="training"
)
gmc.plots.plot_fmin_true_vs_est(
    label_df.loc[val_ids], y_fmin_est_val, output_dir, title="validation"
)

gmc.plots.plot_fmin_true_vs_est(
    label_df.loc[train_ids], y_fmin_est_train, output_dir, title="training", zoom=True
)
gmc.plots.plot_fmin_true_vs_est(
    label_df.loc[val_ids], y_fmin_est_val, output_dir, title="validation", zoom=True
)

exit()
