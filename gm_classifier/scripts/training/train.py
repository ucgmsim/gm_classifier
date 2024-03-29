import shutil
from pathlib import Path

import wandb
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from wandb.keras import WandbCallback

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
from gm_classifier.src.console import console

# --------------- Config ------------------

label_ffp = Path(
    "path/to/labels_file"
)

features_dir = Path(
    "path/to/features/dir"
)
base_output_dir = Path("output_dir")

tags = [f"features_{str(features_dir.stem)}"]

# --------------- Loading ------------------
scalar_feature_config = gmc.utils.load_yaml("./feature_config.yaml")
snr_feature_names = [
    f"snr_value_{freq:.3f}"
    for freq in np.logspace(np.log(0.01), np.log(25), 100, base=np.e)
]
hyperparams = gmc.utils.load_yaml("./hyperparams.yaml")

console.print("[green]Loading scalar features[/]")
X_scalar, label_df = gmc.data.load_dataset(
    features_dir, label_ffp, features=list(scalar_feature_config.keys()),
)
console.print("[green]Loading SNR series[/]")
X_snr, _ = gmc.data.load_dataset(
    features_dir, label_ffp, features=list(snr_feature_names),
)
assert np.all(X_snr.index == X_scalar.index)

y_score, y_fmin = label_df["score"], label_df["fmin"]
y_multi = label_df["multi"].astype(int)

console.print(f"Number of labelled samples: {label_df.shape[0]}")

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
    y_multi_train,
    y_multi_val,
    train_ids,
    val_ids,
) = gmc.pre.train_test_split(X_scalar, X_snr, y_score, y_fmin, y_multi)

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
model_config = gmc.model.get_model_config(hyperparams)
gmc_model = gmc.model.build_model(
    model_config,
    len(scalar_feature_config.keys()),
    X_snr_train.shape[1],
    multi_out=True,
)

# Save a plot of the model
keras.utils.plot_model(
    gmc_model,
    output_dir / "model.png",
    expand_nested=True,
    show_dtype=True,
    show_shapes=True,
)

# Setting to 0.0, results in nan values, so set to very small value
weight_lookup = {1.0: 1.0, 0.75: 0.75, 0.5: 0.1, 0.25: 1e-8, 0.0: 1e-8}
# fmin_loss = gmc.training.FMinLoss(weight_lookup, base_loss_fn=gmc.training.create_huber(1.0))
fmin_loss = gmc.training.FMinLoss(weight_lookup, base_loss_fn=gmc.training.fmin_log_loss)
score_loss = gmc.training.create_huber(0.5)


def fmin_loss_metric_fn(y, y_est):
    return fmin_loss.call(y, y_est)


loss_weights = [2.0, 0.05, 0.1]
gmc_model.compile(
    optimizer=hyperparams["optimizer"],
    loss={
        "score": score_loss,
        "fmin": fmin_loss,
        "multi": keras.losses.binary_crossentropy,
    },
    loss_weights=loss_weights,
    metrics={
        "score": [
            gmc.eval.ClassAcc(0.0),
            gmc.eval.ClassAcc(0.25),
            gmc.eval.ClassAcc(0.5),
            gmc.eval.ClassAcc(0.75),
            gmc.eval.ClassAcc(1.0),
        ],
        "fmin": fmin_loss_metric_fn,
        "multi": [keras.metrics.Precision(), keras.metrics.Recall()],
    },
    run_eagerly=False
)

# --------------- Save & print details ------------------
gmc.utils.save_print_data(
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
    train_ids=train_ids,
    val_ids=val_ids,
    loss_weights={"score_loss": loss_weights[0], "fmin_loss": loss_weights[1]},
    model=gmc_model,
)

# --------------- Train ------------------
history = gmc_model.fit(
    {"scalar": X_scalar_train.values, "snr": X_snr_train},
    {
        "score": y_score_train.values,
        "fmin": np.stack((y_score_train.values, y_fmin_train.values), axis=1),
        "multi": y_multi_train,
    },
    hyperparams["batch_size"],
    hyperparams["epochs"],
    verbose=2,
    validation_data=(
        {"scalar": X_scalar_val, "snr": X_snr_val},
        {
            "score": y_score_val.values,
            "fmin": np.stack((y_score_val.values, y_fmin_val.values), axis=1),
            "multi": y_multi_val,
        },
    ),
    callbacks=[
        keras.callbacks.EarlyStopping(
            # min_delta=0.005, patience=50, verbose=1, restore_best_weights=True
            min_delta=0.005, patience=100, verbose=1, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=20, verbose=1, min_lr=1e-6, min_delta=5e-3
            # factor=0.5, patience=50, verbose=1, min_lr=1e-6, min_delta=5e-3
        ),
        WandbCallback(),
    ],
).history

console.print("Saving the model")
model_dir = output_dir / "model"
gmc_model.save(model_dir)
shutil.copy(
    Path(__file__).parent / "feature_config.yaml", model_dir / "feature_config.yaml"
)

console.print("Saving pre-processing parameters")
gmc.utils.write_pickle(pre_params, model_dir / "pre_params.pickle")

# --------------- Eval ------------------
console.rule("[bold]Eval[/]")
gmc.eval.print_model_eval(
    gmc_model,
    X_scalar_train.values,
    X_snr_train,
    y_score_train.values.astype(np.float32),
    y_fmin_train.values.astype(np.float32),
    score_loss,
    fmin_loss,
    fmin_loss,
    score_loss_weight=loss_weights[0],
    fmin_loss_weight=loss_weights[1],
    y_multi=y_multi_train.values,
    multi_loss_weight=loss_weights[2],
)
gmc.eval.print_model_eval(
    gmc_model,
    X_scalar_val.values,
    X_snr_val,
    y_score_val.values.astype(np.float32),
    y_fmin_val.values.astype(np.float32),
    score_loss,
    fmin_loss,
    fmin_loss,
    score_loss_weight=loss_weights[0],
    fmin_loss_weight=loss_weights[1],
    y_multi=y_multi_val.values.astype(bool),
    multi_loss_weight=loss_weights[2],
    prefix="val",
)

gmc.plots.plot_loss(
    history,
    output_ffp=str(output_dir / "loss.png"),
    fig_kwargs=dict(figsize=(16, 10)),
    add_loss_keys=[
        ("score_loss", "b-"),
        ("val_score_loss", "b--"),
        ("fmin_loss", "g-"),
        ("val_fmin_loss", "g--"),
        ("multi_loss", "c-"),
        ("val_multi_loss", "c--"),
    ],
    ylim=(0.0, 1.0)
)

(
    y_score_est_train,
    y_score_est_train_std,
    y_fmin_est_train,
    y_fmin_est_train_std,
    y_multi_est_train,
    y_multi_est_train_std,
) = gmc.eval.get_combined_prediction(
    gmc_model,
    X_scalar_train,
    X_snr_train,
    n_preds=25,
    index=X_scalar_train.index.values.astype(str),
    multi_output=True,
)

(
    y_score_est_val,
    y_score_est_val_std,
    y_fmin_est_val,
    y_fmin_est_val_std,
    y_multi_est_val,
    y_multi_est_val_std,
) = gmc.eval.get_combined_prediction(
    gmc_model,
    X_scalar_val,
    X_snr_val,
    n_preds=25,
    index=X_scalar_val.index.values.astype(str),
    multi_output=True,
)

gmc.plots.plot_confusion_matrix(
    y_multi_train.values.astype(bool),
    (y_multi_est_train > 0.5).values,
    output_dir,
    "Training",
    "Multi_EQ",
)
gmc.plots.plot_confusion_matrix(
    y_multi_val.values.astype(bool),
    (y_multi_est_val > 0.5).values,
    output_dir,
    "Validation",
    "Multi_EQ",
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

gmc.plots.plot_fmin_true_vs_est(
    label_df.loc[train_ids], y_fmin_est_train, output_dir, title="training", zoom=True, log=True
)
gmc.plots.plot_fmin_true_vs_est(
    label_df.loc[val_ids], y_fmin_est_val, output_dir, title="validation", zoom=True, log=True
)
