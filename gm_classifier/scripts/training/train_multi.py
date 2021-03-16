import json
from pathlib import Path

import tensorflow as tf
import wandb
import seaborn as sns

import ml_tools

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


def get_act_fn(act_fn_type: str, p, x_min, x_max, z_min, z_max):
    if act_fn_type == "linear":
        return tf.keras.activations.linear
    if act_fn_type == "sigmoid":
        return tf.keras.activations.sigmoid
    else:
        return gmc.training.create_soft_clipping(
            p, z_min=z_min, z_max=z_max, x_min=x_min, x_max=x_max
        ),



# ----- Config -----
label_dir = Path(
    "/home/claudy/dev/work/data/gm_classifier/records/training_data/labels"
)

# features_dir = "/home/claudy/dev/work/data/gm_classifier/records/training_data/features/tmp"
features_dir = (
    "/home/claudy/dev/work/data/gm_classifier/records/training_data/features/210226"
)

base_output_dir = Path("/home/claudy/dev/work/data/gm_classifier/results/test")
ignore_ids_ffp = label_dir / "ignore_ids.txt"

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
    "spike_detector": ["standard", "whiten"],
    "jerk_detector": ["standard", "whiten"],
    "lowres_detector": ["standard", "whiten"],
    "gainjump_detector": ["standard", "whiten"],
    "flatline_detector": ["standard", "whiten"],
    "p_numpeaks_detector": ["standard", "whiten"],
    "p_multimax_detector": ["standard", "whiten"],
    "p_multidist_detector": ["standard", "whiten"],
    "s_numpeaks_detector": ["standard", "whiten"],
    "s_multimax_detector": ["standard", "whiten"],
    "s_multidist_detector": ["standard", "whiten"],
}


# For hyperparameter tuning using wandb
hyperparam_defaults = dict(
    scalar_n_units=16,
    scalar_n_layers=2,
    scalar_dropout=0.02,
    snr_n_filters_1=10,
    snr_kernel_size_1=10,
    snr_n_filters_2=20,
    snr_kernel_size_2=5,
    snr_pool_size=2,
    snr_dropout=0.02,
    snr_lstm_n_units_1=16,
    snr_lstm_n_units_2=8,
    comb_n_units=32,
    comb_n_layers=2,
    comb_dropout=0.02,
    out_n_units=16,
    out_n_layers=1,
    out_dropout=0.02,
    batch_size=32,
    score_out_act_f="linear",
    score_clipping_x_min=-1.0,
    score_clipping_x_max=1.0,
    score_clipping_p=10,
    f_min_out_act_f="clipping",
    f_min_clipping_x_min=-1.0,
    f_min_clipping_x_max=1.0,
    f_min_clipping_p=30,
)

# ---- Training ----
run_id = gmc.utils.create_run_id()
output_dir = base_output_dir / run_id
output_dir.mkdir(exist_ok=False, parents=False)

# Data loading
label_dfs = gmc.utils.load_labels_from_dir(
    str(label_dir),
    f_min_100_value=10,
    drop_na=True,
    drop_f_min_101=True,
    multi_eq_value=0.0,
    malf_value=0.0,
    merge=False,
    ignore_ids_ffp=ignore_ids_ffp,
)
feature_dfs = gmc.utils.load_features_from_dir(features_dir, merge=False)

# wandb setup
tags = ["multi_output"]
if "jerk_detector" in feature_config.keys():
    tags.append("malf_features")
if "p_numpeaks_detector" in feature_config.keys():
    tags.append("multi_eq_features")

wandb.init(config=hyperparam_defaults, project="gmc", name=run_id, tags=tags)
hyperparam_config = wandb.config

wandb.config.labels_dir = str(label_dir)
wandb.config.features_dir = str(features_dir)


model_config = {
    "dense_scalar_config": {
        # "input_dropout": None,
        "hidden_layer_func": ml_tools.hidden_layers.selu_mc_dropout,
        "hidden_layer_config": {"dropout": hyperparam_config["scalar_dropout"]},
        "units": hyperparam_config["scalar_n_layers"]
        * [hyperparam_config["scalar_n_units"]],
    },
    "snr_config": {
        "filters": [
            hyperparam_config["snr_n_filters_1"],
            hyperparam_config["snr_n_filters_2"],
        ],
        "kernel_sizes": [
            hyperparam_config["snr_kernel_size_1"],
            hyperparam_config["snr_kernel_size_2"],
        ],
        "layer_config": {
            "activation": "elu",
            "kernel_initializer": "glorot_uniform",
            "padding": "same",
        },
        "pool_size": hyperparam_config["snr_pool_size"],
        "dropout": hyperparam_config["snr_dropout"],
        "lstm_units": [
            hyperparam_config["snr_lstm_n_units_1"],
            hyperparam_config["snr_lstm_n_units_2"],
        ],
    },
    "dense_comb_config": {
        "hidden_layer_func": ml_tools.hidden_layers.selu_mc_dropout,
        "hidden_layer_config": {"dropout": hyperparam_config["comb_dropout"]},
        "units": hyperparam_config["comb_n_layers"]
        * [hyperparam_config["comb_n_units"]],
    },
    "output_config": {
        "score": {
            "hidden_layer_func": ml_tools.hidden_layers.selu_mc_dropout,
            "hidden_layer_config": {"dropout": hyperparam_config["out_dropout"]},
            "units": hyperparam_config["out_n_layers"]
            * [hyperparam_config["out_n_units"]],
            # "out_act_func": tf.keras.activations.linear,
            "out_act_func": get_act_fn(hyperparam_config["score_out_act_f"],
                                       hyperparam_config["score_clipping_p"],
                                       hyperparam_config["score_clipping_x_min"],
                                       hyperparam_config["score_clipping_x_max"],
                                       0.0, 1.0)
        },
        "f_min": {
            "hidden_layer_func": ml_tools.hidden_layers.relu_dropout,
            "hidden_layer_config": {"dropout": hyperparam_config["out_dropout"]},
            "units": hyperparam_config["out_n_layers"]
            * [hyperparam_config["out_n_units"]],
            "out_act_func": get_act_fn(hyperparam_config["f_min_out_act_f"],
                                       hyperparam_config["f_min_clipping_p"],
                                       hyperparam_config["f_min_clipping_x_min"],
                                       hyperparam_config["f_min_clipping_x_max"],
                                       0.1, 10.0),
            # "out_act_func": tf.keras.activations.linear,
        },
    },
}
print(
    "Model_config\n"
    + json.dumps(
        model_config,
        cls=ml_tools.utils.GenericObjJSONEncoder,
        sort_keys=False,
        indent=4,
    )
)

gm_model = gmc.RecordCompModel.from_config(
    output_dir,
    log_wandb=True,
    feature_config=feature_config,
    model_config=model_config,
)
print("Scalar features: ", gm_model.feature_names)

# Run training of the model
fit_kwargs = {
    "batch_size": hyperparam_config["batch_size"],
    "epochs": 250,
    "verbose": 2,
}
gm_model.train(
    feature_dfs,
    label_dfs,
    val_size=0.2,
    fit_kwargs=fit_kwargs,
    compile_kwargs={
        "optimizer": "Adam",
        "run_eagerly": False,
        # "loss": dict(score=tf.keras.losses.mse),
        # "loss_weights": [1.0]
        "loss": dict(score=tf.keras.losses.mse, f_min=tf.keras.losses.mse),
        "loss_weights": [1.0, 0.01]
    },
)
# gm_model.train(feature_dfs, label_dfs, val_size=0.2, fit_kwargs=fit_kwargs)

# Create eval plots

gmc.plots.create_eval_plots(
    output_dir,
    gm_model,
    feature_dfs,
    label_dfs,
    gm_model.train_ids,
    gm_model.val_ids,
    gm_model.train_history,
    n_preds=10,
)

exit()
