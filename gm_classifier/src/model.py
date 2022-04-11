from typing import Dict

import tensorflow as tf
import tensorflow.keras as keras

from . import hidden_layers

HIDDEN_LAYER_MAPPING = {
    "relu_bn": hidden_layers.relu_bn,
    "relu_dropout": hidden_layers.relu_dropout,
    "selu_dropout": hidden_layers.selu_dropout,
    "elu_mc_dropout": hidden_layers.elu_mc_dropout,
    "selu_mc_dropout": hidden_layers.selu_mc_dropout,
}


def get_hidden_layer_fn(name: str):
    return HIDDEN_LAYER_MAPPING[name]


def get_fmin_sigmoid():
    def fmin_sigmoid(z):
        # return (tf.keras.activations.sigmoid(z) + (0.1 / 9.9)) * 9.9
        return (tf.keras.activations.sigmoid(z - 5) + (0.01 / 10.0)) * 10.0

    return tf.function(fmin_sigmoid)


def get_fmin_act():
    def fmin_act(z):
        z = (3.16) ** (z + -2)
        z = tf.where(z > 10.0, tf.constant(10, dtype=tf.float32), z)
        z = tf.where(z < 0.01, tf.constant(0.01, dtype=tf.float32), z)
        return z

    return tf.function(fmin_act)


def get_model_config(hyperparams, **params):
    filters = [
        hyperparams["snr_n_filters_1"],
        hyperparams["snr_n_filters_2"],
        hyperparams["snr_n_filters_3"],
    ]
    kernel_sizes = [
        hyperparams["snr_kernel_size_1"],
        hyperparams["snr_kernel_size_2"],
        hyperparams["snr_kernel_size_3"],
    ]

    return {
        **hyperparams,
        **{
            # General
            "dense_hidden_layer_fn": get_hidden_layer_fn(
                hyperparams["dense_hidden_layer_fn"]
            ),
            # Scalar
            "scalar_units": [hyperparams["scalar_n_units"]]
            * hyperparams["scalar_n_layers"],
            # SNR
            "snr_filters": filters[: hyperparams["snr_n_cnn_layers"]],
            "snr_kernel_sizes": kernel_sizes[: hyperparams["snr_n_cnn_layers"]],
            "snr_lstm_units": [hyperparams["snr_n_lstm_units"]]
            * hyperparams["snr_n_lstm_layers"],
            # Combined
            "comb_dense_units": [hyperparams["comb_n_dense_units"]]
            * hyperparams["comb_n_dense_layers"],
            # Out
            "out_dense_units": [hyperparams["out_n_dense_units"]]
            * hyperparams["out_n_dense_layers"],
        },
        **params,
    }


def build_model(
    model_config: Dict,
    n_features: int,
    n_snr_steps: int,
    multi_out: bool = False,
    malf_out: bool = False,
):
    # Scalar Dense
    scalar_input = x_scalar = keras.Input(n_features, name="scalar")

    if model_config.get("scalar_input_dropout") is not None:
        x_scalar = keras.layers.Dropout(model_config["scalar_input_dropout"])(x_scalar)

    for n_units in model_config["scalar_units"]:
        x_scalar = model_config["dense_hidden_layer_fn"](
            x_scalar, n_units, dropout=model_config["dense_dropout"], l2_reg=model_config["dense_l2_reg"]
        )

    # SNR
    snr_input = x_snr = keras.Input((n_snr_steps, 1), name="snr")

    for cur_n_filters, cur_kernel_size in zip(
        model_config["snr_filters"], model_config["snr_kernel_sizes"]
    ):
        x_snr = hidden_layers.cnn1_mc_dropout_pool(
            x_snr,
            cur_n_filters,
            cur_kernel_size,
            model_config["snr_cnn_layer_config"],
            dropout=model_config["snr_cnn_dropout"],
            l2_reg=model_config["snr_l2_reg"],
        )

    for cur_n_units in model_config["snr_lstm_units"]:
        x_snr = hidden_layers.bi_lstm(
            x_snr, cur_n_units, return_state=False, return_sequences=True, l2_reg=model_config["snr_lstm_l2_reg"]
        )

    x_snr = hidden_layers.bi_lstm(
        x_snr,
        model_config["snr_n_final_lstm_units"],
        return_state=True,
        return_sequences=False,
        l2_reg=model_config["snr_lstm_l2_reg"]
    )
    x_snr = keras.layers.Concatenate()(x_snr[1:])

    x = keras.layers.Concatenate()([x_scalar, x_snr])

    for cur_n_units in model_config["comb_dense_units"]:
        x = model_config["dense_hidden_layer_fn"](
            x, cur_n_units, dropout=model_config["dense_dropout"], l2_reg=model_config["dense_l2_reg"]
        )

    # Score
    x_score = x
    for cur_n_units in model_config["out_dense_units"]:
        x_score = model_config["dense_hidden_layer_fn"](
            x_score, cur_n_units, dropout=model_config["dense_dropout"], l2_reg=model_config["dense_l2_reg"]
        )
    score_out = keras.layers.Dense(1, activation="sigmoid", name="score")(x_score)

    # Fmin
    x_fmin = x
    for cur_n_units in model_config["out_dense_units"]:
        x_fmin = model_config["dense_hidden_layer_fn"](
            x_fmin, cur_n_units, dropout=model_config["dense_dropout"], l2_reg=model_config["dense_l2_reg"]
        )
    # fmin_out = keras.layers.Dense(2, activation=get_fmin_sigmoid(), name="fmin")(x_fmin)
    fmin_out = keras.layers.Dense(2, activation=get_fmin_act(), name="fmin")(x_fmin)

    outputs = [score_out, fmin_out]

    if multi_out:
        x_multi = x
        for cur_n_units in model_config["out_dense_units"]:
            x_multi = model_config["dense_hidden_layer_fn"](
                x_multi, cur_n_units, dropout=model_config["dense_dropout"], l2_reg=model_config["dense_l2_reg"]
            )
        multi_out = keras.layers.Dense(1, activation="sigmoid", name="multi")(x_multi)
        outputs.append(multi_out)

    # if malf_out:
    #     x_malf = x
    #     for cur_n_units in model_config["out_dense_units"]:
    #         x_malf = model_config["dense_hidden_layer_fn"](
    #             x_malf, cur_n_units, dropout=model_config["dense_dropout"]
    #         )
    #     malf_out = keras.layers.Dense(1, activation="sigmoid", name="malf")(x_malf)
    #     outputs.append(malf_out)

    return keras.Model(inputs=[scalar_input, snr_input], outputs=outputs)
