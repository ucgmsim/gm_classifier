from typing import List, Dict, Tuple, Union, Callable


import wandb
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import ml_tools
from . import training


def get_score_act_fn(act_fn_type: str, p, x_min, x_max, z_min, z_max):
    if act_fn_type == "linear":
        return tf.keras.activations.linear
    if act_fn_type == "sigmoid":
        return tf.keras.activations.sigmoid
    else:
        return training.create_soft_clipping(
            p, z_min=z_min, z_max=z_max, x_min=x_min, x_max=x_max
        )


def get_score_model_config(hyperparams: Dict, **params):
    hidden_layer_fn = ml_tools.utils.get_hidden_layer_fn(hyperparams["hidden_layer_fn"])
    units = [hyperparams["n_units"]] * hyperparams["n_layers"]

    return {
        **hyperparams,
        **{
            "hidden_layer_fn": hidden_layer_fn,
            "hidden_layer_config": {"dropout": hyperparams["dropout"]},
            "units": units,
            "out_act_fn": get_score_act_fn(
                hyperparams["out_act_fn"],
                hyperparams["out_clipping_p"],
                hyperparams["out_clipping_x_min"],
                hyperparams["out_clipping_x_max"],
                0.0,
                1.0,
            ),
        },
        **params,
    }


def build_score_model(model_config: Dict, n_features: int) -> keras.Model:
    inputs = x_nn = keras.Input(n_features, name="features")

    if "input_dropout" in model_config:
        x_nn = keras.layers.Dropout(model_config["input_dropout"])(x_nn)

    for n_units in model_config["units"]:
        x_nn = model_config["hidden_layer_fn"](
            x_nn, n_units, **model_config["hidden_layer_config"]
        )

    outputs = keras.layers.Dense(1, activation=model_config["out_act_fn"])(x_nn)

    return keras.Model(inputs=inputs, outputs=outputs)


def build_fmin_model(model_config: Dict, n_snr_steps: int) -> keras.Model:
    input = x = keras.Input((n_snr_steps, 1))

    for cur_n_filters, cur_kernel_size in zip(
        model_config["filters"], model_config["kernel_sizes"]
    ):
        x = ml_tools.hidden_layers.cnn1_mc_dropout_pool(
            x, cur_n_filters, cur_kernel_size, model_config["cnn_config"]
        )

    for cur_n_units in model_config["lstm_units"]:
        x = ml_tools.hidden_layers.bi_lstm(
            x, cur_n_units, return_state=False, return_sequences=True
        )

    x = ml_tools.hidden_layers.bi_lstm(
        x, model_config["final_lstm_units"], return_state=True, return_sequences=False
    )
    x = keras.layers.Concatenate()(x[1:])

    for cur_n_units in model_config["dense_units"]:
        x = model_config["hidden_layer_fn"](x, cur_n_units)

    output = keras.layers.Dense(1, activation=model_config["out_act_fn"])(x)

    return keras.Model(inputs=input, outputs=output)


def build_dense_cnn_model(
    dense_scalar_config: Dict,
    snr_config: Dict,
    dense_comb_config: Dict,
    n_features: int,
    n_snr_steps: int,
    n_outputs: int = None,
    out_act_func: tf.function = None,
    output_config: Dict = None,
):
    hidden_layer_func = dense_scalar_config["hidden_layer_func"]
    hidden_layer_config = dense_scalar_config["hidden_layer_config"]

    # Build the scalar features dense layers
    scalar_inputs = x_nn = keras.Input(n_features, name="features")

    if "input_dropout" in dense_scalar_config:
        x_nn = keras.layers.Dropout(dense_scalar_config["input_dropout"])(x_nn)

    for n_units in dense_scalar_config["units"]:
        x_nn = hidden_layer_func(x_nn, n_units, **hidden_layer_config)

    # Build the CNN + LSTM layers
    cnn_filters, cnn_kernels = snr_config["filters"], snr_config["kernel_sizes"]
    dropout, cnn_layer_config = snr_config["dropout"], snr_config["layer_config"]
    pool_size, lstm_units = snr_config["pool_size"], snr_config["lstm_units"]

    snr_input = keras.Input((n_snr_steps, 1), name="snr_series")
    x_snr = keras.layers.Conv1D(
        filters=cnn_filters[0], kernel_size=cnn_kernels[0], **cnn_layer_config
    )(snr_input)
    for n_filters, n_kernels in zip(cnn_filters[1:], cnn_kernels[1:]):
        keras.layers.Conv1D(
            filters=n_filters, kernel_size=n_kernels, **cnn_layer_config
        )(x_snr)
        if dropout is not None:
            x_snr = ml_tools.hidden_layers.MCSpatialDropout1D(rate=dropout)(x_snr)
        if pool_size is not None:
            x_snr = keras.layers.MaxPooling1D(pool_size)(x_snr)

    if len(lstm_units) > 0:
        for ix, n_units in enumerate(lstm_units):
            last_lstm = True if ix + 1 == len(lstm_units) else False
            x_snr = keras.layers.Bidirectional(
                layer=keras.layers.LSTM(
                    units=n_units,
                    return_sequences=True if not last_lstm else False,
                    return_state=True if last_lstm else False,
                )
            )(x_snr)
            if last_lstm:
                x_snr = keras.layers.Concatenate()(x_snr[1:])
    else:
        x_snr = keras.layers.Flatten()(x_snr)

    # Build the output layers
    x = keras.layers.Concatenate()([x_nn, x_snr])

    for n_units in dense_comb_config["units"]:
        x = dense_comb_config["hidden_layer_func"](
            x, n_units, **dense_comb_config["hidden_layer_config"]
        )

    if n_outputs is not None and out_act_func is not None:
        outputs = keras.layers.Dense(n_outputs, activation=out_act_func, name="output")(
            x
        )
    elif output_config is not None:
        outputs = []
        score_output = None
        for cur_output_name, cur_config in output_config.items():
            cur_x = x

            # if cur_output_name == "f_min":
            #     cur_x = keras.layers.Concatenate()([x, score_output])

            for n_units in cur_config["units"]:
                cur_x = dense_comb_config["hidden_layer_func"](
                    cur_x, n_units, **dense_comb_config["hidden_layer_config"]
                )

            cur_output = keras.layers.Dense(
                1, activation=cur_config["out_act_func"], name=cur_output_name
            )(cur_x)
            outputs.append(cur_output)

            if cur_output_name == "score":
                score_output = cur_output
    else:
        raise NotImplementedError()

    return keras.Model(inputs=[scalar_inputs, snr_input], outputs=outputs)
