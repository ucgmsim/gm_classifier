from typing import List, Dict, Tuple, Union, Callable


import wandb
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import ml_tools

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
        outputs = keras.layers.Dense(n_outputs, activation=out_act_func, name="output")(x)
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

            cur_output = keras.layers.Dense(1, activation=cur_config["out_act_func"], name=cur_output_name)(cur_x)
            outputs.append(cur_output)

            if cur_output_name == "score":
                score_output = cur_output
    else:
        raise NotImplementedError()

    return keras.Model(inputs=[scalar_inputs, snr_input], outputs=outputs)
