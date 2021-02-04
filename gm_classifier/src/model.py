import os
from typing import List, Dict, Tuple, Union, Callable

import tensorflow as tf
import tensorflow.keras as keras

import ml_tools


def build_dense_cnn_model(
    dense_config: Dict,
    snr_config: Dict,
    dense_final_config: Dict,
    n_features: int,
    n_snr_steps: int,
    n_outputs: int,
    out_act_func: tf.function,
):
    hidden_layer_func = dense_config["hidden_layer_func"]
    hidden_layer_config = dense_config["hidden_layer_config"]

    # Build the scalar features dense layers
    scalar_inputs = keras.Input(n_features, name="features")
    x_nn = hidden_layer_func(
        scalar_inputs, dense_config["units"][0], **hidden_layer_config
    )
    for n_units in dense_config["units"][1:]:
        x_nn = hidden_layer_func(x_nn, n_units, **hidden_layer_config)

    # Build the CNN + LSTM layers
    cnn_filters, cnn_kernels = snr_config["filters"], snr_config["kernel_sizes"]
    dropout, cnn_layer_config = snr_config["dropout"], snr_config["layer_config"]
    lstm_units = snr_config["lstm_units"]

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


    for ix, n_units in enumerate(lstm_units):
        x_snr = keras.layers.Bidirectional(
            layer=keras.layers.LSTM(
                units=n_units,
                return_sequences=True if ix + 1 < len(lstm_units) else False,
            )
        )(x_snr)

    # Build the output layers
    x = keras.layers.Concatenate()([x_nn, x_snr])

    for n_units in dense_final_config["units"]:
        x = dense_final_config["hidden_layer_func"](
            x, n_units, **dense_final_config["hidden_layer_config"]
        )

    outputs = keras.layers.Dense(n_outputs, activation=out_act_func, name="output")(x)
    return keras.Model(inputs=[scalar_inputs, snr_input], outputs=outputs)