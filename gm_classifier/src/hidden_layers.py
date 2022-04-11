from typing import Dict

from tensorflow import keras
from tensorflow.keras import layers


class MCDropout(keras.layers.Dropout):
    def call(self, inputs, **kwargs):
        return super().call(inputs, training=True)


class MCAlphaDropout(keras.layers.AlphaDropout):
    def call(self, inputs, **kwargs):
        return super(MCAlphaDropout, self).call(inputs, training=True)


class MCSpatialDropout1D(keras.layers.SpatialDropout1D):
    def call(self, inputs, **kwargs):
        return super().call(inputs, training=True)


def elu_mc_dropout(input: layers.Layer, n_units: int, dropout: float = None):
    x = layers.Dense(
        units=n_units, activation="elu", kernel_initializer="glorot_uniform"
    )(input)
    if dropout is not None:
        x = MCDropout(rate=dropout)(x)

    return x


def relu_bn(input: layers.Layer, n_units: int, l2: float = None):
    reg = keras.regularizers.L2(l2) if l2 is not None else None
    x = layers.Dense(units=n_units, activation="relu", kernel_regularizer=reg)(input)
    x = layers.BatchNormalization()(x)

    return x


def relu_dropout(input: layers.Layer, n_units: int, dropout: float = 0.2):
    x = layers.Dense(units=n_units, activation="relu")(input)
    if dropout is not None:
        x = layers.Dropout(rate=dropout)(x)

    return x


def selu_dropout(input: layers.Layer, n_units: int, dropout: float = 0.2, l2_reg: float = None):
    l2_reg = keras.regularizers.l2(l2_reg) if l2_reg is not None else None
    x = layers.Dense(
        units=n_units, activation="selu", kernel_initializer="lecun_normal", kernel_regularizer=l2_reg,
    )(input)
    if dropout is not None:
        x = layers.AlphaDropout(dropout)(x)

    return x


def selu_mc_dropout(input: layers.Layer, n_units: int, dropout: float = 0.05, l2_reg: float = None):
    l2_reg = keras.regularizers.l2(l2_reg) if l2_reg is not None else None
    x = layers.Dense(
        units=n_units, activation="selu", kernel_initializer="lecun_normal", kernel_regularizer=l2_reg
    )(input)
    if dropout is not None:
        x = MCAlphaDropout(dropout)(x)

    return x

def cnn1_mc_dropout_pool(
    input: layers.Layer,
    filters: int,
    kernel_size: int,
    cnn_config: Dict,
    dropout: float = 0.1,
    pool_size: int = 2,
    l2_reg: float = None
):
    l2_reg = keras.regularizers.l2(l2_reg) if l2_reg is not None else None
    x = layers.Conv1D(filters, kernel_size, kernel_regularizer=l2_reg, **cnn_config)(input)
    if dropout is not None:
        x = MCSpatialDropout1D(rate=dropout)(x)
    if pool_size is not None:
        x = keras.layers.MaxPooling1D(pool_size)(x)

    return x


def bi_lstm(input: layers.Layer, n_units: int,
            l2_reg: float = None,
            **lstm_config):
    l2_reg = keras.regularizers.l2(l2_reg) if l2_reg is not None else None

    x = keras.layers.Bidirectional(
        layer=keras.layers.LSTM(units=n_units, kernel_regularizer=l2_reg, **lstm_config)
    )(input)

    return x
