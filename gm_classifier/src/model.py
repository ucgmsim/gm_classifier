import os
from typing import List, Dict, Tuple, Union, Callable

import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


class GenericLayer(layers.Layer):
    def __init__(
        self, layer_config: List[Tuple[layers.Layer, Dict]], name: str, **kwargs
    ):
        super(GenericLayer, self).__init__(name=name, **kwargs)
        self.layer_config = layer_config

        self._layers = []

        # Build the layer
        self._build()

    def _build(self):
        for layer_type, layer_kwargs in self.layer_config:
            self._layers.append(layer_type(**layer_kwargs))

    def call(self, inputs, training=None, **kwargs):
        x = self._layers[0](inputs)
        for cur_layer in self._layers[1:]:
            x = cur_layer(x, training=training)

        return x


class DenseModel(keras.models.Model):
    def __init__(
        self,
        name: str,
        layer_config: List[Tuple[layers.Layer, Dict]],
        n_outputs: int,
        output_act: str = None,
        **kwargs
    ):
        super(DenseModel, self).__init__(name, **kwargs)

        self.dense = GenericLayer(layer_config, "Dense")
        self.dense_output = layers.Dense(n_outputs, activation=output_act)

    def call(self, inputs, training=None, mask=None):
        x = self.dense(inputs, training=training)
        return self.dense_output(x)


class CnnSnrModel(keras.models.Model):
    def __init__(
        self,
        dense_layer_config: List[Tuple[layers.Layer, Dict]],
        dense_input_name: str,
        cnn_layer_config: List[Tuple[layers.Layer, Dict]],
        cnn_input_name: str,
        comb_layer_config: List[Tuple[layers.Layer, Dict]],
        output: Union[keras.layers.Layer, keras.layers.Layer],
        **kwargs
    ):
        super(CnnSnrModel, self).__init__(**kwargs)

        self.dense_layer_config = dense_layer_config
        self.dense_input_name = dense_input_name

        self.cnn_layer_config = cnn_layer_config
        self.cnn_input_name = cnn_input_name

        self.comb_layer_config = comb_layer_config

        # Build the layers
        self.dense = GenericLayer(self.dense_layer_config, "Dense")
        self.cnn = GenericLayer(self.cnn_layer_config, "CNN")

        self.flatten_cnn = layers.Flatten()
        self.conc = layers.Concatenate()
        self.comb_dense = GenericLayer(self.comb_layer_config, "Dense_Combined")

        self.model_output = output

    def call(self, inputs, training=None, mask=None):
        x_dense = self.dense(inputs[self.dense_input_name], training=training)
        x_cnn = self.cnn(inputs[self.cnn_input_name], training=training)

        x_cnn = self.flatten_cnn(x_cnn)
        x = self.conc([x_dense, x_cnn])
        x = self.comb_dense(x, training=training)

        if isinstance(self.model_output, list):
            result = []
            for cur_output in self.model_output:
                result.append(cur_output(x))
            return tuple(result)
        else:
            return self.model_output(x)

    def save(self, *args, **kwargs):
        print(f"Saving with arguments {args} and kwargs {kwargs}")
        super(CnnSnrModel, self).save(*args, **kwargs)

    @classmethod
    def from_custom_config(cls, model_config: Dict):
        return CnnSnrModel(
            model_config["dense_layer_config"],
            model_config["dense_input_name"],
            model_config["cnn_layer_config"],
            model_config["cnn_input_name"],
            model_config["comb_layer_config"],
            model_config["output"],
        )


# class DenseNN:
#     def __init__(
#         self,
#         n_inputs: int,
#         units: List[int],
#         act_funcs: Union[str, List[str]],
#         n_outputs: int,
#         dropout: Union[List[float], float] = 0.0,
#         output_act_func: Union[str, List[str]] = None,
#     ):
#         self.n_inputs = n_inputs
#         self.units = units
#         self.dropout = dropout if isinstance(dropout, list) else [dropout for ix in range(len(units))]
#         self.act_funcs = act_funcs
#         self.n_outputs = n_outputs
#         self.output_act_func = output_act_func
#
#     def build(self) -> keras.Model:
#         # Input
#         inputs = keras.Input(self.n_inputs)
#
#         # Hidden layers
#         x = None
#         for ix, (n_units, cur_dropout) in enumerate(zip(self.units, self.dropout)):
#             cur_layer = layers.Dense(
#                 n_units,
#                 activation=self.act_funcs[ix]
#                 if isinstance(self.act_funcs, list)
#                 else self.act_funcs,
#             )
#
#             if x is None:
#                 x = cur_layer(inputs)
#             else:
#                 x = cur_layer(x)
#
#             # Add dropout if specified
#             if cur_dropout is not None:
#                 x = layers.Dropout(cur_dropout)(x)
#
#         # Output
#         output = layers.Dense(self.n_outputs, activation=self.output_act_func)(x)
#         return keras.Model(inputs=inputs, outputs=output)
#
#     @classmethod
#     def from_dict(cls, n_inputs: int, model_dict: Dict) -> "DenseNN":
#         return DenseNN(
#             n_inputs,
#             model_dict["units"],
#             model_dict["act_funcs"],
#             model_dict["n_outputs"],
#             dropout=model_dict["dropout"],
#             output_act_func=model_dict["output_act_func"],
#         )
#
