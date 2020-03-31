import os
from typing import List, Dict, Tuple, Union

import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


class ModelArchitecture:
    def __init__(
        self,
        n_inputs: int,
        units: List[int],
        act_funcs: Union[str, List[str]],
        n_outputs: int,
        dropout: Union[List[float], float] = 0.0,
        output_act_func: Union[str, List[str]] = None,
        output_names: List[str] = None,
    ):
        self.n_inputs = n_inputs
        self.units = units
        self.dropout = dropout if isinstance(dropout, list) else [dropout for ix in range(len(units))]
        self.act_funcs = act_funcs
        self.n_outputs = n_outputs
        self.output_act_func = output_act_func

        if n_outputs > 1 and (output_names is None or len(output_names) != n_outputs):
            raise ValueError(
                "If the model has more than one target variable, the output names "
                "have to be specified (in the same order as "
                "in the labelled data (along axis 1))"
            )
        self.output_names = output_names

    def build(self) -> keras.Model:
        # Input
        inputs = keras.Input(self.n_inputs)

        # Hidden layers
        x = None
        for ix, (n_units, cur_dropout) in enumerate(zip(self.units, self.dropout)):
            cur_layer = layers.Dense(
                n_units,
                activation=self.act_funcs[ix]
                if isinstance(self.act_funcs, list)
                else self.act_funcs,
            )

            if x is None:
                x = cur_layer(inputs)
            else:
                x = cur_layer(x)

            # Add dropout if specified
            if cur_dropout is not None:
                x = layers.Dropout(cur_dropout)(x)

        # Output
        if self.n_outputs == 1:
            output = layers.Dense(1, activation=self.output_act_func)(x)
            return keras.Model(inputs=inputs, outputs=output)
        else:
            self.output_act_func = (
                self.output_act_func
                if isinstance(self.output_act_func, list)
                else [self.output_act_func for ix in range(self.n_outputs)]
            )
            outputs = [
                layers.Dense(1, activation=cur_act_func, name=cur_name)(x)
                for cur_act_func, cur_name in zip(
                    self.output_act_func, self.output_names
                )
            ]
            return keras.Model(inputs=inputs, outputs=outputs)

    @classmethod
    def from_dict(cls, n_inputs: int, model_dict: Dict) -> "ModelArchitecture":
        return ModelArchitecture(
            n_inputs,
            model_dict["units"],
            model_dict["act_funcs"],
            model_dict["n_outputs"],
            dropout=model_dict["dropout"],
            output_act_func=model_dict["output_act_func"],
            output_names=model_dict["output_names"]
        )
