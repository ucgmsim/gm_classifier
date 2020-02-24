import os
from typing import List, Dict, Tuple, Union

import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import gm_classifier as gm

class ModelArchitecture:
    def __init__(
        self,
        n_inputs: int,
        units: List[int],
        act_funcs: Union[str, List[str]],
        n_outputs: int,
        output_act_func: str = None,
    ):
        self.n_inputs = n_inputs
        self.units = units
        self.act_funcs = act_funcs
        self.n_outputs = n_outputs
        self.output_act_func = output_act_func

    def build(self) -> keras.Model:
        # Input
        inputs = keras.Input(self.n_inputs)

        # Hidden layers
        x = None
        for ix, n_units in enumerate(self.units):
            cur_layer = layers.Dense(n_units, activation=self.act_funcs[ix])

            if x is None:
                x = cur_layer(inputs)
            else:
                x = cur_layer(x)

        # Output
        output = layers.Dense(self.n_outputs, activation=self.output_act_func)(x)

        return keras.Model(inputs=inputs, outputs=output)


def get_orig_model(
    model_dir: str
) -> keras.Model:
    with open(os.path.join(model_dir, "masterF.txt"), "r") as f:
        line = f.readline()
    values = [value.strip() for value in line.split(",")]

    # Read the weights
    weights_1 = np.loadtxt(os.path.join(model_dir, "weight_1.csv"), delimiter=",")
    bias_1 = np.loadtxt(os.path.join(model_dir, "bias_1.csv"), delimiter=",")

    weights_out = np.loadtxt(os.path.join(model_dir, "weight_output.csv"), delimiter=",")
    bias_out = np.loadtxt(os.path.join(model_dir, "bias_output.csv"), delimiter=",")

    if len(values) == 7:
        n_inputs = int(values[0])
        n_units_1, act_1 = int(values[1]), values[2]
        n_units_2, act_2 = int(values[3]), values[4]
        n_outputs, act_out = int(values[5]), values[6]

        model_arch = gm.model.ModelArchitecture(
            n_inputs, [n_units_1, n_units_2], [act_1, act_2], n_outputs, act_out
        )

        # Read the extra weights
        weights_2 = np.loadtxt(os.path.join(model_dir, "weight_2.csv"), delimiter=",")
        bias_2 = np.loadtxt(os.path.join(model_dir, "bias_2.csv"), delimiter=",")

        weights = [weights_1, bias_1, weights_2, bias_2, weights_out, bias_out]

    elif len(values) == 5:
        n_inputs = int(values[0])
        n_units_1, act_1 = int(values[1]), values[2]
        n_outputs, act_out = int(values[3]), values[4]

        model_arch = gm.model.ModelArchitecture(
            n_inputs, [n_units_1], [act_1], n_outputs, act_out
        )
        weights = [weights_1, bias_1, weights_out, bias_out]
    else:
        raise Exception(f"Invalid masterF.txt file in model dir {model_dir}")

    model = model_arch.build()
    model.set_weights(weights)

    return model
