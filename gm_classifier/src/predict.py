import os

import numpy as np
import pandas as pd

from . import model
from . import features
from . import pre_processing as pre


def run_original(model_dir: str, model_name: str, input_df: pd.DataFrame) -> pd.DataFrame:
    """Runs the specified original model for the given input data

    Parameters
    ----------
    model_dir: string
        Path to the directory that contains all the original model's information,
        i.e. weights, bias, mu, sigma, etc.
    model_name: string
        Name of the model, required for correct pre-processing
        Either canterbury or canterbury_wellington
    input_data_ffp: string
        Path to the input data to use

    Returns
    -------
    pandas dataframe
        The probability of each record being
        either high or low quality
        Columns: [y_low, y_high]
    """
    # Load the model
    cur_model = model.get_orig_model(model_dir)

    # Pre-processing of input data
    input_data = input_df.loc[:, features.FEATURE_NAMES].values

    mu_sigma = np.loadtxt(os.path.join(model_dir, "mu_sigma.csv"), delimiter=",")
    mu, sigma = mu_sigma[0, :], mu_sigma[1, :]

    m = np.loadtxt(os.path.join(model_dir, "M.csv"), delimiter=",")

    input_data = pre.apply_pre_original(model_name, input_data, mu, sigma, m)

    result = cur_model.predict(input_data)
    result_df = pd.DataFrame(data=result, columns=["y_low", "y_high"])

    return result_df
