import json
import os

import numpy as np
import pandas as pd
import tensorflow.keras as keras

from . import model
from . import features
from . import pre_processing as pre


def classify(model_dir: str, input_df: pd.DataFrame) -> pd.DataFrame:
    """Runs classification

    Parameters
    ----------
    model_dir: string
        Path to directory that contains the model and
        pre-processing information as produced by training.train
    input_df: Dataframe
        Feature data, requires columns features.FEATURE_NAMES

    Returns
    -------
    Dataframe
        Same as input dataframe with the additional
        columns 'y_low' and 'y_high'
    """
    # Load the model
    model = keras.models.load_model(os.path.join(model_dir, "model.h5"))

    # Load the config
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)

    # Apply the pre-processing
    X = input_df.loc[:, features.FEATURE_NAMES].values

    mu_ffp = os.path.join(model_dir, "mu.npy")
    sigma_ffp = os.path.join(model_dir, "sigma.npy")
    W_ffp = os.path.join(model_dir, "W.npy")

    pre_config = config["preprocessing"]

    X = pre.apply(
        X,
        deskew=pre_config["deskew"],
        mu=np.load(mu_ffp) if os.path.isfile(mu_ffp) else None,
        sigma=np.load(sigma_ffp) if os.path.isfile(sigma_ffp) else None,
        W=np.load(W_ffp) if os.path.isfile(W_ffp) else None,
    )

    # Run classification
    y = model.predict(X)

    # Create result dataframes
    result_df = input_df.copy()
    result_df["y_low"] = 1 - y
    result_df["y_high"] = y

    return result_df


def classify_original(model_name: str, input_df: pd.DataFrame) -> pd.DataFrame:
    """Runs the specified original model for the given input data

    Parameters
    ----------
    model_name: string
        Name of the model, required for correct pre-processing
        Either canterbury or canterbury_wellington
    input_df: DataFrame
        Dataframe with the features for each record to classify
        Also expects columns event_id, station and record_id as index

    Returns
    -------
    pandas dataframe
        The probability of each record being
        either high or low quality
        Columns: [y_low, y_high]
    """
    file_dir = os.path.dirname(__file__)
    if model_name == "canterbury":
        model_dir = os.path.join(file_dir, "../original_models/Canterbury/model")
    elif model_name == "canterbury_wellington":
        model_dir = os.path.join(
            file_dir, "../original_models/CanterburyWellington/model"
        )
    else:
        raise ValueError(
            f"model_name of {model_name} is not valid, has to be "
            f"one of ['canterbury', 'canterbury_wellington']"
        )

    # Load the model
    cur_model = model.get_orig_model(model_dir)

    # Pre-processing of input data
    input_data = input_df.loc[:, features.FEATURE_NAMES].values

    mu_sigma = np.loadtxt(os.path.join(model_dir, "mu_sigma.csv"), delimiter=",")
    mu, sigma = mu_sigma[0, :], mu_sigma[1, :]

    m = np.loadtxt(os.path.join(model_dir, "M.csv"), delimiter=",")

    input_data = pre.apply_pre_original(model_name, input_data, mu, sigma, m)

    result = cur_model.predict(input_data)
    result_df = input_df.copy()
    result_df["y_low"] = result[:, 0]
    result_df["y_high"] = result[:, 1]

    return result_df
