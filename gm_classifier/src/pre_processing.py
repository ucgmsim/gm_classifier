from dataclasses import dataclass
from typing import Union, Tuple, Dict

import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn


@dataclass
class StandardParams:
    mu: pd.Series
    sigma: pd.Series


@dataclass
class PreParams:
    std_params: StandardParams = None


def run_preprocessing(X: pd.DataFrame, feature_config: Dict, params: PreParams = None):
    # Compute the required params
    std_keys = get_standard_keys(feature_config)
    if params is None:
        params = PreParams()

        # Standardisation
        if len(std_keys) > 0:
            # Compute mean and std from training data
            mu, sigma = (
                X.loc[:, std_keys].mean(axis=0),
                X.loc[:, std_keys].std(axis=0),
            )

            params.std_params = StandardParams(mu, sigma)

    # Apply
    if len(std_keys) > 0:
        X.loc[:, std_keys] = standardise(
            X.loc[:, std_keys], params.std_params.mu, params.std_params.sigma
        )

    return X, params


def train_test_split(*dfs):
    index = dfs[0].index
    assert np.all([np.all(cur_df.index == index) for cur_df in dfs])

    train_ids, val_ids = sklearn.model_selection.train_test_split(
        np.unique(index.values.astype(str)), test_size=0.2, random_state=12
    )

    result = []
    for cur_df in dfs:
        result.append(cur_df.loc[train_ids])
        result.append(cur_df.loc[val_ids])

    return tuple(result + [train_ids, val_ids])


# ---- old -----


def standardise(
    data: np.ndarray,
    mu: Union[float, pd.Series, np.ndarray],
    sigma: [float, pd.Series, np.ndarray],
):
    """Standardises the data

    Parameters
    ----------
    data: numpy array of floats
        The data to standardise, can either be 1D or 2D
        If 2D, then the rows are assumed to be the samples
        and the columns the different features along which to
        standardise, i.e. shape = [n_samples, n_features]

    mu: float or numpy array of floats
        The mean value to use
        If data is 2D this has to be a 1D array with each
        entry corresponding to the column in data,
        i.e. shape = [n_features]
    sigma: float or numpy array of floats
        Standard deviation to use, same shape restrictions as
        for mu apply

    Returns
    -------
    data: numpy array of floats
        The standardised data
        """
    return (data - mu) / sigma


def reverse_standardise(
    data: np.ndarray, mu: Union[float, np.ndarray], sigma: Union[float, np.ndarray]
):
    """Transforms the standardised data back to its original values,
    paramers are the same as for standardise function, except that data is the
    standardised data"""
    return (data * sigma) + mu


def whiten(X: np.ndarray, W: np.ndarray):
    """Decorrolates the data using the
    Mahalanobis transform

    Parameters
    ----------
    X: numpy array of floats
    W: numpy array of floats
        Whitening matrix to use

    Returns
    -------
    data: numpy array of floats
        Decorrelated data
    """
    return W.dot(X.T).T


def min_max_scale(
    X: Union[float, np.ndarray],
    range: Tuple[float, float] = (0, 1),
    x_min: float = None,
    x_max: float = None,
):
    """Scales the given data to the specified range, the min & max values of the data can
    either be given or inferred from the data.

    Note: If a float is given, then x_min and x_max have to be set"""
    x_min = x_min if x_min is not None else X.min()
    x_max = x_max if x_max is not None else X.max()

    X_std = (X - x_min) / (x_max - x_min)
    X_scaled = X_std * (range[1] - range[0]) + range[0]

    return X_scaled


@tf.function
def tf_min_max_scale(X, target_min, target_max, x_min, x_max):
    X_std = (X - x_min) / (x_max - x_min)
    X_scaled = X_std * (target_max - target_min) + target_min

    return X_scaled


def compute_W_ZCA(X: np.ndarray) -> np.ndarray:
    """
    Computes the ZCA (or Mahalanobis) whitening matrix

    Parameters
    ----------
    X: numpy array of floats
        The data for which to compute the whitening matrix
        Shape: [n_samples, n_features]

    Returns
    -------
    numpy array of floats
        Whitening matrix of shape [n_features, n_features]
    """
    cov_X = np.cov(X, rowvar=False)

    # Compute square root of cov(X)
    (L, V) = np.linalg.eig(cov_X)
    cov_X_sqrt = V.dot(np.diag(np.sqrt(L))).dot(V.T)

    # Compute inverse
    return np.linalg.inv(cov_X_sqrt)


def apply(
    X: pd.DataFrame,
    config: Dict,
    mu: pd.Series = None,
    sigma: pd.Series = None,
    W: np.ndarray = None,
):
    """Applies the pre-processing

    Parameters
    ----------
    X: dataframe
        Input data
        Shape: [n_samples, n_features]
    mu: dataframe
    sigma: dataframe
        Standardises the data (i.e. zero mean and standard dev of one) (per feature)
        Shape: [n_features]
    W: numpy array of floats
        Whitening matrix, applies whitening if specified
        Shape: [n_wth_feature, n_wth_features]

    Returns
    -------
    numpy array of floats
        Pre-processed input data
    """
    if mu is not None and sigma is not None:
        print("Standardising input data")
        std_features = get_standard_keys(config)
        X.loc[:, std_features] = standardise(X.loc[:, std_features], mu, sigma)

    if W is not None:
        print("Whitening input data")
        whiten_features = get_whiten_keys(config)
        X.loc[:, whiten_features] = whiten(X.loc[:, whiten_features], W)

    return X


def get_whiten_keys(config: Dict):
    return [
        key
        for key, val in config.items()
        if (isinstance(val, str) or isinstance(val, list)) and "whiten" in val
    ]


def get_standard_keys(config: Dict):
    return [
        key
        for key, val in config.items()
        if (isinstance(val, str) or isinstance(val, list)) and "standard" in val
    ]


def get_custom_fn_keys(config: Dict):
    return [
        key for key, val in config.items() if val is not None and callable(config[key])
    ]
