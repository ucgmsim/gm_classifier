from typing import Union, Tuple, Dict

import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale

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

def scale_snr_values(X: np.ndarray, **kwargs):
    """Min-max scales the SNR values along the frequency axis"""
    return minmax_scale(X, axis=1)


def apply(
    X: pd.DataFrame,
    config: Dict = None,
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
    return [key for key, val in config.items() if (isinstance(val, str) or isinstance(val, list)) and "whiten" in val]


def get_standard_keys(config: Dict):
    return [key for key, val in config.items() if (isinstance(val, str) or isinstance(val, list)) and "standard" in val]


def get_custom_fn_keys(config: Dict):
    return [key for key, val in config.items() if val is not None and callable(config[key])]
