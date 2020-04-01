from typing import Union, Tuple

import numpy as np


def standardise(
    data: np.ndarray, mu: Union[float, np.ndarray], sigma: [float, np.ndarray]
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


def apply(
    X: np.ndarray, mu: np.ndarray = None, sigma: np.ndarray = None, W: np.ndarray = None
):
    """Applies the pre-processing

    Parameters
    ----------
    X: numpy array of floats
        Input data
        Shape: [n_samples, n_features]
    deskew: str or bool, optional
        Applies either 'canterbury' or 'canterbury_wellington' if specified
    mu: numpy array of floats
    sigma: numpy array of floats
        Standardises the data (i.e. zero mean and standard dev of one) (per feature)
        Shape: [n_features]
    W: numpy array of floats
        Whitening matrix, applies whitening if specified
        Shape: [n_feature, n_features]

    Returns
    -------
    numpy array of floats
        Pre-processed input data
    """
    if mu is not None and sigma is not None:
        print("Standardising input data")
        X = standardise(X, mu, sigma)

    if W is not None:
        print("Whitening input data")
        X = whiten(X, W)

    return X

