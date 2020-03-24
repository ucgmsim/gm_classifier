from typing import Union, Tuple

import numpy as np


def deskew_canterbury(data):
    """Deskews the data based on a Tukey's ladder of powers

    Note: This is only valid for the Canterbury model
    due to the specific coefficients used

    Parameters
    ----------
    data: numpy array of floats

    Returns
    -------
    data: numpy array of floats
    """
    data[:, [0, 1, 11, 15, 16]] = np.log(data[:, [0, 1, 11, 15, 16]])
    data[:, 17] = -1.0 / data[:, 17] ** 1.2
    data[:, 2] = data[:, 2] ** (-0.2)
    data[:, 10] = data[:, 10] ** (-0.06)
    data[:, 19] = data[:, 19] ** 0.43
    data[:, 7] = data[:, 7] ** 0.25
    data[:, 8] = data[:, 8] ** 0.23
    data[:, 9] = data[:, 9] ** 0.05
    data[:, 18] = data[:, 18] ** 0.33
    data[:, 3] = data[:, 3] ** 0.12
    data[:, 5] = data[:, 5] ** 0.48
    data[:, 6] = data[:, 6] ** 0.37
    data[:, 12] = data[:, 12] ** 0.05
    data[:, 13] = data[:, 13] ** 0.08
    data[:, 4] = data[:, 4] ** 0.16
    data[:, 14] = data[:, 14] ** 0.1

    return data


def deskew_canterbury_wellington(data):
    """Deskews the data based on a Tukey's ladder of powers

    Note: This is only valid for the Canterbury-Wellington model
    due to the specific coefficients used

    Parameters
    ----------
    data: numpy array of floats

    Returns
    -------
    data: numpy array of floats
    """
    data[:, [0, 1, 11, 15, 16]] = np.log(data[:, [0, 1, 11, 15, 16]])
    data[:, 17] = -1.0 / data[:, 17] ** 1.2
    data[:, 2] = data[:, 2] ** (-0.2)
    data[:, 10] = data[:, 10] ** (-0.06)
    data[:, 19] = data[:, 19] ** 0.43
    data[:, 7] = data[:, 7] ** 0.1
    data[:, 8] = data[:, 8] ** 0.23
    data[:, 9] = data[:, 9] ** 0.2
    data[:, 18] = data[:, 18] ** 0.33
    data[:, 3] = data[:, 3] ** 0.05
    data[:, 5] = data[:, 5] ** 0.3
    data[:, 6] = data[:, 6] ** 0.37
    data[:, 12] = data[:, 12] ** 0.05
    data[:, 13] = data[:, 13] ** 0.08
    data[:, 4] = data[:, 4] ** 0.05
    data[:, 14] = data[:, 14] ** 0.05

    return data


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


def get_label_from_score(
    scores: np.ndarray, high_th: float = 0.99, low_th: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """Gets the binary label from the score
    I.e.
        score > high_th -> 1
        score < low_th -> 0
    All records in range low_th < score < high_th are dropped


    Parameters
    ----------
    scores: numpy array of floats
        One dimensional array of scores to convert to labels
    high_th: float
        Score threshold for high quality records
    low_th
        Score threshold for low quality records

    Returns
    -------
    numpy array of ints
        The labelled records
    numpy array of bools
        Mask indicating which records to keep as
        their scores meet the specified high/low
        score threshold
    """
    labels = scores.copy()
    labels[scores > high_th] = 1
    labels[scores < low_th] = 0

    mask = np.isin(labels, [0.0, 1.0])
    return labels[mask].astype(int), mask


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
    X: np.ndarray,
    deskew: Union[str, bool] = False,
    mu: np.ndarray = None,
    sigma: np.ndarray = None,
    W: np.ndarray = None,
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
    if deskew is not False:
        print(f"Applying {deskew} deskew")
        if deskew == "canterbury":
            X = deskew_canterbury(X)
        elif deskew == "canterbury_wellington":
            X = deskew_canterbury_wellington(X)
        else:
            raise ValueError(f"{deskew} is not a valid deskew value")

    if mu is not None and sigma is not None:
        print("Standardising input data")
        X = standardise(X, mu, sigma)

    if W is not None:
        print("Whitening input data")
        X = whiten(X, W)

    return X


def apply_pre_original(
    model_name: str, X: np.ndarray, mu: np.ndarray, sigma: np.ndarray, W: np.ndarray
):
    """Applies pre-processing for an original model"""
    if model_name == "canterbury":
        X = deskew_canterbury(X)
    elif model_name == "canterbury_wellington":
        X = deskew_canterbury_wellington(X)
    else:
        raise ValueError(
            f"Model name {model_name} is not valid, has to be "
            f"one of ['canterbury', 'canterbury_wellington']"
        )

    X = standardise(X, mu, sigma)
    X = whiten(X, W)

    return X
