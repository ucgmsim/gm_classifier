from typing import Union

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


def decorrelate(data: np.ndarray, m: np.ndarray):
    """Decorrolates the data using the
    Mahalanobis transform

    Parameters
    ----------
    data: numpy array of floats
    m: numpy array of floats
        Mahalanobis matrix to use

    Returns
    -------
    data: numpy array of floats
        Decorrelated data
    """
    return m.dot(data.T).T


def apply_pre_original(
    model_name: str, data: np.ndarray, mu: np.ndarray, sigma: np.ndarray, m: np.ndarray
):
    if model_name == "canterbury":
        data = deskew_canterbury(data)
    elif model_name == "canterbury_wellington":
        data = deskew_canterbury_wellington(data)
    else:
        raise ValueError(
            f"Model name {model_name} is not valid, has to be "
            f"one of ['canterbury', 'canterbury_wellington']"
        )

    data = standardise(data, mu, sigma)
    data = decorrelate(data, m)

    return data
