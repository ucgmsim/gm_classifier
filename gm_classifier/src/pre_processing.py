from dataclasses import dataclass
from typing import Union, Dict

import pandas as pd
import numpy as np
from sklearn import model_selection


@dataclass
class StandardParams:
    mu: pd.Series
    sigma: pd.Series


@dataclass
class PreParams:
    std_params: StandardParams = None


def run_preprocessing(X: pd.DataFrame, feature_config: Dict, params: PreParams = None):
    """Runs the pre-processing"""
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
    """Splits the records into training and validation records
    Note: Split is done based on records, not component of records to ensure
    that training and validations are independent
    """
    index = dfs[0].index
    assert np.all([np.all(cur_df.index == index) for cur_df in dfs])

    train_ids, val_ids = model_selection.train_test_split(
        np.unique(index.values.astype(str)), test_size=0.2, random_state=12
    )

    result = []
    for cur_df in dfs:
        result.append(cur_df.loc[train_ids])
        result.append(cur_df.loc[val_ids])

    return tuple(result + [train_ids, val_ids])


def get_standard_keys(config: Dict):
    return [
        key
        for key, val in config.items()
        if (isinstance(val, str) or isinstance(val, list)) and "standard" in val
    ]


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
