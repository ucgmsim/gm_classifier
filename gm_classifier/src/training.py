import json
import os
from pathlib import Path
from typing import Dict, Tuple, Union, Callable, Any, Iterable, List

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

from . import features
from . import model
from . import pre_processing as pre


def get_multi_output_y(
    df: pd.DataFrame, label_names: Iterable[str], ind: np.ndarray = None
):
    """
    Creates y in the correct for model.fit when using a multi-output model

    Parameters
    ----------
    df: dataframe
        Dataframe that contains the labels for each sample
    label_names: iterable of strings
        The labels to be predicted by the model
    ind: np.ndarray, optional
        The indices of the samples from the dataframe to use

    Returns
    -------
    dictionary
    """
    y = {}
    for cur_label_name in label_names:
        data = (
            df[cur_label_name].values
            if ind is None
            else df[cur_label_name].iloc[ind].values
        )
        y[cur_label_name] = data

    return y


def train_val_split(
    train_df: pd.DataFrame,
    feature_names: List[str],
    label_names: List[str],
    val_size: float = 0.25,
):
    """Performs the training & validation data split for the given dataframe
    Also only retrieves the features/labels of interest"""
    # Get indices for splitting into training & validation set
    train_ind, val_ind = train_test_split(
        np.arange(train_df.shape[0], dtype=int), test_size=val_size
    )

    # Create training and validation datasets
    X_train = train_df.loc[:, feature_names].iloc[train_ind]
    X_val = train_df.loc[:, feature_names].iloc[val_ind]

    y_train = train_df.loc[:, label_names].iloc[train_ind]
    y_val = train_df.loc[:, label_names].iloc[val_ind]

    ids_train = train_df.iloc[train_ind].index.values
    ids_val = train_df.iloc[val_ind].index.values

    train_data = (X_train, y_train, ids_train)
    val_data = (X_val, y_val, ids_val)

    return train_data, val_data


def train(
    output_dir: Path,
    feature_config: Dict[str, Any],
    model_config: Dict[str, Any],
    training_data: Tuple[pd.DataFrame, np.ndarray, np.ndarray],
    val_data: Union[None, Tuple[pd.DataFrame, np.ndarray, np.ndarray]] = None,
    label_config: Dict[str, Any] = None,
    compile_kwargs: Dict[str, Any] = None,
    fit_kwargs: Dict[str, Any] = None,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs the training for the specified
    training and validation data

    Parameters
    ----------
    output_dir: str
        Path to directory where results are saved
    feature_config: dictionary with string keys
        Dictionary that contains preprocessing details for features
        Has to be per feature, valid keys are:
        - "standard" (standardises features to zero mean and unit variance)
        - None
    model_config: dictionary with string keys
        Dictionary that contains model details
    training_data: triplet
        Training data, expected tuple data:
        (X_train, y_train, ids_train)
    val_data: triplet
        Validation data, expected tuple data:
        (X_train, y_train, ids_train)
    label_config: dictionary with string keys
        Dictionary that contains preprocessing details for labels
    compile_kwargs: dictionary
        Keyword arguments for the model compile,
        see https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
    fit_kwargs: dictionary
        Keyword arguments for the model fit functions,
        see https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit

    Returns
    -------
    dictionary:
        The training history
    X_train: dataframe
        The pre-precossed training data
    X_val: dataframe
        The pre-precossed validation data
    y_train: dataframe
    y_val: dataframe
        If label pre-processing is specified, then these
        are the pre-processed training & validation labels.
        Otherwise just the passed labels
    """
    # Unroll training & validation data
    X_train, y_train, ids_train = training_data
    X_val, y_val, ids_val = val_data if val_data is not None else (None, None, None)

    # Save training and validation records
    np.save(output_dir / "train_ids.npy", ids_train)
    if ids_val is not None:
        np.save(output_dir / "val_ids.npy", ids_val)

    # Apply the pre-processing
    X_train, X_val = _apply_pre(
        X_train, feature_config, output_dir, val_data=X_val, output_prefix="feature"
    )
    if label_config is not None:
        y_train, y_val = _apply_pre(
            y_train, label_config, output_dir, val_data=y_val, output_prefix="label"
        )

    # Build the model architecture
    model_arch = model.ModelArchitecture.from_dict(X_train.shape[1], model_config)

    # Get the model
    gm_model = model_arch.build()
    print(gm_model.summary())

    # Train the model
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(output_dir / "model.h5"), save_best_only=True
        )
    ]
    gm_model.compile(**compile_kwargs)
    history = gm_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        **fit_kwargs,
    )

    # Save the history
    hist_df = pd.DataFrame.from_dict(history.history, orient="columns")
    hist_df.to_csv(output_dir / "history.csv", index_label="epoch")

    # Save the config
    config = {
        "feature_config": feature_config,
        "model_config": model_config,
        "compiler_kwargs": str(compile_kwargs),
        "fit_kwargs": fit_kwargs,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f)

    return history.history, X_train, X_val, y_train, y_val


def _apply_pre(
    train_data: pd.DataFrame,
    feature_config: Dict,
    output_dir: Path,
    val_data: pd.DataFrame = None,
    output_prefix: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Applies the pre-processing as per the given config"""
    output_prefix = "" if output_prefix is None else f"{output_prefix}_"

    # Pre-processing
    std_keys = pre.get_standard_keys(feature_config)
    if len(std_keys) > 0:
        # Compute mean and std from training data
        mu, sigma = (
            train_data.loc[:, std_keys].mean(axis=0),
            train_data.loc[:, std_keys].std(axis=0),
        )

        # Apply to both train and val data
        train_data = pre.standardise(train_data.loc[:, std_keys], mu, sigma)
        val_data = (
            pre.standardise(val_data.loc[:, std_keys], mu, sigma)
            if val_data is not None
            else val_data
        )

        # Sanity check
        assert np.all(
            np.isclose(
                np.mean(train_data.loc[:, std_keys], axis=0), np.zeros(len(std_keys))
            )
        )
        assert np.all(
            np.isclose(train_data.loc[:, std_keys].std(axis=0), np.ones(len(std_keys)))
        )

        # Save mu and sigma
        mu.to_csv(output_dir / f"{output_prefix}mu.csv")
        sigma.to_csv(output_dir / f"{output_prefix}sigma.csv")

    whiten_keys = pre.get_whiten_keys(feature_config)
    if len(whiten_keys) > 0:
        # Compute whitening matrix
        W = pre.compute_W_ZCA(train_data.loc[:, whiten_keys].values)

        # No idea if this is legit...
        W = W.astype(float)

        # Apply
        train_data.loc[:, whiten_keys] = pre.whiten(
            train_data.loc[:, whiten_keys].values, W
        )
        val_data.loc[:, whiten_keys] = (
            pre.whiten(val_data.loc[:, whiten_keys].values, W)
            if val_data is not None
            else val_data
        )

        # Sanity check
        # assert np.all(
        #     np.isclose(np.cov(train_data, rowvar=False), np.identity(n_features))
        # )

        # Save whitening matrix
        np.save(output_dir / f"{output_prefix}W.npy", W)

    # shift = feature_config.get("shift")
    # if shift is not None:
    #     assert len(shift) == train_data.shape[1]
    #     shift = np.asarray(shift)
    #
    #     train_data = train_data + shift
    #     val_data = val_data + shift

    return train_data, val_data


def mape(y_true, y_pred):
    """Mean absolute percentage error"""
    return tf.reduce_sum(tf.abs(y_true - y_pred) / y_true, axis=1)
