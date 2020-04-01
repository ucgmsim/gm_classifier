import json
import os
from pathlib import Path
from typing import Dict, Tuple, Union, Callable, Any, Iterable

import pandas as pd
import numpy as np
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


def train(
    output_dir: Path,
    feature_pre_config: Dict[str, Any],
    model_config: Dict[str, Any],
    training_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    val_data: Union[
        None, Tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = None,
    label_pre_config: Dict[str, Any] = None,
    compile_kwargs: Dict[str, Any] = None,
    fit_kwargs: Dict[str, Any] = None,
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs the training for the specified
    training and validation data

    Parameters
    ----------
    output_dir: str
        Path to directory where results are saved
    feature_pre_config: dictionary with string keys
        Dictionary that contains preprocessing details for features
    model_config: dictionary with string keys
        Dictionary that contains model details
    training_data: triplet of numpy arrays
        Training data, expected tuple data:
        (X_train, y_train, ids_train)
    val_data: triplet of numpy arrays, optional
        Validation data, expected tuple data:
        (X_train, y_train, ids_train)
    label_pre_config: dictionary with string keys
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
    X_train: numpy array of floats
        The pre-precossed training data
    X_val: numpy array of floats
        The pre-precossed validation data
    y_train: numpy array of floats
    y_val: numpy array of floats
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
        X_train, feature_pre_config, output_dir, val_data=X_val, output_prefix="feature"
    )
    if label_pre_config is not None:
        y_train, y_val = _apply_pre(
            y_train, label_pre_config, output_dir, val_data=y_val, output_prefix="label"
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
        "pre_config": feature_pre_config,
        "model_config": model_config,
        "compiler_kwargs": str(compile_kwargs),
        "fit_kwargs": fit_kwargs,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f)

    return history.history, X_train, X_val, y_train, y_val


def _apply_pre(
    train_data: np.ndarray,
    pre_config: Dict,
    output_dir: Path,
    val_data: np.ndarray = None,
    output_prefix: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies the pre-processing as per the given config"""
    n_features = train_data.shape[1]
    output_prefix = "" if output_prefix is None else f"{output_prefix}_"

    # Pre-processing
    standardise = pre_config.get("standardise")
    if standardise is True:
        # Compute mean and std from training data
        mu, sigma = np.mean(train_data, axis=0), np.std(train_data, axis=0)

        # Apply to both train and val data
        train_data = pre.standardise(train_data, mu, sigma)
        val_data = (
            pre.standardise(val_data, mu, sigma) if val_data is not None else val_data
        )

        # Sanity check
        assert np.all(np.isclose(np.mean(train_data, axis=0), np.zeros(n_features)))
        assert np.all(np.isclose(np.std(train_data), np.ones(n_features)))

        # Save mu and sigma
        np.save(output_dir / f"{output_prefix}mu.npy", mu)
        np.save(output_dir / f"{output_prefix}sigma.npy", sigma)

    whiten = pre_config.get("whiten")
    if whiten is True:
        # Compute whitening matrix
        W = pre.compute_W_ZCA(train_data)

        # Apply
        train_data = pre.whiten(train_data, W)
        val_data = pre.whiten(val_data, W) if val_data is not None else val_data

        # Sanity check
        assert np.all(
            np.isclose(np.cov(train_data, rowvar=False), np.identity(n_features))
        )

        # Save whitening matrix
        np.save(output_dir / f"{output_prefix}W.npy", W)

    shift = pre_config.get("shift")
    if shift is not None:
        assert len(shift) == train_data.shape[1]
        shift = np.asarray(shift)

        train_data = train_data + shift
        val_data = val_data + shift

    return train_data, val_data


def __old_run_training(
    output_dir: Path,
    features_df: pd.DataFrame,
    label_df: pd.DataFrame,
    config: Dict,
    record_ids_filter: np.ndarray = None,
    val_split: float = 0.1,
    sample_weight_fn: Callable[
        [pd.DataFrame, np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ] = None,
    verbose: int = 2,
) -> Tuple[
    pd.DataFrame,
    Dict,
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Trains a model and saves the results
    in the specified output directory

    Parameters
    ----------
    output_dir: str
        Path to directory where results are saved
    features_df: DataFrame
        Features for each sample, requires columns as
        specified in the provided config
        The index has to be the record_id
    label_df: DataFrame
        Labels for each sample, can be more than one
        label for a single sample (i.e. multi-output NN)
        All columns in this dataframe are considered labels!
    config: dictionary with string keys
        Dictionary that contains the model architecture,
        pre-processing & training details
        See train_config.json for an example
    record_ids_filter: numpy array of strings, optional
        All records that should be used for training & validation
        If not specified then all labelled data is used
    val_split: float, optional
        The proportion of data to use for validation (default 0.1)
    sample_weight_fn: callable
        Function that computes the weight for each training sample, must take
        the following inputs: training_df, X_train, y_train, ids_train
        and return a 1-D array of size: X_train.shape[0]
    verbose: int, optional
        Verbosity level for keras training,
        see verbose parameter for
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    """
    # Get the training data
    labels = label_df.columns.values.astype(str)
    train_df = pd.merge(features_df, label_df, left_index=True, right_index=True)
    train_df.to_csv(output_dir / "training_data.csv")

    # Some sanity checks
    feature_names = config["features"]
    assert np.all(
        np.isin(feature_names, features_df.columns.values)
    ), "Not all features are in the feature dataframe"

    if record_ids_filter is not None:
        # Apply the filter
        filter_mask = np.isin(train_df.record_id.values, record_ids_filter)
        train_df = train_df.loc[filter_mask, :]

        # Print out any records for which data is missing
        missing_records = record_ids_filter[
            ~np.isin(record_ids_filter, train_df.record_id.values)
        ]
        if missing_records.size > 0:
            print(
                "No training data was found for the following records:\n{}".format(
                    "\n".join([record for record in missing_records])
                )
            )

    # Get training data
    y = train_df.loc[:, labels].values
    X = train_df.loc[:, feature_names].values
    ids = train_df.index.values[:].astype(str)

    # Sanity check
    assert (
        y.shape[0] == X.shape[0] == ids.shape[0]
    ), "Shapes of X, y and ids have to match!"
    print(f"Number of total labelled samples - {y.shape[0]}")

    # Training & validation data split
    if val_split > 0:
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            X, y, ids, test_size=val_split
        )
        print(
            f"Labelled data split into {X_train.shape[0]} training "
            f"and {X_val.shape[0]} validation samples"
        )
    else:
        print("No validation data used")
        X_train, y_train, ids_train = X, y, ids
        X_val, y_val, ids_val = None, None, None

    sample_weights = (
        None
        if sample_weight_fn is None
        else sample_weight_fn(train_df, X_train, y_train, ids_train)
    )

    history, X_train, X_val = train(
        output_dir,
        config,
        (X_train, y_train, ids_train),
        val_data=(X_val, y_val, ids_val),
        sample_weights=sample_weights,
        verbose=verbose,
    )

    return train_df, history, (X_train, y_train, ids_train), (X_val, y_val, ids_val)
