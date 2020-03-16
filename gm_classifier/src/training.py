import json
import os
from pathlib import Path
from typing import Dict, Tuple, Union, Callable

import pandas as pd
import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

from . import features
from . import model
from . import pre_processing as pre


def run_trainining(
    output_dir: Path,
    features_df: pd.DataFrame,
    label_df: pd.DataFrame,
    config: Dict,
    record_ids_filter: np.ndarray = None,
    val_split: float = 0.1,
    score_th: Tuple[float, float] = (0.01, 0.99),
    record_weight_fn: Callable[
        [pd.DataFrame, np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ] = None,
    verbose: int = 2
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
        Features for each record, requires columns as
        specified in features.FEATURE_NAMES
        The index has to be the record_id
    label_df: DataFrame
        Labels for each record, requires column 'score'
        The index has to be the record_id
    config: dictionary with string keys
        Dictionary that contains the model architecture,
        pre-processing & training details
        See train_config.json for an example
    record_ids_filter: numpy array of strings, optional
        All records that should be used for training & validation
        If not specified then all labelled data is used
    val_split: float, optional
        The proportion of data to use for validation (default 0.1)
    score_th: pair of floats
        Low and high score thresholds to use for identifying
        low (high) quality records.
        I.e. all records with score < low_score_th (> high_score_th) will be used as
        low (high) quality record
        Note: All records not falling within those ranges are dropped
    record_weight_fn: callable
        Function that computes the weight for each training record, must take
        the following inputs: training_df, X_train, y_train, ids_train
        and return a 1-D array of size: X_train.shape[0]
    verbose: int, optional
        Verbosity level for keras training,
        see verbose parameter for
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    """
    # Get the training data
    train_df = pd.merge(features_df, label_df, left_index=True, right_index=True)
    train_df.to_csv(output_dir / "training_data.csv")

    # Some sanity checks
    assert np.all(np.isin(features.FEATURE_NAMES, features_df.columns.values))

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
    y, mask = pre.get_label_from_score(
        train_df.loc[:, "score"].values, low_th=score_th[0], high_th=score_th[1]
    )
    X = train_df.loc[mask, features.FEATURE_NAMES].values
    ids = train_df.index.values[mask].astype(str)

    # Sanity check
    assert y.shape[0] == X.shape[0] == ids.shape[0]
    print(
        f"Number of high quality records {np.count_nonzero(y == 1)}, "
        f"low quality records {np.count_nonzero(y==0)}"
    )
    print(f"Number of total labelled records - {y.shape[0]}")

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
        if record_weight_fn is None
        else record_weight_fn(train_df, X_train, y_train, ids_train)
    )

    history, X_train, X_val = train(
        output_dir,
        config,
        (X_train, y_train, ids_train),
        val_data=(X_val, y_val, ids_val),
        sample_weights=sample_weights,
        verbose=verbose
    )

    return train_df, history, (X_train, y_train, ids_train), (X_val, y_val, ids_val)


def train(
    output_dir: Path,
    config: Dict,
    training_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    val_data: Union[None, Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    sample_weights: np.ndarray = None,
    verbose: int = 1,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Performs the training for the specified
    training and validation data

    Parameters
    ----------
    output_dir: str
        Path to directory where results are saved
    config: dictionary with string keys
        Dictionary that contains the model architecture,
        pre-processing & training details
        See train_config.json for an example
    training_data: triplet of numpy arrays
        Training data, expected tuple data:
        (X_train, y_train, ids_train)
    val_data: triplet of numpy arrays, optional
        Validation data, expected tuple data:
        (X_train, y_train, ids_train)
    sample_weights: numpy array of floats
        Weights for each training sample,
        Shape: [n_training_samples]
    verbose: int, optional
        Verbosity level for keras training,
        see verbose parameter for
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    """
    # Unroll training & validation data
    X_train, y_train, ids_train = training_data
    X_val, y_val, ids_val = val_data if val_data is not None else (None, None, None)

    # Sanity check
    if sample_weights is not None:
        assert sample_weights.shape[0] == X_train.shape[0]

    # Save training and validation records
    np.save(output_dir / "train_records_ids.npy", ids_train)
    if ids_val is not None:
        np.save(output_dir / "val_records_ids.npy", ids_val)

    # Load the configs
    train_config = config["training"]
    pre_config = config["preprocessing"]

    # Apply the pre-processing
    X_train, X_val = _apply_pre(X_train, pre_config, output_dir, X_val)

    # Build the model architecture
    model_arch = model.ModelArchitecture.from_dict(
        len(features.FEATURE_NAMES), 1, config["model"]
    )

    # Get the model
    gm_model = model_arch.build()

    # Train the model
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(output_dir / "model.h5"), save_best_only=True
        )
    ]
    gm_model.compile(
        optimizer=train_config["optimizer"],
        loss=train_config["loss"],
        metrics=["accuracy"],
        sample_weight_mode=None,
    )
    history = gm_model.fit(
        X_train,
        y_train,
        batch_size=train_config["batch_size"],
        epochs=train_config["n_epochs"],
        validation_data=(X_val, y_val),
        sample_weight=sample_weights,
        callbacks=callbacks,
        verbose=verbose,
    )

    # Save the history
    hist_df = pd.DataFrame.from_dict(history.history, orient="columns")
    hist_df.to_csv(output_dir / "history.csv", index_label="epoch")

    # Save the model
    gm_model.save(output_dir / "model.h5")

    # Save the config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f)

    return history.history, X_train, X_val


def score_weighting(
    train_df: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    ids_train: np.ndarray,
):
    """Weighs the records based on their score,
    score in:
        [0.0, 1.0]      => 1.0
        [0.25, 0.75]    => 0.75
        [0.5]           => 0.5

    Parameters
    ----------
    train_df: Dataframe
    X_train: numpy array of floats
    y_train: numpy array of floats
    ids_train: numpy array of strings

    Returns
    -------
    numpy array of floats
        The sample weights for each record in X_train (and y_train obviously)
    """
    scores = train_df.loc[ids_train, "score"].values

    weights = np.full(X_train.shape[0], np.nan, dtype=float)
    weights[np.isin(scores, [0.0, 1.0])] = 1.0
    weights[np.isin(scores, [0.25, 0.75])] = 0.75
    weights[scores == 0.5] = 0.5

    assert np.all(~np.isnan(weights))

    return weights


def _apply_pre(
    X_train: np.ndarray, pre_config: Dict, output_dir: Path, X_val: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies the pre-processing as per the given config"""
    n_features = X_train.shape[1]

    # Pre-processing
    deskew = pre_config["deskew"]
    if deskew is not False:
        if deskew == "canterbury":
            deskew_fn = pre.deskew_canterbury
        elif deskew == "canterbury_wellington":
            deskew_fn = pre.deskew_canterbury_wellington
        else:
            print(
                f"Value of {deskew} for 'deskew' is not valid, "
                f"has to be one of ['canterbury', 'canterbury_wellington'] "
                f"or false for no deskew. No deskew applied."
            )
            deskew_fn = None

        X_train = deskew_fn(X_train) if deskew is not None else X_train
        X_val = deskew_fn(X_val) if deskew is not None and X_val is not None else X_val

    standardise = pre_config["standardise"]
    if standardise is True:
        # Compute mean and std from training data
        mu, sigma = np.mean(X_train, axis=0), np.std(X_train, axis=0)

        # Apply to both train and val data
        X_train = pre.standardise(X_train, mu, sigma)
        X_val = pre.standardise(X_val, mu, sigma) if X_val is not None else X_val

        # Sanity check
        assert np.all(np.isclose(np.mean(X_train, axis=0), np.zeros(n_features)))
        assert np.all(np.isclose(np.std(X_train), np.ones(n_features)))

        # Save mu and sigma
        np.save(output_dir / "mu.npy", mu)
        np.save(output_dir / "sigma.npy", sigma)

    whiten = pre_config["whiten"]
    if whiten is True:
        # Compute whitening matrix
        W = pre.compute_W_ZCA(X_train)

        # Apply
        X_train = pre.whiten(X_train, W)
        X_val = pre.whiten(X_val, W) if X_val is not None else X_val

        # Sanity check
        assert np.all(
            np.isclose(np.cov(X_train, rowvar=False), np.identity(n_features))
        )

        # Save whitening matrix
        np.save(output_dir / "W.npy", W)

    return X_train, X_val
