import json
import os
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

from . import features
from . import model
from . import pre_processing as pre


def train(
    output_dir: str,
    features_df: pd.DataFrame,
    label_df: pd.DataFrame,
    config: Dict,
    record_ids_filter: np.ndarray = None,
    val_split: float = 0.1,
    score_th: Tuple[float, float] = (0.01, 0.99),
) -> None:
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
    """
    # Get the training data
    train_df = pd.merge(features_df, label_df, left_index=True, right_index=True)

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

    # Get labels from scores
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
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(X, y, ids)
        print(
            f"Labelled data split into {X_train.shape[0]} training "
            f"and {X_val.shape[0]} validation samples"
        )
    else:
        print("No validation data used")
        X_train, y_train, ids_train = X, y, ids
        X_val, y_val, ids_val = None, None, None

    # Save training and validation records
    np.save(os.path.join(output_dir, "train_records_ids.npy"), ids_train)
    if ids_val is not None:
        np.save(os.path.join(output_dir, "val_records_ids.npy"), ids_val)

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
    callbacks = [keras.callbacks.ModelCheckpoint(os.path.join(output_dir, "model.h5"))]
    gm_model.compile(
        optimizer=train_config["optimizer"],
        loss=train_config["loss"],
        metrics=["accuracy"],
    )
    history = gm_model.fit(
        X_train,
        y_train,
        batch_size=train_config["batch_size"],
        epochs=train_config["n_epochs"],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    # Save the history
    hist_df = pd.DataFrame.from_dict(history.history, orient="columns")
    hist_df.to_csv(os.path.join(output_dir, "history.csv"), index_label="epoch")

    # Save the model
    gm_model.save(os.path.join(output_dir, "model.h5"))

    # Save the config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f)


def _apply_pre(
    X_train: np.ndarray, pre_config: Dict, output_dir: str, X_val: np.ndarray = None
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
        np.save(os.path.join(output_dir, "mu.npy"), mu)
        np.save(os.path.join(output_dir, "sigma.npy"), sigma)

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
        np.save(os.path.join(output_dir, "W.npy"), W)

    return X_train, X_val
