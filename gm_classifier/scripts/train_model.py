import os
import json
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

import gm_classifier as gm


def train(
    output_dir: str,
    features_ffp: str,
    label_ffp: str,
    config_ffp: str,
    record_list_ffp: str = None,
    val_split: float = 0.1,
):
    # Get the training data
    features_df = pd.read_csv(features_ffp, index_col="record_id")
    label_df = pd.read_csv(label_ffp, index_col="record_id")

    train_df = pd.merge(features_df, label_df, left_index=True, right_index=True)

    n_features = len(gm.features.FEATURE_NAMES)

    # Some sanity checks
    assert np.all(np.isin(gm.features.FEATURE_NAMES, features_df.columns.values))

    if record_list_ffp is not None:
        # Get the records
        record_ids_filter = gm.records.get_record_ids_filter(record_list_ffp)

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

    X = train_df.loc[:, gm.features.FEATURE_NAMES]
    y = gm.pre.get_label_from_score(train_df.loc[:, "score"].values)
    ids = train_df.loc[:, "record_ids"].values.astype(str)

    # Training & validation data split
    if val_split > 0:
        X_train, y_train, ids_train, X_val, y_val, ids_val = train_test_split(X, y, ids)
    else:
        X_train, y_train, ids_train = X, y, ids
        X_val, y_val, ids_val = None, None, None

    # Read the model & training config
    with open(config_ffp, "r") as f:
        config = json.load(f)

    # Pre-processing
    deskew = str(config["deskew"]).strip()
    if deskew is not None:
        if deskew == "canterbury":
            deskew_fn = gm.pre.deskew_canterbury
        elif deskew == "canterbury_wellington":
            deskew_fn = gm.pre.deskew_canterbury_wellington
        else:
            print(
                f"Value of {deskew} for 'deskew' is not valid, "
                f"has to be one of ['canterbury', 'canterbury_wellington']."
                f"No deskew applied."
            )
            deskew_fn = None

        X_train = deskew_fn(X_train) if deskew is not None else X_train
        X_val = deskew_fn(X_val) if deskew is not None else X_val

        # Require indicating which deskew coefficients were used
        # TODO: Improve this, when generalising deskew approach
        with open(os.path.join(output_dir, "deskew.txt")) as f:
            f.write(deskew)

    standardise = config["standardise"]
    if standardise is True:
        # Compute mean and std from training data
        mu, sigma = np.mean(X_train, axis=1), np.std(X_train, axis=1)

        # Apply to both train and val data
        X_train = gm.pre.standardise(X_train, mu, sigma)
        X_val = gm.pre.standardise(X_val, mu, sigma)

        # Sanity check
        assert np.isclose(np.mean(X_train), 0.0)
        assert np.isclose(np.std(X_train), 1.0)

        # Save mu and sigma
        np.save(os.path.join(output_dir, "mu.npy"), mu)
        np.save(os.path.join(output_dir, "sigma.npy"), sigma)

    whiten = config["whiten"]
    if whiten is True:
        # Compute whitening matrix
        W = gm.pre.compute_W_ZCA(X_train)

        # Apply
        X_train = gm.pre.whiten(X_train, W)
        X_val = gm.pre.whiten(X_val, W)

        # Sanity check
        assert np.isclose(np.cov(X_train, rowvar=False), np.identity(n_features))

        # Save whitening matrix
        np.save(os.path.join(output_dir, "W.npy"), W)

    # Build the model architecture
    model_arch = gm.model.ModelArchitecture.from_dict(
        len(gm.features.FEATURE_NAMES), 1, config["model"]
    )

    # Get the model
    model = model_arch.build()

    # Train the model
    callbacks = [keras.callbacks.ModelCheckpoint(os.path.join(output_dir, "model.h5"))]
    model.compile(optimizer=config["optimizer"], loss=config["loss"])
    model.fit(
        X_train,
        y_train,
        batch_size=config["batch_size"],
        epochs=config["n_epochs"],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("output_dir", help="Output directory", type=str)
    parser.add_argument(
        "features_ffp",
        help="csv file with all the features, "
        "as generated by the 'extract_features' script",
        type=str,
    )
    parser.add_argument(
        "label_ffp",
        help="CSV file with the scores for each record, "
        "required columns: ['record_id', 'score']",
    )
    parser.add_argument(
        "config_ffp", help="Config file, that contains model and training details"
    )
    parser.add_argument(
        "--record_list_ffp",
        type=str,
        help="Path to file that lists all records to use (one per line)",
        default=None,
    )
    parser.add_argument(
        "--val_split",
        type=str,
        help="The proportion of the labelled data to use for validation",
        default=0.1,
    )

    args = parser.parse_args()

    train(
        args.output_dir,
        args.features_ffp,
        args.label_ffp,
        args.config_ffp,
        args.record_list_ffp,
    )