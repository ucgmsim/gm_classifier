import shutil
import fnmatch
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Union, Callable, Any, Iterable, List

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from scipy import stats
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


# def train_val_split(
#     train_df: pd.DataFrame,
#     feature_names: List[str],
#     label_names: List[str],
#     val_size: float = 0.25,
# ):
#     """Performs the training & validation data split for the given dataframe
#     Also only retrieves the features/labels of interest"""
#     # Get indices for splitting into training & validation set
#     train_ind, val_ind = train_test_split(
#         np.arange(train_df.shape[0], dtype=int), test_size=val_size
#     )
#
#     # Create training and validation datasets
#     X_train = train_df.loc[:, feature_names].iloc[train_ind].copy()
#     X_val = train_df.loc[:, feature_names].iloc[val_ind].copy()
#
#     y_train = train_df.loc[:, label_names].iloc[train_ind].copy()
#     y_val = train_df.loc[:, label_names].iloc[val_ind].copy()
#
#     ids_train = train_df.iloc[train_ind].index.values
#     ids_val = train_df.iloc[val_ind].index.values
#
#     train_data = (X_train, y_train, ids_train)
#     val_data = (X_val, y_val, ids_val)
#
#     return train_data, val_data


def train(
    output_dir: Path,
    model_type: keras.Model,
    model_config: Dict,
    training_data: Tuple[
        Union[np.ndarray, Dict[str, np.ndarray]], np.ndarray, np.ndarray
    ],
    val_data: Union[
        None, Tuple[Union[np.ndarray, Dict[str, np.ndarray]], np.ndarray, np.ndarray]
    ] = None,
    compile_kwargs: Dict[str, Any] = None,
    fit_kwargs: Dict[str, Any] = None,
) -> Tuple[Dict, keras.Model]:
    """
    Performs the training for the specified
    training and validation data

    Parameters
    ----------
    output_dir: str
        Path to directory where results are saved
    model_config: dictionary with string keys
        Dictionary that contains model details
    training_data: triplet
        Training data, expected tuple data:
        (X_train, y_train, ids_train)
    val_data: triplet
        Validation data, expected tuple data:
        (X_train, y_train, ids_train)
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
    """
    # Unroll training & validation data
    X_train, y_train, ids_train = training_data
    X_val, y_val, ids_val = val_data if val_data is not None else (None, None, None)

    # Save training and validation records
    np.save(output_dir / "train_ids.npy", ids_train)
    if ids_val is not None:
        np.save(output_dir / "val_ids.npy", ids_val)

    # Build the model architecture
    gm_model = model_type.from_custom_config(model_config)

    # Train the model
    tensorboard_output_dir = output_dir / "tensorboard_log"
    if tensorboard_output_dir.is_dir():
        shutil.rmtree(tensorboard_output_dir)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(output_dir / "model.h5"), save_best_only=True, save_weights_only=True
        ),
        keras.callbacks.TensorBoard(tensorboard_output_dir, write_images=True),
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

    return history.history, gm_model


def apply_pre(
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
        train_data.loc[:, std_keys] = pre.standardise(
            train_data.loc[:, std_keys], mu, sigma
        )
        val_data.loc[:, std_keys] = (
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
        assert np.all(
            np.isclose(
                np.cov(train_data.loc[:, whiten_keys], rowvar=False),
                np.identity(len(whiten_keys)),
            )
        )

        # Save whitening matrix
        np.save(output_dir / f"{output_prefix}W.npy", W)

    cust_func_keys = pre.get_custom_fn_keys(feature_config)
    if len(whiten_keys) > 0:
        for cur_key in cust_func_keys:
            cur_fn = feature_config[cur_key]
            cur_key = (
                _match_keys(cur_key, train_data.columns) if "*" in cur_key else cur_key
            )
            train_data.loc[:, cur_key] = cur_fn(train_data.loc[:, cur_key].values)
            val_data.loc[:, cur_key] = cur_fn(val_data.loc[:, cur_key].values)

    return train_data, val_data


def _match_keys(key_filter: str, columns: np.ndarray):
    return fnmatch.filter(columns, key_filter)


def mape(y_true, y_pred):
    """Mean absolute percentage error"""
    return tf.reduce_sum(tf.abs(y_true - y_pred) / y_true, axis=1)


def f_min_loss_weights(x):
    x_min, x_max = -6.0, 3.0
    y_min = 1.961_839_939_667_620_5e-06
    y_max = 0.989_491_871_886_240_7

    x_s = x * (x_max - x_min) + x_min
    y = stats.norm.cdf(x_s, 0, 1.3)

    return (y - y_min) / (y_max - y_min)


class CustomLoss(keras.losses.Loss):
    """Custom loss function that weights the f_min loss
    based on the (true) score value, as f_min is not relevant for
    records with a low quality score

    Note: Expects the following format for y: [n_samples, 6],
    where the columns have to be in the order
    [score_X, f_min_X, score_Y, f_min_Y, score_Z, f_min_Z]
    """

    def __init__(self, scores: np.ndarray, weights: np.ndarray, **kwargs):
        super().__init__(**kwargs)

        scores = tf.constant((scores * 4), tf.int32)
        values = tf.constant(weights, tf.float32)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(scores, values),
            default_value=tf.constant(np.nan),
        )

    def call(self, y_true, y_pred):
        # Get the weights from the lookup table
        f_min_weights = self.table.lookup(
            tf.cast(tf.gather(y_true, [0, 2, 4], axis=1) * 4, tf.int32)
        )

        # Score weights are hard-coded to 1 at this stage
        score_weights = tf.ones_like(f_min_weights)

        # Combine the weights
        weights = tf.stack(
            (
                score_weights[:, 0],
                f_min_weights[:, 0],
                score_weights[:, 1],
                f_min_weights[:, 1],
                score_weights[:, 2],
                f_min_weights[:, 2],
            ),
            axis=1,
        )

        # Compute MSE and apply the weights
        return weights * tf.math.squared_difference(y_pred, y_true)
