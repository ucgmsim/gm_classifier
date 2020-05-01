import shutil
import fnmatch
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Union, Callable, Any, Iterable, List, Type

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from scipy import stats

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


def fit(
    output_dir: Path,
    model: Union[Type, keras.Model],
    model_config: Dict,
    training_data: Tuple[
        Union[np.ndarray, Dict[str, np.ndarray]], np.ndarray, np.ndarray
    ],
    val_data: Union[
        None, Tuple[Union[np.ndarray, Dict[str, np.ndarray]], np.ndarray, np.ndarray]
    ] = None,
    compile_kwargs: Dict[str, Any] = None,
    fit_kwargs: Dict[str, Any] = None,
    tensorboard_cb_kwargs: Dict[str, Any] = None,
) -> Tuple[Dict, keras.Model]:
    """
    Performs the training for the specified
    training and validation data

    Parameters
    ----------
    output_dir: str
        Path to directory where results are saved
    model: keras model or type of model
        If an keras model instance is passed in, it is used
        If a type is passed in, this is expected to be a sub-class
        of keras.Model and implement the from_custom_config method which
        will be used to create the actual model instance from the model_config
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
    tensorboard_cb_kwargs: dictionary, optional
            Tensorboard callback keyword arguments

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

    # Build the model architecture (if required)
    gm_model = model.from_custom_config(model_config) if isinstance(model, type) else model

    # Train the model
    tensorboard_output_dir = output_dir / "tensorboard_log"
    if tensorboard_output_dir.is_dir():
        shutil.rmtree(tensorboard_output_dir)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(output_dir / "model.h5"), save_best_only=True, save_weights_only=True
        ),
        keras.callbacks.TensorBoard(
            tensorboard_output_dir, write_graph=True, **tensorboard_cb_kwargs
        ),
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
                atol=1e-7,
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


def create_huber(threshold: float = 1.0):
    """Creates a huber loss function using the specified threshold"""
    threshold = (
        threshold
        if isinstance(threshold, tf.Tensor)
        else tf.constant(threshold, dtype=tf.float32)
    )

    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        small_mask = tf.abs(error) < threshold
        return tf.where(
            small_mask,
            tf.square(error) / 2,
            threshold * (tf.abs(error) - 0.5 * threshold),
        )

    return tf.function(huber_fn)


def f_min_loss_weights(x):
    """Arbitrary function that defines f_min loss weights
    Uses normal cdf, not really needed should just define these
    weights manually"""
    x_min, x_max = -6.0, 3.0
    y_min = 1.961_839_939_667_620_5e-06
    y_max = 0.989_491_871_886_240_7

    x_s = x * (x_max - x_min) + x_min
    y = stats.norm.cdf(x_s, 0, 1.3)

    return (y - y_min) / (y_max - y_min)


class WeightedFMinMSELoss(keras.losses.Loss):
    """Custom loss function that weights the f_min MSE loss
    based on the (true) score value, as f_min is not relevant for
    records with a low quality score
    Score loss is also calculated using MSE

    Note: Expects the following format for y: [n_samples, 6],
    where the columns have to be in the order
    [score_X, f_min_X, score_Y, f_min_Y, score_Z, f_min_Z]
    """

    def __init__(
        self,
        scores: np.ndarray,
        f_min_weights: np.ndarray,
        score_weights: np.ndarray = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Keys can't be floats, therefore convert to integer
        scores = tf.constant((scores * 4), tf.int32)

        f_min_weights = tf.constant(f_min_weights, tf.float32)
        self.f_min_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(scores, f_min_weights),
            default_value=tf.constant(np.nan),
        )

        score_weights = (
            tf.constant(score_weights, tf.float32)
            if score_weights is not None
            else np.ones(scores.shape[0], dtype=np.float32)
        )
        self.score_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(scores, score_weights), default_value=1
        )

    def call(self, y_true, y_pred):
        score_keys = tf.cast(tf.gather(y_true, [0, 2, 4], axis=1) * 4, tf.int32)

        # Get the weights from the lookup table
        f_min_weights = self.f_min_table.lookup(score_keys)

        # Score weights are hard-coded to 1 at this stage
        score_weights = self.score_table.lookup(score_keys)

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


class CustomScaledLoss(keras.losses.Loss):
    """
    """

    def __init__(
        self,
        score_loss_fn: Callable,
        f_min_loss_fn: Callable,
        scores: np.ndarray,
        f_min_weights: np.ndarray,
        score_loss_max: float,
        f_min_loss_max: float,
        **kwargs,
    ):
        """
        Parameters
        ----------
        score_loss_fn: Callable
        f_min_loss_fn: Callable
            The function to use for computing the score/f_min loss
            Must take the arguments y_pred, y_true
        scores: numpy array of floats
            The possible score values
        f_min_weights: numpy array of floats
            The f_min weighting corresponding to the
            scores parameter
        score_loss_max: float
        f_min_loss_max: float
            The values to use for min/max scaling of the different losses
            The given will corresponds to a scaled loss of 1.0
        """
        super().__init__(**kwargs)
        self.score_loss_fn = score_loss_fn
        self.f_min_loss_fn = f_min_loss_fn

        self.scores = scores
        self.f_min_weights = f_min_weights

        self.score_loss_max = score_loss_max
        self.f_min_loss_max = f_min_loss_max

        # Setup of f_min weights lookup
        # Keys can't be floats, therefore convert to integer
        scores = tf.constant((scores * 4), tf.int32)
        f_min_weights = tf.constant(f_min_weights, tf.float32)
        self.f_min_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(scores, f_min_weights),
            default_value=tf.constant(np.nan),
        )

    def call(self, y_true, y_pred):
        # Split into score & f_min tensors
        scores_true = tf.gather(y_true, [0, 2, 4], axis=1)
        scores_pred = tf.gather(y_pred, [0, 2, 4], axis=1)

        f_min_true = tf.gather(y_true, [1, 3, 5], axis=1)
        f_min_pred = tf.gather(y_pred, [1, 3, 5], axis=1)

        # Compute the score loss
        score_loss = self.score_loss_fn(scores_true, scores_pred)

        # Min/Max scale the score loss
        score_loss = pre.tf_min_max_scale(
            score_loss, 0.0, 1.0, 0.0, self.score_loss_max
        )

        # Compute the f_min loss
        f_min_loss = self.f_min_loss_fn(f_min_true, f_min_pred)

        # Min/Max scale
        f_min_loss = pre.tf_min_max_scale(
            f_min_loss, 0.0, 1.0, 0.0, self.f_min_loss_max
        )

        # Apply f_min weighting based on true scores
        score_keys = tf.cast(scores_true * 4, tf.int32)
        f_min_weights = self.f_min_table.lookup(score_keys)
        f_min_loss = f_min_loss * f_min_weights

        loss = tf.stack(
            (
                score_loss[:, 0],
                f_min_loss[:, 0],
                score_loss[:, 1],
                f_min_loss[:, 1],
                score_loss[:, 2],
                f_min_loss[:, 2],
            ),
            axis=1,
        )
        return loss


def create_soft_clipping(alpha, z_min: float = 0.0, z_max: float = 1.0):
    """Returns the soft-clipping activation function as used in
    this paper https://arxiv.org/pdf/1810.11509.pdf

    The function is approx. linear within (0, 1) and asymptotes very
    quickly outside that range. To allow other ranges, this implementation
    supports min-max scaling of the input & output

    Mainly useful as a output layer activation function,
    when wanting constrained regression (i.e. linear activation within a range)

    Parameters
    ----------
    alpha: float
        Parameter of the soft-clipping function that determines
        how close to linear the central region is and how sharply the linear
        region turns to the asymptotic values
    z_min: float
    z_max: float
        The range of interest of the output values

    Returns
    -------
    tensorflow function
    """
    alpha = (
        alpha if isinstance(alpha, tf.Tensor) else tf.constant(alpha, dtype=tf.float32)
    )

    def soft_clipping(z):
        z_scaled = pre.tf_min_max_scale(
            z, target_min=0, target_max=1, x_min=0, x_max=10
        )
        result = (1 / alpha) * tf.math.log(
            (1 + tf.math.exp(alpha * z_scaled))
            / (1 + tf.math.exp(alpha * (z_scaled - 1)))
        )
        result = pre.tf_min_max_scale(
            result, target_min=z_min, target_max=z_max, x_min=0, x_max=1
        )
        return result

    return tf.function(soft_clipping)
