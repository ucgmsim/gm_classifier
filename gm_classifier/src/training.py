import shutil
import fnmatch
from pathlib import Path
from typing import Dict, Tuple, Union, Callable, Any, Iterable, Type, Sequence

import wandb
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from scipy import stats

from . import pre_processing as pre


class ClassAccuracy(keras.metrics.Metric):
    def __init__(self, class_range: Tuple[float, float], **kwargs):
        self.class_min, self.class_max = class_range[0], class_range[1]



        super().__init__(**kwargs)

class MetricRecorder(keras.callbacks.Callback):
    def __init__(
        self,
        training_data: Tuple,
        val_data: Tuple,
        loss_fn: tf.function,
        log_wandb: bool = False,
    ):
        super().__init__()

        self.X_train, self.y_train, _ = training_data
        self.X_val, self.y_val, _ = val_data

        self.loss_fn = loss_fn

        self.log_wandb = log_wandb

    def on_epoch_end(self, epoch, logs=None):
        y_train_est = self.model.predict(self.X_train)
        y_val_est = self.model.predict(self.X_val)

        score_train_mse = tf.losses.mse(self.y_train[:, 0], y_train_est[:, 0]).numpy()
        f_min_train_mse = tf.losses.mse(self.y_train[:, 1], y_train_est[:, 1]).numpy()

        score_val_mse = tf.losses.mse(self.y_val[:, 0], y_val_est[:, 0]).numpy()
        f_min_val_mse = tf.losses.mse(self.y_val[:, 1], y_val_est[:, 1]).numpy()

        logs["score_train_mse"], logs["f_min_train_mse"] = (
            score_train_mse,
            f_min_train_mse,
        )
        logs["score_val_mse"], logs["f_min_val_mse"] = score_val_mse, f_min_val_mse

        if epoch % 25 == 0:
            mc_loss = [
                self.loss_fn(self.y_train, self.model.predict(self.X_train)).numpy()
                for ix in range(10)
            ]
            mc_val_loss = [
                self.loss_fn(self.y_val, self.model.predict(self.X_val)).numpy()
                for ix in range(10)
            ]

            mc_score_train_mse = [
                tf.losses.mse(
                    self.y_train[:, 0], self.model.predict(self.X_train)[:, 0]
                ).numpy()
                for ix in range(10)
            ]
            mc_score_val_mse = [
                tf.losses.mse(
                    self.y_val[:, 0], self.model.predict(self.X_val)[:, 0]
                ).numpy()
                for ix in range(10)
            ]

            mc_score_train_mse, mc_score_train_mse_std = (
                np.mean(mc_score_train_mse),
                np.std(mc_score_train_mse),
            )
            mc_score_val_mse, mc_score_val_mse_std = (
                np.mean(mc_score_val_mse),
                np.std(mc_score_val_mse),
            )

            mc_loss, mc_loss_std = np.mean(mc_loss), np.std(mc_loss)
            mc_val_loss, mc_val_loss_std = np.mean(mc_val_loss), np.std(mc_val_loss)

            print(
                f"MC Loss: {mc_loss:.4f} +/- {mc_loss_std:.4f}, "
                f"MC Val Loss: {mc_val_loss:.4f} +/- {mc_val_loss_std:.4f}"
            )
            print(
                f"MC Score MSE - Training: {mc_score_train_mse:.4f} +/- {mc_score_train_mse_std:.4f}, "
                f"Validation: {mc_score_val_mse:.4f} +/- {mc_score_val_mse_std:.4f} "
            )

        if self.log_wandb:
            wandb.log(
                {
                    f"score_train_mse": score_train_mse,
                    f"f_min_train_mse": f_min_train_mse,
                    f"score_val_mse": score_val_mse,
                    f"f_min_val_mse": f_min_val_mse,
                    f"loss": logs["loss"],
                    f"val_loss": logs["val_loss"],
                    "epoch": epoch,
                }
            )

        print(
            f"Score - Train: {score_train_mse:.4f}, Val: {score_val_mse:.4f} -- "
            f"F_min - Train: {f_min_train_mse:.4f}, Val: {f_min_val_mse:.4f}"
        )

        return


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


def fit(
    output_dir: Path,
    model: Union[Type, keras.Model],
    training_data: Tuple[
        Union[np.ndarray, Dict[str, np.ndarray]], np.ndarray, np.ndarray
    ],
    val_data: Union[
        None, Tuple[Union[np.ndarray, Dict[str, np.ndarray]], np.ndarray, np.ndarray]
    ] = None,
    compile_kwargs: Dict[str, Any] = None,
    fit_kwargs: Dict[str, Any] = None,
    callbacks: Sequence[keras.callbacks.Callback] = None,
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
    training_data: triplet
        Training data, expected tuple data:
        (X_train, y_train, ids_train)
    model_config: dictionary with string keys, optional
        Dictionary that contains model details
        If not specified, then the model parameter has to be a
        keras model instance
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

    tensorboard_output_dir = output_dir / "tensorboard_log"
    if tensorboard_output_dir.is_dir():
        shutil.rmtree(tensorboard_output_dir)

    # Train the model
    model.compile(**compile_kwargs)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        **fit_kwargs,
    )

    # Save the model
    model.save(output_dir / "best_model", save_format="tf")

    # Save the history
    hist_df = pd.DataFrame.from_dict(history.history, orient="columns")
    hist_df.to_csv(output_dir / "history.csv", index_label="epoch")

    model_plot_ffp = output_dir / "model.png"
    keras.utils.plot_model(
        model,
        model_plot_ffp,
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True,
    )
    wandb.log({"model": model_plot_ffp})

    return history.history, model


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


@tf.function
def squared_error(y_true, y_pred):
    return tf.square(y_true - y_pred)


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
        scores_true = tf.cast(tf.gather(y_true, [0], axis=1), tf.float32)
        scores_pred = tf.gather(y_pred, [0], axis=1)

        f_min_true = tf.cast(tf.gather(y_true, [1], axis=1), tf.float32)
        f_min_pred = tf.gather(y_pred, [1], axis=1)

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

        loss = tf.stack((score_loss[:, 0], f_min_loss[:, 0],), axis=1,)
        return loss


def create_soft_clipping(
    p, z_min: float = 0.0, z_max: float = 1.0, x_min: float = 0, x_max: float = 10
):
    """Returns the scaled soft-clipping function

    Parameters
    ----------
    p: float
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
    p = p if isinstance(p, tf.Tensor) else tf.constant(p, dtype=tf.float32)

    def scaled_soft_clipping(z):
        z_scaled = pre.tf_min_max_scale(
            z, target_min=0, target_max=1, x_min=x_min, x_max=x_max
        )
        result = soft_clipping(z_scaled, p)
        result = pre.tf_min_max_scale(
            result, target_min=z_min, target_max=z_max, x_min=0, x_max=1
        )
        return result

    return tf.function(scaled_soft_clipping)


@tf.function
def soft_clipping(z: tf.Tensor, p: tf.Tensor):
    """Implementation of the soft-clipping activation function as used in
    this paper https://arxiv.org/pdf/1810.11509.pdf

    The function is approx. linear within (0, 1) and asymptotes very
    quickly outside that range. To allow other ranges, this implementation
    supports min-max scaling of the input & output

    Mainly useful as a output layer activation function,
    when wanting constrained regression (i.e. linear activation within a range)
    """
    z, p = tf.cast(z, dtype=tf.float64), tf.cast(p, dtype=tf.float64)
    return tf.cast(
        (1 / p)
        * tf.math.log((1 + tf.math.exp(p * z)) / (1 + tf.math.exp(p * (z - 1)))),
        dtype=tf.float32,
    )


def create_scaled_sigmoid(shift: float = 0.0, scale: float = 1.0):
    def scaled_sigmoid(z):
        return (tf.sigmoid(z) + shift) * scale

    return tf.function(scaled_sigmoid)


def create_custom_act_fn(score_act_fn: tf.function, f_min_act_fn: tf.function):
    def custom_act_fn(z):
        score_z = tf.gather(z, [0], axis=1)
        score_res = score_act_fn(score_z)

        f_min_z = tf.gather(z, [1], axis=1)
        f_min_res = f_min_act_fn(f_min_z)

        output = tf.stack((score_res[:, 0], f_min_res[:, 0],), axis=1)

        return output

    return tf.function(custom_act_fn)
