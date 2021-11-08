import shutil
from pathlib import Path
from typing import Dict, Tuple, Union, Callable, Any, Iterable, Type, Sequence


import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


@tf.function
def squared_error(y_true, y_pred):
    return tf.square(y_true - y_pred)


class FMinLoss(keras.losses.Loss):
    def __init__(
        self,
        fmin_weights_mapping: Dict,
        base_loss_fn: Callable = squared_error,
        **kwargs,
    ):
        """
        Parameters
        ----------
        fmin_weights_mapping: numpy array of floats
            The f_min weighting corresponding to the
            scores parameter
        """
        super().__init__(**kwargs)

        self.base_loss_fn = base_loss_fn
        self.fmin_weights_mapping = fmin_weights_mapping

        # Setup of f_min weights lookup
        # Keys can't be floats, therefore convert to integer
        scores = tf.constant(
            np.asarray(list(self.fmin_weights_mapping.keys())) * 4, tf.int32
        )
        fmin_weights = tf.constant(list(self.fmin_weights_mapping.values()), tf.float32)
        self.fmin_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(scores, fmin_weights),
            default_value=tf.constant(np.nan),
        )

    def call(self, y_true, y_pred):
        sample_loss, sample_weights = self.compute_sample_loss(y_true, y_pred)
        return tf.reduce_sum(sample_loss) / tf.reduce_sum(sample_weights)

    def compute_sample_loss(self, y_true, y_pred):
        # Split into score & f_min tensors
        scores_true, f_min_true = y_true[:, 0], y_true[:, 1]
        f_min_pred = y_pred[:, 1]

        # Compute the f_min loss
        fmin_sample_loss = self.base_loss_fn(f_min_true, f_min_pred)

        # Apply f_min weighting based on true scores
        score_keys = tf.cast(scores_true * 4, tf.int32)
        fmin_weights = self.fmin_table.lookup(score_keys)
        fmin_sample_loss = fmin_sample_loss * fmin_weights

        return fmin_sample_loss, fmin_weights


def fit(
    output_dir: Path,
    model: Union[Type, keras.Model],
    training_data: Tuple,
    val_data: Tuple = None,
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
    import wandb

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
    wandb.save(str(model_plot_ffp))

    return history.history, model


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
