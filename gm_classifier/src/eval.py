import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras


def get_mc_single_predictions(model: keras.Model, X: pd.DataFrame, n_preds: int = 10):
    y_est = np.asarray([model.predict(X.values).ravel() for ix in range(n_preds)]).T

    return (
        pd.Series(index=X.index, data=np.mean(y_est, axis=1)),
        pd.Series(index=X.index, data=np.std(y_est, axis=1)),
    )


def compute_avg_single_loss(
    model: keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: tf.function,
    n_preds: int = 10,
):
    y_est = [model.predict(X).ravel() for ix in range(n_preds)]

    loss = [loss_fn(y, cur_y_est) for cur_y_est in y_est]

    return np.mean(loss), np.std(loss)


def print_single_model_eval(
    model: keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: tf.function,
    n_preds: int = 25,
    prefix: str = "train",
):
    mean_loss, std_loss = compute_avg_single_loss(model, X, y, loss_fn, n_preds=n_preds)

    print(f"Model {prefix} loss: {mean_loss:.4f} +/- {std_loss:.4f}")
