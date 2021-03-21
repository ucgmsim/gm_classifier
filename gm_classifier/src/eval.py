from typing import Union

import wandb
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras


def get_mc_single_predictions(model: keras.Model, X: Union[pd.DataFrame, np.ndarray], n_preds: int = 10, index: np.ndarray = None):
    inputs = X.values if isinstance(X, pd.DataFrame) else X
    index = index if index is not None else X.index

    y_est = np.asarray([model.predict(inputs).ravel() for ix in range(n_preds)]).T


    return (
        pd.Series(index=index, data=np.mean(y_est, axis=1)),
        pd.Series(index=index, data=np.std(y_est, axis=1)),
    )


def compute_avg_single_loss(
    model: keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: tf.function,
    sample_weights: np.ndarray = None,
    n_preds: int = 10,
):
    y_est = [model.predict(X).ravel() for ix in range(n_preds)]

    if sample_weights is not None:
        loss = [loss_fn(y[:, None], cur_y_est[:, None], sample_weight=sample_weights) for cur_y_est in y_est]
    else:
        loss = [loss_fn(y, cur_y_est) for cur_y_est in y_est]

    return np.mean(loss), np.std(loss)


def print_single_model_eval(
    model: keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: tf.function,
    sample_weights: np.ndarray,
    n_preds: int = 25,
    prefix: str = "train",
    wandb_save: bool = True
):
    mean_loss, std_loss = compute_avg_single_loss(model, X, y, loss_fn, sample_weights=sample_weights, n_preds=n_preds)

    print(f"Model {prefix} loss: {mean_loss:.4f} +/- {std_loss:.4f}")

    if wandb_save:
        wandb.run.summary[f"final_avg_{prefix}_loss"] = mean_loss
