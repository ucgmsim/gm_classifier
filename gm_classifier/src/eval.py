from typing import Union

import wandb
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras


def get_mc_single_predictions(
    model: keras.Model,
    X: Union[pd.DataFrame, np.ndarray],
    n_preds: int = 10,
    index: np.ndarray = None,
):
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
        loss = [
            loss_fn(y[:, None], cur_y_est[:, None], sample_weight=sample_weights)
            for cur_y_est in y_est
        ]
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
    wandb_save: bool = True,
):
    mean_loss, std_loss = compute_avg_single_loss(
        model, X, y, loss_fn, sample_weights=sample_weights, n_preds=n_preds
    )

    print(f"Model {prefix} loss: {mean_loss:.4f} +/- {std_loss:.4f}")

    if wandb_save:
        wandb.run.summary[f"final_avg_{prefix}_loss"] = mean_loss


def print_combined_model_eval(
    model: keras.Model,
    X_scalar: np.ndarray,
    X_snr: np.ndarray,
    y_score: np.ndarray,
    y_fmin: np.ndarray,
    score_loss_fn: tf.function,
    fmin_loss_fn: tf.function,
    n_preds: int = 25,
    prefix: str = "train",
    wandb_save: bool = True,
):
    y_est = [model.predict({"scalar": X_scalar, "snr": X_snr}) for ix in range(n_preds)]

    score_loss_values, fmin_loss_values = [], []
    for cur_score_pred, cur_fmin_pred in y_est:
        score_loss_values.append(score_loss_fn(y_score, cur_score_pred.ravel()).numpy())

        fmin_loss_values.append(
            fmin_loss_fn(np.stack((y_score, y_fmin), axis=1), cur_fmin_pred)
        )

    score_loss_values, fmin_loss_values = (
        np.asarray(score_loss_values),
        np.asarray(fmin_loss_values),
    )
    total_loss_values = score_loss_values + fmin_loss_values

    score_loss_mean, score_loss_std = (
        np.mean(score_loss_values),
        np.std(score_loss_values),
    )
    fmin_loss_mean, fmin_loss_std = np.mean(fmin_loss_values), np.std(fmin_loss_values)
    total_loss_mean, total_loss_std = (
        np.mean(total_loss_values),
        np.std(total_loss_values),
    )

    print(
        f"\nModel {prefix}, Total loss: {total_loss_mean:.4f} +/- {total_loss_std:.4f},\n"
        f"Score loss: {score_loss_mean:.4f} +/- {score_loss_std:.4f},\n"
        f"Fmin loss: {fmin_loss_mean:.4f} +/- {fmin_loss_std:.4f}"
    )

    if wandb_save:
        wandb.run.summary[f"final_{prefix}_total_loss_mean"] = total_loss_mean
        wandb.run.summary[f"final_{prefix}_total_loss_std"] = total_loss_std

        wandb.run.summary[f"final_{prefix}_score_loss_mean"] = score_loss_mean
        wandb.run.summary[f"final_{prefix}_score_loss_std"] = score_loss_std

        wandb.run.summary[f"final_{prefix}_fmin_loss_mean"] = fmin_loss_mean
        wandb.run.summary[f"final_{prefix}_fmin_loss_std"] = fmin_loss_std


def get_combined_prediction(
    model: keras.Model,
    X_scalar: np.ndarray,
    X_snr: np.ndarray,
    n_preds: int,
    index: np.ndarray,
):
    score_preds, fmin_preds = [], []
    for ix in range(n_preds):
        cur_score, cur_fmin = model.predict({"scalar": X_scalar, "snr": X_snr})

        score_preds.append(cur_score.ravel())
        fmin_preds.append(cur_fmin[:, 1])

    score_preds = np.stack(score_preds, axis=1)
    fmin_preds = np.stack(fmin_preds, axis=1)

    return (
        pd.Series(data=score_preds.mean(axis=1), index=index),
        pd.Series(data=score_preds.std(axis=1), index=index),
        pd.Series(data=fmin_preds.mean(axis=1), index=index),
        pd.Series(data=fmin_preds.std(axis=1), index=index),
    )
