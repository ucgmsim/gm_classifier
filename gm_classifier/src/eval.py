from typing import Union, Sequence

import wandb
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import recall_score, precision_score


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


def run_binary_output_eval(
    y_est: Sequence[np.ndarray],
    y_true: np.ndarray,
    loss_weight: float,
    out_name: str,
    prefix: str,
    wandb_save: bool = True
):
    loss_values = []
    recall_values, precision_values = [], []
    for cur_pred_prob in y_est:
        cur_loss = (
            tf.keras.losses.binary_crossentropy(y_true, cur_pred_prob) * loss_weight
        )
        loss_values.append(cur_loss)

        cur_y_pred = cur_pred_prob > 0.5

        recall_values.append(recall_score(y_true, cur_y_pred))
        precision_values.append(precision_score(y_true, cur_y_pred))

    loss_mean, loss_std = np.mean(loss_values), np.std(loss_values)
    recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
    precision_mean, precision_std = np.mean(precision_values), np.std(precision_values)

    print(
        f"{out_name} loss: {loss_mean:.4f} +/- {loss_std:.4f},\n"
        f"{out_name} recall: {recall_mean:.4f} +/- {recall_std:.4f}\n"
        f"{out_name} precision: {precision_mean:.4f} +/- {precision_std:.4f}"
    )

    if wandb_save:
        wandb.run.summary[f"final_{prefix}_{out_name}_loss_mean"] = loss_mean
        wandb.run.summary[f"final_{prefix}_{out_name}_loss_std"] = loss_std

        wandb.run.summary[f"final_{prefix}_{out_name}_recall_mean"] = recall_mean
        wandb.run.summary[f"final_{prefix}_{out_name}_recall_std"] = recall_std

        wandb.run.summary[f"final_{prefix}_{out_name}_precision_mean"] = precision_mean
        wandb.run.summary[f"final_{prefix}_{out_name}_precision_std"] = precision_std


def print_combined_model_eval(
    model: keras.Model,
    X_scalar: np.ndarray,
    X_snr: np.ndarray,
    y_score: np.ndarray,
    y_fmin: np.ndarray,
    score_loss_fn: tf.function,
    fmin_loss_fn: tf.function,
    score_loss_weight: float = 1.0,
    fmin_loss_weight: float = 0.01,
    y_malf: np.ndarray = None,
    malf_loss_weight: float = 0.1,
    y_multi: np.ndarray = None,
    multi_loss_weight: float = 0.1,
    n_preds: int = 25,
    prefix: str = "train",
    wandb_save: bool = True,
):
    y_est = [model.predict({"scalar": X_scalar, "snr": X_snr}) for ix in range(n_preds)]

    score_loss_values, fmin_loss_values = [], []
    for ix in range(len(y_est)):
        cur_score_pred, cur_fmin_pred = y_est[ix][0], y_est[ix][1]
        cur_score_loss = (
            score_loss_fn(y_score, cur_score_pred.ravel()).numpy() * score_loss_weight
        )
        score_loss_values.append(cur_score_loss)

        cur_fmin_loss = (
            fmin_loss_fn(np.stack((y_score, y_fmin), axis=1), cur_fmin_pred).numpy()
            * fmin_loss_weight
        )
        fmin_loss_values.append(cur_fmin_loss)

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

    if y_multi is not None:
        run_binary_output_eval([cur_y_est[2].ravel() for cur_y_est in y_est], y_multi, multi_loss_weight, "Multi", prefix=prefix, wandb_save=wandb_save)

    if y_malf is not None:
        run_binary_output_eval([cur_y_est[3].ravel() for cur_y_est in y_est], y_malf, malf_loss_weight, "Malf", prefix=prefix, wandb_save=wandb_save)

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
    multi_output: bool = False,
    malf_output: bool = False,
):
    score_preds, fmin_preds = [], []
    multi_preds, malf_preds = [], []
    for ix in range(n_preds):
        cur_pred = model.predict({"scalar": X_scalar, "snr": X_snr})

        score_preds.append(cur_pred[0].ravel())
        fmin_preds.append(cur_pred[1][:, 1])

        if multi_output:
            multi_preds.append(cur_pred[2].ravel())
        if malf_output:
            malf_preds.append(cur_pred[3].ravel())

    score_preds = np.stack(score_preds, axis=1)
    fmin_preds = np.stack(fmin_preds, axis=1)

    results = [pd.Series(data=score_preds.mean(axis=1), index=index),
        pd.Series(data=score_preds.std(axis=1), index=index),
        pd.Series(data=fmin_preds.mean(axis=1), index=index),
        pd.Series(data=fmin_preds.std(axis=1), index=index),]

    if multi_output:
        multi_preds = np.stack(multi_preds, axis=1)

        results.append(pd.Series(data=multi_preds.mean(axis=1), index=index))
        results.append(pd.Series(data=multi_preds.std(axis=1), index=index))

    if malf_output:
        malf_preds = np.stack(malf_preds, axis=1)

        results.append(pd.Series(data=malf_preds.mean(axis=1), index=index))
        results.append(pd.Series(data=malf_preds.std(axis=1), index=index))

    return results
