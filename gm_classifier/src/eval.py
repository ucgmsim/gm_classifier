from typing import Union, Sequence

import wandb
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import recall_score, precision_score

from gm_classifier.src.console import console

class ClassAcc(keras.metrics.Metric):

    def __init__(self, true_value: float, **kwargs):
        super().__init__(name=f"class_acc_{str(true_value).replace('.', 'p')}", **kwargs)

        self.true_value = true_value

        self.tp = self.add_weight("tp", initializer="zeros", dtype=tf.int64)
        self.fn = self.add_weight("fn", initializer="zeros", dtype=tf.int64)

    def update_state(self, y_true, y_pred, **kwargs):
        est_mask = tf.logical_and(y_pred < self.true_value + 0.125, y_pred > self.true_value - 0.125)
        true_mask = tf.convert_to_tensor(tf.experimental.numpy.isclose(y_true, self.true_value))

        self.tp.assign_add(tf.math.count_nonzero(tf.logical_and(true_mask, est_mask)))
        self.fn.assign_add(tf.math.count_nonzero(tf.logical_and(true_mask, tf.logical_not(est_mask))))

    def result(self):
        return self.tp / (self.tp + self.fn)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "true_value": self.true_value}


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
    out_name: str,
    prefix: str,
    wandb_save: bool = True
):
    loss_values = []
    recall_values, precision_values = [], []
    for cur_pred_prob in y_est:
        cur_loss = (
            tf.keras.losses.binary_crossentropy(y_true, cur_pred_prob)
        )
        loss_values.append(cur_loss)

        cur_y_pred = cur_pred_prob > 0.5

        recall_values.append(recall_score(y_true, cur_y_pred))
        precision_values.append(precision_score(y_true, cur_y_pred))

    loss_mean, loss_std = np.mean(loss_values), np.std(loss_values)
    recall_mean, recall_std = np.mean(recall_values), np.std(recall_values)
    precision_mean, precision_std = np.mean(precision_values), np.std(precision_values)

    outputs = [
        f"\tRecall: {recall_mean:.4f} +/- {recall_std:.4f}",
        f"\tPrecision: {precision_mean:.4f} +/- {precision_std:.4f}",
    ]

    if wandb_save:
        # wandb.run.summary[f"final_{prefix}_{out_name}_loss_mean"] = loss_mean
        # wandb.run.summary[f"final_{prefix}_{out_name}_loss_std"] = loss_std

        wandb.run.summary[f"final_{prefix}_{out_name}_recall_mean"] = recall_mean
        wandb.run.summary[f"final_{prefix}_{out_name}_recall_std"] = recall_std

        wandb.run.summary[f"final_{prefix}_{out_name}_precision_mean"] = precision_mean
        wandb.run.summary[f"final_{prefix}_{out_name}_precision_std"] = precision_std

    return np.asarray(loss_values), outputs


def print_combined_model_eval(
    model: keras.Model,
    X_scalar: np.ndarray,
    X_snr: np.ndarray,
    y_score: np.ndarray,
    y_fmin: np.ndarray,
    score_loss_fn: tf.function,
    fmin_loss_fn: tf.function,
    fmin_metric_fn: tf.function,
    score_loss_weight: float = 1.0,
    fmin_loss_weight: float = 0.01,
    y_multi: np.ndarray = None,
    multi_loss_weight: float = 0.1,
    n_preds: int = 25,
    prefix: str = "train",
    wandb_save: bool = True,
):
    true_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    y_est = [model.predict({"scalar": X_scalar, "snr": X_snr}) for ix in range(n_preds)]

    score_loss_values, fmin_loss_values = [], []
    fmin_metric_values = []
    score_metric_values = []
    for ix in range(len(y_est)):
        cur_score_pred, cur_fmin_pred = y_est[ix][0], y_est[ix][1]
        cur_score_loss = score_loss_fn(y_score, cur_score_pred.ravel()).numpy()
        score_loss_values.append(cur_score_loss)

        cur_score_metric_values = []
        for cur_true_val in true_values:
            cur_class_acc = ClassAcc(cur_true_val)
            cur_class_acc.update_state(y_score, cur_score_pred.ravel())
            cur_result = cur_class_acc.result()

            cur_score_metric_values.append(cur_result)
        score_metric_values.append(cur_score_metric_values)

        cur_fmin_loss = fmin_loss_fn(np.stack((y_score, y_fmin), axis=1), cur_fmin_pred).numpy()
        fmin_loss_values.append(cur_fmin_loss)

        cur_fmin_metric = fmin_metric_fn(np.stack((y_score, y_fmin), axis=1), cur_fmin_pred).numpy()
        fmin_metric_values.append(cur_fmin_metric)

    score_loss_values, fmin_loss_values = (
        np.asarray(score_loss_values),
        np.asarray(fmin_loss_values),
    )
    total_loss_values = score_loss_values * score_loss_weight + fmin_loss_values * fmin_loss_weight

    score_metric_mean = np.mean(np.asarray(score_metric_values), axis=0)
    score_metric_std = np.std(np.asarray(score_metric_values), axis=0)

    fmin_metric_mean = np.mean(fmin_metric_values)
    fmin_metric_std = np.std(fmin_metric_values)

    if y_multi is not None:
        multi_loss_values, multi_outputs = run_binary_output_eval([cur_y_est[2].ravel() for cur_y_est in y_est], y_multi, "Multi", prefix=prefix, wandb_save=wandb_save)
        total_loss_values = total_loss_values + multi_loss_values * multi_loss_weight

    total_loss_mean, total_loss_std = (
        np.mean(total_loss_values),
        np.std(total_loss_values),
    )

    score_loss_mean, score_loss_std = np.mean(score_loss_values), np.std(score_loss_values)

    console.print(
        f"\n[bold]{prefix} - Total (weighted) Loss:[/] \n\t{total_loss_mean:.4f} +/- {total_loss_std:.4f}")

    console.print(f"\n[bold]{prefix} - Score Class Acc:[/]")
    for ix, cur_true_val in enumerate(true_values):
        console.print(f"\t{cur_true_val}: {score_metric_mean[ix]:.4f} +/- {score_metric_std[ix]:.4f}")

    console.print(f"\n[bold]{prefix} - Fmin MSE[/]:\n\t{fmin_metric_mean:.4f} +/- {fmin_metric_std:.4f}")

    if y_multi is not None:
        console.print(f"\n[bold]{prefix} - Multi[/]")
        for output in multi_outputs:
            console.print(output)

    if wandb_save:
        wandb.run.summary[f"final_{prefix}_total_loss_mean"] = total_loss_mean
        wandb.run.summary[f"final_{prefix}_total_loss_std"] = total_loss_std

        wandb.run.summary[f"final_{prefix}_score_loss_mean"] = score_loss_mean
        wandb.run.summary[f"final_{prefix}_score_loss_std"] = score_loss_std

        for cur_true_val, cur_mean, cur_std in zip(true_values, score_metric_mean, score_metric_std):
            wandb.run.summary[f"final_{prefix}_score_mean_class_acc_{cur_true_val}"] = cur_mean
            wandb.run.summary[f"final_{prefix}_score_std_class_acc_{cur_true_val}"] = cur_std

        wandb.run.summary[f"final_{prefix}_fmin_metric_mean"] = fmin_metric_mean
        wandb.run.summary[f"final_{prefix}_fmin_metric_std"] = fmin_metric_std

    return


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

    results = [pd.Series(data=score_preds.mean(axis=1), index=index, name="score_mean"),
        pd.Series(data=score_preds.std(axis=1), index=index, name="score_std"),
        pd.Series(data=fmin_preds.mean(axis=1), index=index, name="fmin_mean"),
        pd.Series(data=fmin_preds.std(axis=1), index=index, name="fmin_std"),]

    if multi_output:
        multi_preds = np.stack(multi_preds, axis=1)

        results.append(pd.Series(data=multi_preds.mean(axis=1), index=index, name="multi_mean"))
        results.append(pd.Series(data=multi_preds.std(axis=1), index=index, name="multi_std"))

    if malf_output:
        malf_preds = np.stack(malf_preds, axis=1)

        results.append(pd.Series(data=malf_preds.mean(axis=1), index=index))
        results.append(pd.Series(data=malf_preds.std(axis=1), index=index))

    return results
