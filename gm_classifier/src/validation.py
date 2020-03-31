from pathlib import Path
from typing import Dict, Tuple, Callable

import pandas as pd
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.model_selection import KFold

from . import training
from . import features
from . import pre_processing as pre


def k_fold(
    output_dir: str,
    features_df: pd.DataFrame,
    label_df: pd.DataFrame,
    config: Dict,
    eval_fn: Callable,
    n_splits: int = 10,
    score_th: Tuple[float, float] = (0.01, 0.99),
    record_weight_fn: Callable[
        [pd.DataFrame, np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ] = None,
    verbose: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Runs k-fold validation for the given labelled data, evaluation is
    done using the provided evaluation function"""
    output_dir = Path(output_dir)

    train_df = pd.merge(features_df, label_df, left_index=True, right_index=True)
    train_df.to_csv(output_dir / "training_data.csv")

    # Some sanity checks
    assert np.all(np.isin(features.FEATURE_NAMES, features_df.columns.values))

    # Get training data
    y, mask = pre.get_label_from_score(
        train_df.loc[:, "score"].values, low_th=score_th[0], high_th=score_th[1]
    )
    X = train_df.loc[mask, features.FEATURE_NAMES].values
    ids = train_df.index.values[mask].astype(str)

    # Run the k-iterations
    result_dict, loss_dict = {}, {}
    kf = KFold(n_splits=n_splits, shuffle=True)
    for ix, (train_ind, val_ind) in enumerate(kf.split(X)):
        if verbose > 0:
            print(f"=============== Running iteration {ix} ===============")
        cur_id = f"iter_{ix}"

        # Create the output directory
        cur_output_dir = output_dir / cur_id
        cur_output_dir.mkdir()

        # Get the current training and validation data
        X_train, y_train, ids_train = X[train_ind], y[train_ind], ids[train_ind]
        X_val, y_val, ids_val = X[val_ind], y[val_ind], ids[val_ind]

        sample_weights = (
            None
            if record_weight_fn is None
            else record_weight_fn(train_df, X_train, y_train, ids_train)
        )

        # Run the training
        history, X_train, X_val = training.train(
            cur_output_dir,
            config,
            (X_train, y_train, ids_train),
            val_data=(X_val, y_val, ids_val),
            verbose=verbose,
            sample_weights=sample_weights
        )
        loss_dict[cur_id] = {"train": history["loss"], "val": history["val_loss"]}

        # Load the resulting model
        model = keras.models.load_model(cur_output_dir / "model.h5")

        # Call the evaluation function
        result_dict[cur_id] = eval_fn(model, X_train, y_train, X_val, y_val)

        if verbose > 0:
            print(f"======================================================\n")

    result_df = pd.DataFrame(result_dict).T
    return train_df, result_df, loss_dict


def eval(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
):
    y_train_est = label_from_prob(model.predict(X_train).reshape(-1))
    y_val_est = label_from_prob(model.predict(X_val).reshape(-1))

    tn_train, fp_train, fn_train, tp_train = metrics.confusion_matrix(
        y_train, y_train_est
    ).ravel()
    tn_val, fp_val, fn_val, tp_val = metrics.confusion_matrix(y_val, y_val_est).ravel()

    return {
        "prec_train": metrics.precision_score(y_train, y_train_est),
        "rec_train": metrics.recall_score(y_train, y_train_est),
        "f1_train": metrics.f1_score(y_train, y_train_est),
        "acc_train": metrics.accuracy_score(y_train, y_train_est),
        "tn_train": tn_train,
        "fp_train": fp_train,
        "fn_train": fn_train,
        "tp_train": tp_train,
        "prec_val": metrics.precision_score(y_val, y_val_est),
        "rec_val": metrics.recall_score(y_val_est, y_val_est),
        "f1_val": metrics.f1_score(y_val, y_val_est),
        "acc_val": metrics.accuracy_score(y_val, y_val_est),
        "tn_val": tn_val,
        "fp_val": fp_val,
        "fn_val": fn_val,
        "tp_val": tp_val,
    }


def label_from_prob(prob: np.ndarray, threshold: float = 0.5):
    """Gets the low-high quality label from NN result"""
    y = np.zeros(prob.shape[0], dtype=int)
    y[prob > threshold] = 1

    return y


def plot_multi_loss(loss_dict: Dict, ax: plt.Axes = None, colour_char: str = "b"):
    """Plots multiple losses (and their average) on a single plot"""
    if ax is None:
        fig, ax = plt.subplots()

    epochs = np.arange(len(loss_dict["iter_0"]["train"]))

    train_sum, val_sum = None, None
    for cur_values in loss_dict.values():
        pass
        ax.plot(epochs, cur_values["train"], f"{colour_char}-", lw=1)
        ax.plot(epochs, cur_values["val"], f"{colour_char}--", lw=1)

        train_sum = (
            np.asarray(cur_values["train"])
            if train_sum is None
            else train_sum + np.asarray(cur_values["train"])
        )
        val_sum = (
            np.asarray(cur_values["val"])
            if train_sum is None
            else train_sum + np.asarray(cur_values["val"])
        )

    ax.plot(epochs, train_sum / len(loss_dict), "k-", lw=2, label="Training")
    ax.plot(epochs, val_sum / len(loss_dict), "k--", lw=2, label="Validation")

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()


def compute_multi_mean_min_loss(loss_dict: Dict):
    """Computes the mean minimum loss across multiple trainings"""
    min_train_loss_values = np.asarray([np.min(cur_values["train"]) for cur_values in loss_dict.values()])
    min_val_losss_values = np.asarray([np.min(cur_values["val"]) for cur_values in loss_dict.values()])

    return min_train_loss_values.mean(), min_val_losss_values.mean()

