from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def plot_loss(
    history: Dict, ax: plt.Axes = None, output_ffp: str = None, fig_kwargs: Dict = None
):
    """Plots single output loss"""
    if ax is None:
        fig, ax = plt.subplots(**fig_kwargs)

    epochs = np.arange(len(history["loss"]))
    ax.plot(epochs, history["loss"], "k-", label=f"Training - {np.min(history['loss'])}")
    ax.plot(
        epochs,
        history["val_loss"],
        "k--",
        label=f"Validation - {np.min(history['val_loss'])}",
    )
    ax.legend()

    if output_ffp is not None:
        plt.savefig(output_ffp)
        plt.close()
        return None, None

    return fig, ax


def plot_multi_loss(
    history: Dict,
    label_names: List[str],
    output_ffp: str = None,
    ax: plt.Axes = None,
    fig_kwargs: Dict = {},
):
    """Plots the loss for multi-output models"""
    colours = ["r", "b", "g", "m"]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(**fig_kwargs)
    epochs = np.arange(len(history["loss"]))

    for cur_label_name, c in zip(label_names, colours):
        cur_train_key = f"{cur_label_name}_loss"
        ax.plot(
            epochs,
            history[cur_train_key],
            f"{c}-",
            label=f"{cur_label_name} training - {np.min(history[cur_train_key]):.4f}",
        )
        cur_val_key = f"val_{cur_label_name}_loss"
        ax.plot(
            epochs,
            history[cur_val_key],
            f"{c}--",
            label=f"{cur_label_name} validation - {np.min(history[cur_val_key]):.4f}",
        )

    ax.plot(
        epochs,
        history["loss"],
        "k-",
        label=f"Total training - {np.min(history['loss']):.4f}",
    )
    ax.plot(
        epochs,
        history["val_loss"],
        "k--",
        label=f"Total validation - {np.min(history['val_loss']):.4f}",
    )
    ax.legend()

    if output_ffp is not None:
        plt.savefig(output_ffp)
        plt.close()
        return None, None

    return fig, ax


def plot_true_vs_est(
    y_est: np.ndarray,
    y_true: np.ndarray,
    y_val_est: np.ndarray = None,
    y_val_true: np.ndarray = None,
    output_ffp: str = None,
    ax: plt.Axes = None,
    min_max: Tuple[float, float] = None,
    title: str = None,
    fig_kwargs: Dict = {},
    scatter_kwargs: Dict = {},
):
    """Plots true (y-axis) vs estimated (x-axis)"""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(**fig_kwargs)

    label = None if y_val_est is None else "training"
    ax.scatter(y_est, y_true, c="b", label=label, **scatter_kwargs)
    if y_val_est is not None and y_val_true is not None:
        ax.scatter(y_val_est, y_val_true, c="r", label="validation", **scatter_kwargs)
        ax.legend()

    if min_max is not None:
        ax.plot([min_max[0], min_max[1]], [min_max[0], min_max[1]], "k--")

    ax.set_xlabel("Estimated")
    ax.set_ylabel("True")

    if title:
        ax.set_title(title)

    if output_ffp is not None:
        plt.savefig(output_ffp)
        plt.close()
        return None, None

    return fig, ax


def plot_residual(
    res: np.ndarray,
    x: np.ndarray,
    res_val: np.ndarray = None,
    x_val: np.ndarray = None,
    output_ffp: str = None,
    ax: plt.Axes = None,
    min_max: Tuple[float, float] = None,
    title: str = None,
    x_label: str = None,
    fig_kwargs: Dict = {},
    scatter_kwargs: Dict = {},
):
    # Ensure that x_val is set if val_res is given
    assert (
        res_val is None or x_val is not None
    ), "If res_val is specified, then x_val has to be set as well"

    fig = None
    if ax is None:
        fig, ax = plt.subplots(**fig_kwargs)

    sort_ind = np.argsort(x)

    ax.scatter(x[sort_ind], res[sort_ind], c="b", **scatter_kwargs)
    if res_val is not None:
        sort_ind_val = np.argsort(x_val)
        ax.scatter(x_val[sort_ind_val], res_val[sort_ind_val], c="r", **scatter_kwargs)

    if min_max is not None:
        ax.plot([min_max[0], min_max[1]], [0, 0])

    ax.set_ylabel("True - Estimated")

    if x_label is not None:
        ax.set_xlabel(x_label)

    if title is not None:
        ax.set_title(title)

    if output_ffp is not None:
        plt.savefig(output_ffp)
        plt.close()
        return None, None

    return fig, ax
