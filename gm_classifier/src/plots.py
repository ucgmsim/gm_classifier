from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def plot_loss(history: Dict, ax: plt.Axes = None, **kwargs):
    """Plots single output loss"""
    if ax is None:
        fig, ax = plt.subplots(**kwargs)

    epochs = np.arange(len(history["loss"]))
    ax.plot(epochs, history["loss"], "k-", label="Training")
    ax.plot(epochs, history["val_loss"], "k--", label="Validation")
    ax.legend()

    return fig, ax


def plot_multi_loss(
    history: Dict, label_names: List[str], output_ffp: str = None, ax: plt.Axes = None, **kwargs
):
    colours = ["r", "b", "g", "m"]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    epochs = np.arange(len(history["loss"]))

    for cur_label_name, c in zip(label_names, colours):
        ax.plot(epochs, history[f"{cur_label_name}_loss"], f"{c}-", label=f"{cur_label_name} training")
        ax.plot(epochs, history[f"val_{cur_label_name}_loss"], f"{c}--", label=f"{cur_label_name} validation")

    ax.plot(epochs, history["loss"], "k-", label="Total training")
    ax.plot(epochs, history["val_loss"], "k--", label="Total validation")
    ax.legend()

    if output_ffp is not None:
        plt.savefig(output_ffp)
        plt.close()

    return fig, ax
