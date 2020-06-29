from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_loss(
    history: Dict, ax: plt.Axes = None, output_ffp: str = None, fig_kwargs: Dict = None
):
    """Plots single output loss"""
    if ax is None:
        fig, ax = plt.subplots(**fig_kwargs)

    epochs = np.arange(len(history["loss"]))
    ax.plot(
        epochs, history["loss"], "k-", label=f"Training - {np.min(history['loss'])}"
    )
    if "val_loss" in history.keys():
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
    c_train: Union[str, np.ndarray] = None,
    c_val: Union[str, np.ndarray] = None,
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
    train_scatter = ax.scatter(
        y_est, y_true, label=label, c=c_train, marker=".", **scatter_kwargs
    )
    if y_val_est is not None and y_val_true is not None:
        ax.scatter(
            y_val_est,
            y_val_true,
            c=c_val,
            marker="s",
            label="validation",
            **scatter_kwargs,
        )
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
        return None, None, None

    return fig, ax, train_scatter


def plot_residual(
    res: np.ndarray,
    x: np.ndarray,
    res_val: np.ndarray = None,
    x_val: np.ndarray = None,
    c_train: Union[str, np.ndarray] = None,
    c_val: Union[str, np.ndarray] = None,
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

    c_train = c_train if isinstance(c_train, str) else c_train[sort_ind]
    train_scatter = ax.scatter(
        x[sort_ind], res[sort_ind], c=c_train, marker=".", **scatter_kwargs
    )
    if res_val is not None:
        sort_ind_val = np.argsort(x_val)
        c_val = c_val if isinstance(c_val, str) else c_val[sort_ind_val]
        ax.scatter(
            x_val[sort_ind_val],
            res_val[sort_ind_val],
            c=c_val,
            marker="s",
            **scatter_kwargs,
        )

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
        return None, None, None

    return fig, ax, train_scatter


def get_color_marker_comb():
    colours = mcolors.TABLEAU_COLORS
    markers = ["o", "x", "v", "^", ">", "<", "s", "+", "d", "p"]

    return [(c, m) for c in colours for m in markers]


def create_record_eval_plots(
    record_ids: List[str],
    acc_ts: List[np.ndarray],
    dt: List[float],
    snr: List[np.ndarray],
    ft_freq: List[np.ndarray],
    result_df: pd.DataFrame,
    output_dir: Union[Path, str],
    label_df: pd.DataFrame = None,
    p_wave_ind: List[int] = None,
):
    """Creates record eval plots for each of the specified records,
    each plot shows ACC and SNR for each of the three components

    Parameters
    ----------
    record_ids: list of strings
        The record ids for which to generate plots
        Note: The record_ids, acc_ts, dt, snr and ft_freq lists
        all have to be in the same order!
    acc_ts: list of numpy array
        The acc timeseries
    dt: list of floats
        The timestep size
    snr: list of numpy array
        The SNR frequency series
    ft_freq: list of numpy array
        The SNR frequencies
    result_df: pandas datframe
        Expected columns are [score_X, f_min_X....]
    output_dir: str or path
    label_df: pandas dataframe
        Contains the true labels for each record
        Expected columns are [score_X, f_min_X....]
    p_wave_ind: list of ints
        Contains the p-wave index for each record
        Also has to be in the same order as record_ids
    """
    for rec_ix, cur_id in enumerate(record_ids):
        print(f"Processing {rec_ix + 1}/{len(record_ids)}")
        if cur_id not in record_ids:
            print(f"No timeseries data for record {cur_id}, skipping")
            continue

        # Get the relevant data
        ts_ix = np.flatnonzero(record_ids == cur_id)[0]
        cur_acc = acc_ts[ts_ix]
        cur_snr = snr[ts_ix]
        cur_ft_freq = ft_freq[ts_ix]

        fig, axes = plt.subplots(2, 3, figsize=(22, 10))

        for comp_ix, cur_comp in enumerate(["X", "Y", "Z"]):
            ax_1, ax_2 = axes[0, comp_ix], axes[1, comp_ix]
            cur_f_min_true = label_df.loc[cur_id, f"f_min_{cur_comp}"]
            cur_score_true = label_df.loc[cur_id, f"score_{cur_comp}"]
            cur_f_min_est = result_df.loc[cur_id, f"f_min_{cur_comp}"]
            cur_score_est = result_df.loc[cur_id, f"score_{cur_comp}"]

            t = np.arange(cur_acc.shape[0]) * dt[rec_ix]
            ax_1.plot(t, cur_acc[:, comp_ix], label="X")

            if p_wave_ind is not None:
                ax_1.axvline(x=t[p_wave_ind[rec_ix]], c="k", linestyle="--")

            ax_1.set_ylabel("Acc")
            ax_1.set_xlabel("Time")
            ax_1.set_title(
                f"Score - True: {cur_score_true} Est: {cur_score_est:.2f}, f_min - True: {cur_f_min_true}, Est: {cur_f_min_est:.2f}"
            )
            ax_1.grid()

            ax_2.plot(cur_ft_freq, cur_snr[:, comp_ix], "b")
            ax_2.plot(
                [cur_f_min_true, cur_f_min_true],
                [cur_snr[1:, comp_ix].min(), cur_snr[1:, comp_ix].max()],
                "k--",
                linewidth=1.4,
            )
            ax_2.plot(
                [cur_f_min_est, cur_f_min_est],
                [cur_snr[1:, comp_ix].min(), cur_snr[1:, comp_ix].max()],
                "r--",
                linewidth=1.4,
            )
            ax_2.plot(
                [cur_ft_freq.min(), cur_ft_freq.max()],
                [2.0, 2.0],
                color="gray",
                linestyle="--",
                linewidth=1.0,
                alpha=0.75,
            )

            ax_2.set_ylabel("SNR")
            ax_2.set_xlabel("Frequency")
            ax_2.loglog()
            ax_2.grid()
            ax_2.set_ylim((cur_snr[1:, comp_ix].min(), cur_snr[1:, comp_ix].max()))
            fig.tight_layout()

        if not output_dir.is_dir():
            output_dir.mkdir(parents=True)
        fig.savefig(output_dir / f"{cur_id}.png")
        plt.close()

def multi_fig(
    ind_fig_size: Tuple[float, float], n_rows: int, n_cols: int, dpi: int = 100
) -> plt.Figure:
    """
    Returns a figure with n_rows rows and n_cols columns, with each plot
    of size as specified by ind_fig_size

    Parameters
    ----------
    ind_fig_size: pair of ints
        The size size of each plot
    n_rows: int
    n_cols: int
    dpi: int, optional

    Returns
    -------
    Figure
    """
    fig_size = (ind_fig_size[0] * n_cols, ind_fig_size[1] * n_rows)
    return plt.figure(figsize=fig_size, dpi=dpi)