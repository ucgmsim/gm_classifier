from pathlib import Path
from typing import Dict, List, Tuple, Union, Sequence

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from . import features
from .records import Record, get_record_id, RecordError
from .console import console


def plot_record_simple(record: Record):
    """Plots the acceleration time-series of the given record"""
    t = np.arange(record.size) * record.dt
    fig = plt.figure(figsize=(12, 9))

    for ix, (cur_acc, cur_comp) in enumerate(zip(record.acc_arrays, ["X", "Y", "Z"])):
        ax = fig.add_subplot(3, 1, ix + 1, sharex=ax1 if ix > 0 else None)

        if ix == 0:
            ax1 = ax

        ax.plot(t, cur_acc)

    ax.set_xlabel("Time (s)")
    ax1.set_title(record.id)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)

    return fig


def plot_record_full(
    record_ffp: Path,
    ko_matrices: Dict,
    output_dir: Path,
    results_df: pd.DataFrame = None,
    label_df: pd.DataFrame = None,
):
    """Plots the acceleration time-series and SNR series for the specified record

    Is also able to add either the predictions or labels to plot
    """
    try:
        record = Record.load(str(record_ffp))
        record.record_preprocesing()
    except RecordError as ex:
        console.print(
            f"[red]\n{get_record_id(str(record_ffp))}: "
            f"Failed to load record. Due to the error - {ex.error_type}, ffp: {record_ffp}[/]"
        )
        return

    # Create the time vector
    t = np.arange(record.size) * record.dt
    p_wave_ix, s_wave_ix, p_prob_series, s_prob_series = features.run_phase_net(
        np.stack((record.acc_1, record.acc_2, record.acc_v), axis=1)[np.newaxis, ...],
        record.dt,
        t,
        return_prob_series=True,
    )

    freq_arrays, snr_arrays = [], []
    if p_wave_ix == 0:
        console.print(
            f"[orange1]\n{record.id}: P-wave ix == 0, SNR can therefore not be calculated[/]"
        )
        freq_arrays = snr_arrays = [None, None, None]
    else:
        for cur_acc in record.acc_arrays:
            # Compute the fourier transform
            try:
                ft_data = features.comp_fourier_data(
                    np.copy(cur_acc), t, record.dt, p_wave_ix, ko_matrices
                )
            except KeyError as ex:
                console.print(
                    f"[red]\nRecord {record.id} - No konno matrix found for size {ex.args[0]}. Skipping![/]"
                )
                freq_arrays.append(None)
                snr_arrays.append(None)
            else:
                freq_arrays.append(ft_data.ft_freq_signal)
                snr_arrays.append(ft_data.smooth_ft_signal / ft_data.smooth_ft_pe)

    # Create the plot
    fig = plt.figure(figsize=(16, 10))

    wave_form_ax, snr_ax = None, None
    for ix, (cur_channel, cur_acc, cur_freq, cur_snr, c) in enumerate(
        zip(
            ["H1", "H2", "V"],
            record.acc_arrays,
            freq_arrays,
            snr_arrays,
            ["b", "r", "g"],
        )
    ):
        wave_form_ax = fig.add_subplot(4, 2, (ix * 2) + 1, sharex=wave_form_ax)
        wave_form_ax.plot(t, cur_acc, label=cur_channel, c=c)
        wave_form_ax.axvline(t[p_wave_ix], c="k", linewidth=1)
        wave_form_ax.axvline(t[s_wave_ix], c="k", linewidth=1)
        wave_form_ax.legend()

        if ix == 2:
            wave_form_ax.set_xlabel("time")

        if cur_snr is None:
            continue

        snr_ax = fig.add_subplot(4, 2, (ix + 1) * 2, sharex=snr_ax)
        snr_ax.plot(cur_freq, cur_snr, c=c)
        snr_ax.axhline(2.0, c="k", linestyle="--", linewidth=1)
        snr_ax.set_xlim(0.0, 10.0)
        snr_ax.set_ylim(0.0, 20.0)
        snr_ax.set_xscale("log")

        snr_ax.grid(b=True, which="major", linestyle="-", alpha=0.75)
        snr_ax.grid(b=True, which="minor", linestyle="--", alpha=0.5)

        if ix == 2:
            snr_ax.set_xlabel("ln(freq)")

    t_prob = np.arange(p_prob_series.size) * (1.0 / 100.0)
    prob_ax = fig.add_subplot(4, 2, 7, sharex=wave_form_ax)
    prob_ax.plot(t_prob, p_prob_series, label="p-wave prob")
    prob_ax.plot(t_prob, s_prob_series, label="s-wave prob")
    prob_ax.legend()

    if results_df is not None:
        cur_result_df = results_df.loc[results_df.record == record.id]

        if cur_result_df.shape[0] == 3:
            ax_text = fig.add_subplot(4, 2, 8)
            ax_text.text(
                0.0,
                0.7,
                f"{record.id}\n\n"
                + "\n\n".join(
                    [
                        f"{cur_comp[1]} - "
                        f"Score: {cur_result_df.loc[record.id + cur_comp, 'score_mean']:.2f} "
                        f"+/- {cur_result_df.loc[record.id + cur_comp, 'score_std']:.3f}, "
                        f"Fmin: {cur_result_df.loc[record.id + cur_comp, 'fmin_mean']:.2f} "
                        f"+/- {cur_result_df.loc[record.id + cur_comp, 'fmin_std']:.3f}, "
                        f"Multi: {cur_result_df.loc[record.id + cur_comp, 'multi_mean']:.2f} "
                        f"+/- {cur_result_df.loc[record.id + cur_comp, 'multi_std']:.3f}"
                        for cur_comp in ["_X", "_Y", "_Z"]
                    ]
                ),
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax_text.transAxes,
            )

            ax_text.axison = False
        else:
            console.print(
                f"[orange1]\nRecord {record.id} - "
                f"Results missing for some components, skipping result labels[/]"
            )
    elif label_df is not None:
        if record.id in label_df.index:
            cur_label_row = label_df.loc[record.id]

            ax_text = fig.add_subplot(4, 2, 8)
            ax_text.text(
                0.0,
                0.7,
                f"{record.id}\n\n"
                + "\n\n".join(
                    [
                        f"{cur_comp[1]} - "
                        f"Score: {cur_label_row.loc[f'Man_Score{cur_comp}']:.2f} "
                        f"Fmin: {cur_label_row.loc[f'Min_Freq{cur_comp}']:.2f} "
                        for cur_comp in ["_X", "_Y", "_Z"]
                    ]
                ),
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax_text.transAxes,
            )

            ax_text.axison = False
        else:
            console.print(
                f"[orange1]\nRecord {record.id} - " f"Labels missing, skipping label[/]"
            )

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)

    fig.savefig(output_dir / f"{record.id}.png")
    plt.close(fig)

    return


def plot_loss(
    history: Dict,
    ax: plt.Axes = None,
    output_ffp: str = None,
    add_loss_keys: Sequence[Tuple[str, str]] = None,
    ylim: Tuple[float, float] = None,
    fig_kwargs: Dict = None,
):
    """Plots single output loss"""
    fig = None
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
            label=f"Validation - {np.min(history['val_loss']):.2f}",
        )

    if add_loss_keys is not None:
        for cur_key, cur_line_params in add_loss_keys:
            ax.plot(
                epochs,
                history[cur_key],
                cur_line_params,
                label=f"{cur_key} - {np.min(history[cur_key]):.2f}",
            )

    ax.legend()
    ax.grid(True, alpha=0.5, linestyle="--")

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()

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


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_est: np.ndarray,
    output_dir: Path,
    prefix: str,
    label: str,
    wandb_save: bool = True,
):
    """Creates a confusion matrix plot"""
    conf_matrix = confusion_matrix(y_true, y_est)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(
                x=j, y=i, s=conf_matrix[i, j], va="center", ha="center", size="xx-large"
            )

    plt.xlabel("Predictions", fontsize=18)
    plt.ylabel("Actuals", fontsize=18)
    plt.title(f"{prefix} {label} - Confusion Matrix", fontsize=18)

    out_name = f"{prefix.lower()}_{label.lower()}_conf_matrix"
    fig.savefig(output_dir / f"{out_name}.png")

    if wandb_save:
        wandb.log({out_name: fig})
        wandb.save(str(f"{out_name}.png"))

    plt.close()


def plot_true_vs_est(
    y_est: np.ndarray,
    y_true: np.ndarray,
    y_val_est: np.ndarray = None,
    y_val_true: np.ndarray = None,
    c_train: Union[str, np.ndarray] = None,
    c_val: Union[str, np.ndarray] = None,
    y_est_unc: np.ndarray = None,
    output_ffp: str = None,
    ax: plt.Axes = None,
    min_max: Tuple[float, float] = None,
    title: str = None,
    fig_kwargs: Dict = {},
    scatter_kwargs: Dict = {},
    error_plot_kwargs: Dict = {},
):
    """Plots true (y-axis) vs estimated (x-axis)"""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(**fig_kwargs)

    label = None if y_val_est is None else "training"

    train_scatter = ax.scatter(
        y_est, y_true, label=label, c=c_train, marker=".", **scatter_kwargs
    )
    if y_val_est is not None and y_val_true is not None and y_est_unc is not None:
        ax.errorbar(
            y_val_est,
            y_val_true,
            xerr=y_est_unc,
            c=c_val,
            marker="s",
            label="validation",
            **error_plot_kwargs,
        )
        ax.legend()
    elif y_val_est is not None and y_val_true is not None:
        ax.scatter(
            y_val_est,
            y_val_true,
            label="validation",
            c=c_val,
            marker=".",
            **scatter_kwargs,
        )

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


def plot_fmin_true_vs_est(
    label_df: pd.DataFrame,
    y_est: pd.Series,
    output_dir: Path,
    title: str = None,
    wandb_save: bool = True,
    zoom: bool = False,
):
    y = label_df.fmin

    fig, ax = plt.subplots(figsize=(16, 10))
    scatter = ax.scatter(
        y_est.values, y.values, c=label_df.score, s=4.0, cmap="coolwarm", marker=".",
    )

    ax.plot([0.1, 10.0], [0.1, 10.0], "k--")

    if zoom:
        ax.set_xlim((0.0, 2.0))
        ax.set_ylim((0.0, 2.0))
        title = f"{title}_zoomed"

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Estimated")
    ax.set_ylabel("True")
    ax.grid(True, alpha=0.5, linestyle="--")
    cbar = fig.colorbar(scatter)
    cbar.set_label("Quality score (True)")

    fig.tight_layout()

    out_ffp = output_dir / f"fmin_true_vs_est_{title}.png"
    fig.savefig(out_ffp)

    if wandb_save:
        wandb.log({f"fmin_true_vs_est_{title}": fig})
        wandb.save(str(out_ffp))


def plot_score_true_vs_est(
    label_df: pd.DataFrame,
    y_est: pd.Series,
    output_dir: Path,
    title: str = None,
    wandb_save: bool = True,
):
    y = label_df.score
    multi_eq_ids = label_df.index.values.astype(str)[label_df.multi]
    malf_ids = label_df.index.values.astype(str)[label_df.malf]

    noise = pd.Series(
        index=label_df.index, data=np.random.normal(0, 0.01, label_df.shape[0])
    )

    fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
    ax.scatter(
        y_est.values, y.values + noise.values, label="Normal", c="b", s=4.0, marker=".",
    )
    ax.scatter(
        y_est.loc[multi_eq_ids].values,
        y.loc[multi_eq_ids].values + noise.loc[multi_eq_ids].values,
        label="Multi EQ",
        c="b",
        s=7.0,
        marker="x",
    )
    ax.scatter(
        y_est.loc[malf_ids].values,
        y.loc[malf_ids].values + noise.loc[malf_ids].values,
        label="Malfunction",
        c="b",
        s=7.0,
        marker="s",
    )
    ax.scatter([0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0], s=20, c="k")

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Estimated")
    ax.set_ylabel("True")
    ax.legend()
    ax.grid(True, alpha=0.5, linestyle="--")

    fig.tight_layout()

    out_ffp = output_dir / f"score_true_vs_est_{title}.png"
    fig.savefig(out_ffp)

    if wandb_save:
        wandb.log({f"score_true_vs_est_{title}": fig})
        wandb.save(str(out_ffp))
