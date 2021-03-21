from pathlib import Path
from typing import Dict, List, Tuple, Union

import wandb
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from .RecordCompModel import RecordCompModel
from . import constants as const
from .records import Record


def plot_record(record: Record):
    # Generate the plot
    t = np.arange(record.size) * record.dt
    fig = plt.figure(figsize=(12, 9))

    for ix, (cur_acc, cur_comp) in enumerate(zip(record.acc_arrays, ["X", "Y", "Z"])):
        ax = fig.add_subplot(3, 1, ix + 1, sharex=ax1 if ix > 0 else None)

        if ix == 0:
            ax1 = ax

        ax.plot(t, cur_acc)

    ax.set_xlabel("Time (s)")
    ax.set_title(record.id)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)

    return fig


def plot_loss(
    history: Dict, ax: plt.Axes = None, output_ffp: str = None, fig_kwargs: Dict = None
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

    ax.legend()
    ax.grid(True, alpha=0.5, linestyle="--")
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


def plot_score_true_vs_est(
    label_df: pd.DataFrame,
    y_est: pd.Series,
    output_dir: Path,
    title: str = None,
    wandb_save: bool = True,
):
    y = label_df.score
    multi_eq_ids = label_df.index.values.astype(str)[label_df.multi_eq]
    malf_ids = label_df.index.values.astype(str)[label_df.malf]

    noise = pd.Series(
        index=label_df.index, data=np.random.normal(0, 0.01, label_df.shape[0])
    )

    fig, ax = plt.subplots(figsize=(16, 10))
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
        wandb.save(out_ffp)


def plot_fmin_true_vs_est(
    label_df: pd.DataFrame,
    y_est: pd.Series,
    output_dir: Path,
    title: str = None,
    wandb_save: bool = True,
    zoom: bool = False,
):
    y = label_df.f_min

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


# -------- old -----------


def create_eval_plots(
    output_dir: Path,
    model: RecordCompModel,
    feature_dfs: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    label_dfs: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    history: Dict,
    n_preds: int = 10,
):
    with sns.axes_style("whitegrid"):
        fig_size = (16, 10)

        fig, ax = plot_loss(
            history,
            # output_ffp=str(output_dir / "loss_plot.png"),
            fig_kwargs={"figsize": fig_size},
        )
        ax.set_ylim((0, 1))
        plt.savefig(str(output_dir / f"loss_plot.png"))
        plt.close()

        # Predict train and validation
        ids = np.concatenate((train_ids, val_ids))
        est_df, model_uncertainty = model.predict(
            [cur_feature_df.loc[ids] for cur_feature_df in feature_dfs], n_preds=n_preds
        )
        # est_df.to_csv(output_dir / "est_df.csv", index_label="record_id")

        data_df = pd.merge(
            pd.concat(label_dfs, axis=1),
            est_df,
            how="inner",
            left_index=True,
            right_index=True,
        )
        y_est, y = (
            data_df.loc[:, ["score_X", "score_Y", "score_Z"]],
            data_df.loc[:, "score"],
        )
        multi_eq_ids = data_df.index.values.astype(str)[
            data_df.multi_eq.iloc[:, 0] == True
        ]
        malf_ids = data_df.index.values.astype(str)[data_df.malf.iloc[:, 0] == True]
        assert np.all(y_est.index == y.index)

        train_noise = pd.Series(
            index=np.tile(train_ids, 3),
            data=np.random.normal(0, 0.01, train_ids.size * 3),
        )
        val_noise = pd.Series(
            index=np.tile(val_ids, 3), data=np.random.normal(0, 0.01, val_ids.size * 3)
        )

        # Training
        fig, ax = plt.subplots(figsize=fig_size)
        ax.scatter(
            y_est.loc[train_ids].values.ravel(),
            y.loc[train_ids].values.ravel() + train_noise.loc[train_ids],
            label="Training",
            c="b",
            s=4.0,
            marker=".",
        )
        train_multi_eq_ids = train_ids[np.isin(train_ids, multi_eq_ids)]
        ax.scatter(
            y_est.loc[train_multi_eq_ids].values.ravel(),
            y.loc[train_multi_eq_ids].values.ravel()
            + train_noise.loc[train_multi_eq_ids],
            label="Training - Multi EQ",
            c="b",
            s=7.0,
            marker="x",
        )
        train_malf_ids = train_ids[np.isin(train_ids, malf_ids)]
        ax.scatter(
            y_est.loc[train_malf_ids].values.ravel(),
            y.loc[train_malf_ids].values.ravel() + train_noise.loc[train_malf_ids],
            label="Training - Malfunction",
            c="b",
            s=7.0,
            marker="s",
        )
        ax.scatter(
            [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0], s=20, c="k"
        )
        ax.set_xlabel("Estimated")
        ax.set_ylabel("True")
        ax.set_title("Training")
        wandb.log({"score_true_vs_est_training": fig})
        fig.savefig(output_dir / "score_true_vs_est_training.png")

        # Validation
        fig, ax = plt.subplots(figsize=fig_size)
        ax.scatter(
            y_est.loc[val_ids].values.ravel(),
            y.loc[val_ids].values.ravel() + val_noise.loc[val_ids],
            label="Validation",
            c="r",
            s=4.0,
            marker=".",
        )
        val_multi_eq_ids = val_ids[np.isin(val_ids, multi_eq_ids)]
        ax.scatter(
            y_est.loc[val_multi_eq_ids].values.ravel(),
            y.loc[val_multi_eq_ids].values.ravel() + val_noise.loc[val_multi_eq_ids],
            label="Validation - Multi EQ",
            c="r",
            s=7.0,
            marker="X",
        )
        val_malf_ids = val_ids[np.isin(val_ids, malf_ids)]
        ax.scatter(
            y_est.loc[val_malf_ids].values.ravel(),
            y.loc[val_malf_ids].values.ravel() + val_noise.loc[val_malf_ids],
            label="Validation - Malfunction",
            c="r",
            s=7.0,
            marker="s",
        )
        ax.scatter(
            [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0], s=20, c="k"
        )
        ax.set_xlabel("Estimated")
        ax.set_ylabel("True")
        ax.set_title("Validation")
        wandb.log({"score_true_vs_est_validation": fig})
        fig.savefig(output_dir / "score_true_vs_est_validation.png")

        # # Create wandb plots
        # train_data_table = wandb.Table(
        #     data=[
        #         (x, y)
        #         for x, y in zip(
        #             y_est.loc[train_ids].values.ravel(),
        #             y.loc[train_ids].values.ravel() + train_noise,
        #         )
        #     ],
        #     columns=["estimated", "true"],
        # )
        # wandb.log(
        #     {
        #         "score_true_vs_est_training": wandb.plot.scatter(
        #             train_data_table, "estimated", "true"
        #         )
        #     }
        # )
        #
        # val_data_table = wandb.Table(
        #     data=[
        #         (x, y)
        #         for x, y in zip(
        #             y_est.loc[val_ids].values.ravel(),
        #             y.loc[val_ids].values.ravel() + val_noise,
        #         )
        #     ],
        #     columns=["estimated", "true"],
        # )
        # wandb.log(
        #     {
        #         "score_true_vs_est_validation": wandb.plot.scatter(
        #             val_data_table, "estimated", "true"
        #         )
        #     }
        # )

        cmap = "coolwarm"
        # cmap = sns.color_palette("coolwarm")
        m_size = 4.0
        # Create per component plots
        for ix, cur_comp in enumerate(["X", "Y", "Z"]):
            cur_label_df = label_dfs[ix]
            score_key, f_min_key = f"score_{cur_comp}", f"f_min_{cur_comp}"

            # Plot true vs estimated
            score_min, score_max = (
                np.min(cur_label_df["score"]),
                np.max(cur_label_df["score"]),
            )
            fig, ax, train_scatter = plot_true_vs_est(
                est_df.loc[train_ids, score_key],
                cur_label_df.loc[train_ids, "score"]
                + np.random.normal(0, 0.01, train_ids.size),
                # y_est_unc=model_uncertainty.loc[val_ids, score_key],
                c_train="b",
                c_val="r",
                y_val_est=est_df.loc[val_ids, score_key],
                y_val_true=cur_label_df.loc[val_ids, "score"]
                + np.random.normal(0, 0.01, val_ids.size),
                title="Score",
                min_max=(score_min, score_max),
                scatter_kwargs={"s": m_size},
                error_plot_kwargs={
                    "linestyle": "None",
                    "ms": m_size,
                    "elinewidth": 0.75,
                },
                fig_kwargs={"figsize": fig_size},
                output_ffp=output_dir / f"score_true_vs_est_{cur_comp}.png",
            )

            wandb.run.summary[f"score_true_vs_est_{cur_comp}"] = wandb.Image(
                np.asarray(
                    imageio.imread(output_dir / f"score_true_vs_est_{cur_comp}.png")
                )
            )

            f_min_min, f_min_max = (
                np.min(cur_label_df[f"f_min"]),
                np.max(cur_label_df[f"f_min"]),
            )
            fig, ax, train_scatter = plot_true_vs_est(
                est_df.loc[train_ids, f_min_key],
                cur_label_df.loc[train_ids, "f_min"],
                # + np.random.normal(0, 0.025, y_train.iloc[:, f_min_ix].size),
                c_train=cur_label_df.loc[train_ids, "score"].values,
                c_val=cur_label_df.loc[val_ids, "score"].values,
                y_val_est=est_df.loc[val_ids, f_min_key],
                y_val_true=cur_label_df.loc[val_ids, "f_min"],
                # + np.random.normal(0, 0.025, self.y_val.iloc[:, f_min_ix].size),
                title="f_min",
                min_max=(f_min_min, f_min_max),
                scatter_kwargs={"s": m_size, "cmap": cmap},
                fig_kwargs={"figsize": fig_size},
                # output_ffp=output_dir / f"f_min_true_vs_est_{cur_comp}.png",
            )
            cbar = fig.colorbar(train_scatter)
            cbar.set_label("Quality score (True)")
            plt.savefig(output_dir / f"f_min_true_vs_est_{cur_comp}.png")
            plt.close()

            fig, ax, train_scatter = plot_true_vs_est(
                est_df.loc[train_ids, f_min_key],
                cur_label_df.loc[train_ids, "f_min"],
                # + np.random.normal(0, 0.025, y_train.iloc[:, f_min_ix].size),
                c_train=cur_label_df.loc[train_ids, "score"].values,
                c_val=cur_label_df.loc[val_ids, "score"].values,
                y_val_est=est_df.loc[val_ids, f_min_key],
                y_val_true=cur_label_df.loc[val_ids, "f_min"],
                # + np.random.normal(0, 0.025, self.y_val.iloc[:, f_min_ix].size),
                title="f_min",
                min_max=(f_min_min, f_min_max),
                scatter_kwargs={"s": m_size, "cmap": cmap},
                fig_kwargs={"figsize": fig_size},
                # output_ffp=output_dir / f"f_min_true_vs_est_{cur_comp}.png",
            )
            ax.set_xlim((0.0, 2.0))
            ax.set_ylim((0.0, 2.0))
            cbar = fig.colorbar(train_scatter)
            cbar.set_label("Quality score (True)")
            plt.savefig(output_dir / f"f_min_true_vs_est_zoomed_{cur_comp}.png")
            plt.close()
