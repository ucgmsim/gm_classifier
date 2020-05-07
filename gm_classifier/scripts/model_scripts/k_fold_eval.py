import json
import multiprocessing as mp
from pathlib import Path
from typing import Union, Dict, List, Callable

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold

import gm_classifier as gm

# ----- Config -----
label_dir = (
    "/Users/Clus/code/work/gm_classifier/data/records/local/training_data/labels"
)

features_dir = "/Users/Clus/code/work/gm_classifier/data/records/local/training_data/all_records_features/200429"
# features_dir = "/Users/Clus/code/work/gm_classifier/data/records/local/training_data/all_records_features/200429_alg_f_min_feature"

# output_dir = Path("/Users/Clus/code/work/tmp/gm_classifier/record_based/k_means/test_1")
output_dir = Path("/Users/Clus/code/work/gm_classifier/results/200506_results/k_fold")

ts_data_dir = Path("/Volumes/Claudio/work/records/ts/labelled")

# ----- Data -----
label_df = gm.utils.load_labels_from_dir(label_dir, f_min_100_value=10)
feature_df = gm.utils.load_features_from_dir(features_dir)

train_df = pd.merge(
    feature_df, label_df, how="inner", left_index=True, right_index=True
)

train_df.to_csv(output_dir / "training_data.csv", index_label="record_id")

def get_model_fn():
    gm_model = gm.model.CnnSnrModel.from_custom_config(gm.RecordCompModel.model_config)
    return gm_model


snr_freq_values = gm.RecordCompModel.snr_freq_values
feature_config = gm.RecordCompModel.feature_config
feature_names, feature_config, snr_feature_keys = gm.utils.get_feature_details(
    feature_config, snr_freq_values
)

# Get the data
X_features = train_df.loc[:, feature_names]
X_snr = np.stack(
    (
        train_df.loc[:, np.char.add(snr_feature_keys, "_X")].values,
        train_df.loc[:, np.char.add(snr_feature_keys, "_Y")].values,
        train_df.loc[:, np.char.add(snr_feature_keys, "_Z")].values,
    ),
    axis=2,
)
y = train_df.loc[:, gm.RecordCompModel.label_names]

# --- Run k-means ---
compile_kwargs = gm.RecordCompModel.compile_kwargs
fit_kwargs = {"batch_size": 32, "epochs": 250, "verbose": 2}

# Evaluation loss function, only used to include individual loss in the result dataframe,
# hence reduction = 'none'
eval_loss_fn = gm.training.CustomScaledLoss(
    gm.RecordCompModel.score_loss_fn,
    gm.RecordCompModel.f_min_loss_fn,
    gm.RecordCompModel.score_values,
    gm.RecordCompModel.f_min_weights,
    gm.RecordCompModel.score_loss_fn(
        tf.constant(0.0, dtype=tf.float32), tf.constant(1.0, dtype=tf.float32)
    ),
    gm.RecordCompModel.f_min_loss_fn(
        tf.constant(0.1, dtype=tf.float32), tf.constant(2, dtype=tf.float32)
    ),
    reduction="none",
)

result_df = gm.training.run_k_means(
    X_features,
    X_snr,
    y,
    y.index.values.astype(str),
    feature_config,
    get_model_fn,
    compile_kwargs,
    fit_kwargs,
    output_dir,
    n_folds=10,
    eval_loss_fn=eval_loss_fn,
)

result_df.to_csv(output_dir / "results.csv", index_label="record_id")

# --- Plots ---
fig_size = (16, 10)

# General plots
cmap = "coolwarm"
ms = 4.0
for cur_comp in ["X", "Y", "Z"]:
    cur_score_col_name = f"score_{cur_comp}"
    cur_score_est_col_name = f"score_{cur_comp}_est"

    cur_f_min_col_name = f"f_min_{cur_comp}"
    cur_f_min_est_col_name = f"f_min_{cur_comp}_est"

    fig, ax, train_scatter = gm.plots.plot_true_vs_est(
        result_df.loc[:, cur_score_est_col_name],
        result_df.loc[:, cur_score_col_name]
        + np.random.normal(0, 0.01, result_df.shape[0]),
        c_train="b",
        title="Score",
        min_max=(
            result_df.loc[:, cur_score_est_col_name].min(),
            result_df.loc[:, cur_score_est_col_name].max(),
        ),
        scatter_kwargs={"s": ms},
        fig_kwargs={"figsize": fig_size},
        # output_ffp=output_dir / f"score_true_vs_est_{cur_comp}.png",
    )
    ax.set_ylim((-0.03, 1.03))
    ax.grid()
    fig.savefig(output_dir / f"score_true_vs_est_{cur_comp}.png")
    plt.close()

    f_min_min, f_min_max = (
        np.min(label_df[f"f_min_{cur_comp}"]),
        np.max(label_df[f"f_min_{cur_comp}"]),
    )
    fig, ax, scatter = gm.plots.plot_true_vs_est(
        result_df.loc[:, cur_f_min_est_col_name],
        result_df.loc[:, cur_f_min_col_name],
        title="f_min",
        c_train=result_df.loc[:, cur_score_col_name],
        min_max=(f_min_min, f_min_max),
        scatter_kwargs={"s": ms, "cmap": cmap},
        fig_kwargs={"figsize": fig_size},
    )
    ax.loglog()
    fig.colorbar(scatter)
    fig.savefig(output_dir / f"f_min_true_vs_est_{cur_comp}_log_scale.png")
    plt.close()

    fig, ax, scatter = gm.plots.plot_true_vs_est(
        result_df.loc[:, cur_f_min_est_col_name],
        result_df.loc[:, cur_f_min_col_name],
        title="f_min",
        c_train=result_df.loc[:, cur_score_col_name],
        min_max=(f_min_min, f_min_max),
        scatter_kwargs={"s": ms, "cmap": cmap},
        fig_kwargs={"figsize": fig_size},
    )
    ax.set_xlim((0.0, 2.0))
    ax.set_ylim((0.0, 2.0))
    plt.savefig(output_dir / f"f_min_true_vs_est_zoomed_{cur_comp}.png")
    plt.close()

    fig, ax, scatter = gm.plots.plot_true_vs_est(
        result_df.loc[:, cur_f_min_est_col_name],
        result_df.loc[:, cur_f_min_col_name],
        title="f_min",
        c_train=result_df.loc[:, cur_score_col_name],
        min_max=(f_min_min, f_min_max),
        scatter_kwargs={"s": ms, "cmap": cmap},
        fig_kwargs={"figsize": fig_size},
    )
    fig.colorbar(scatter)
    fig.savefig(output_dir / f"f_min_true_vs_est_{cur_comp}.png")
    plt.close()


# --- Individual plots ---
if ts_data_dir is None:
    exit()

plot_output_dir = output_dir / "plots"
plot_output_dir.mkdir()

# Filter results by score
for score in result_df.score_X.unique():
    cur_dir = plot_output_dir / f"{score:.2f}".replace(".", "p")
    cur_dir.mkdir()


# Load the data
n_procs = 3
with mp.Pool(n_procs) as p:
    results = p.map(
        gm.utils.load_ts_data,
        [
            ts_data_dir / cur_record_id
            for cur_record_id in label_df.index.values.astype(str)
        ],
    )

acc_ts = [result[1] for result in results if result[1] is not None]
snr = [result[4] for result in results if result[1] is not None]
records_ids = np.asarray([result[0] for result in results if result[1] is not None])
ft_freq = [result[3] for result in results if result[1] is not None]

for rec_ix, cur_id in enumerate(result_df.index.values):
    print(f"Processing {rec_ix + 1}/{result_df.shape[0]}")
    if cur_id not in records_ids:
        print(f"No timeseries data for record {cur_id}, skipping")
        continue

    # Get the relevant data
    ts_ix = np.flatnonzero(records_ids == cur_id)[0]
    cur_acc = acc_ts[ts_ix]
    cur_snr = snr[ts_ix]
    cur_ft_freq = ft_freq[ts_ix]

    fig, axes = plt.subplots(2, 3, figsize=(22, 10))

    for comp_ix, cur_comp in enumerate(["X", "Y", "Z"]):
        ax_1, ax_2 = axes[0, comp_ix], axes[1, comp_ix]
        cur_f_min_true = result_df.loc[cur_id, f"f_min_{cur_comp}"]
        cur_f_min_est = result_df.loc[cur_id, f"f_min_{cur_comp}_est"]
        cur_score_true = result_df.loc[cur_id, f"score_{cur_comp}"]
        cur_score_est = result_df.loc[cur_id, f"score_{cur_comp}_est"]

        ax_1.plot(np.arange(cur_acc.shape[0]) * (1/200), cur_acc[:, comp_ix], label="X")
        ax_1.set_ylabel("Acc")
        ax_1.set_xlabel("Time")
        ax_1.set_title(
            f"Score - True: {cur_score_true} Est: {cur_score_est:.2f}, f_min - True: {cur_f_min_true}, Est: {cur_f_min_est:.2f}"
        )
        ax_1.grid()

        ax_2.plot(cur_ft_freq, cur_snr[:, comp_ix], "b")
        # ax_2.plot(
        #     snr_freq_values,
        #     X_snr[np.flatnonzero(X_features.index == cur_id)[0], :, comp_ix],
        #     "g",
        # )
        ax_2.plot(
            [cur_f_min_true, cur_f_min_true],
            [cur_snr[1:, comp_ix].min(), cur_snr[1:, comp_ix].max()],
            "k--", linewidth=1.4,
        )
        ax_2.plot(
            [cur_f_min_est, cur_f_min_est],
            [cur_snr[1:, comp_ix].min(), cur_snr[1:, comp_ix].max()],
            "r--", linewidth=1.4,
        )
        ax_2.plot([cur_ft_freq.min(), cur_ft_freq.max()],
                  [2.0, 2.0], color="gray", linestyle="--", linewidth=1.0, alpha=0.75)

        ax_2.set_ylabel("SNR")
        ax_2.set_xlabel("Frequency")
        ax_2.loglog()
        ax_2.grid()
        ax_2.set_ylim((cur_snr[1:, comp_ix].min(), cur_snr[1:, comp_ix].max()))
        fig.tight_layout()

        cur_output_dir = plot_output_dir / f"{cur_score_true:.2f}".replace(".", "p")
    fig.savefig(cur_output_dir / f"{cur_id}.png")
    plt.close()

exit()
