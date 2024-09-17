"""
Script that plots the worst validation & training records for a given model
Have to set 'export CUDA_VISIBLE_DEVICES=-1'
Works on re-labelled labels!
"""

import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import typer

import gm_classifier as gmc


def main(
    data_dir: Path,
    output_dir: Path,
    ko_matrices_dir: Path,
    model_output_dir: Path,
    labels_ffp: Path,
):

    labels_df = pd.read_csv(labels_ffp, index_col=0)
    orig_results_df = pd.read_csv(model_output_dir / "results.csv", index_col=0)

    z_df = orig_results_df.loc[orig_results_df.component == "Z"]
    z_df.columns = np.char.add(z_df.columns.values.astype(str), "_Z")
    results_df = pd.merge(
        pd.merge(
            orig_results_df.loc[orig_results_df.component == "X"],
            orig_results_df.loc[orig_results_df.component == "Y"],
            suffixes=("_X", "_Y"),
            left_on="record",
            right_on="record",
        ).set_index("record"),
        z_df,
        left_index=True,
        right_on="record_Z",
    ).set_index("record_Z")
    results_df = results_df.drop(columns=["component_X", "component_Y", "component_Z"])
    results_df.index.name = "record_id"

    val_ids, train_ids = (
        np.load(str(model_output_dir / "val_ids.npy")),
        np.load(str(model_output_dir / "train_ids.npy")),
    )

    val_data_df = results_df.loc[val_ids].copy(deep=True)
    val_data_df["res_X"] = labels_df.loc[val_ids].score_x - val_data_df.score_mean_X
    val_data_df["res_Y"] = labels_df.loc[val_ids].score_y - val_data_df.score_mean_Y
    val_data_df["res_Z"] = labels_df.loc[val_ids].score_z - val_data_df.score_mean_Z
    val_data_df["abs_res_mean"] = np.mean(
        np.abs(val_data_df.loc[:, ["res_X", "res_Y", "res_Z"]].values), axis=1
    )
    plot_val_ids = val_data_df.sort_values(
        "abs_res_mean", ascending=False
    ).index.values[:10]

    train_data_df = results_df.loc[train_ids].copy(deep=True)
    train_data_df["res_X"] = (
        labels_df.loc[train_ids].score_x - train_data_df.score_mean_X
    )
    train_data_df["res_Y"] = (
        labels_df.loc[train_ids].score_y - train_data_df.score_mean_Y
    )
    train_data_df["res_Z"] = (
        labels_df.loc[train_ids].score_z - train_data_df.score_mean_Z
    )
    train_data_df["abs_res_mean"] = np.mean(
        np.abs(train_data_df.loc[:, ["res_X", "res_Y", "res_Z"]].values), axis=1
    )
    plot_train_ids = train_data_df.sort_values(
        "abs_res_mean", ascending=False
    ).index.values[:10]

    avail_record_ffps = np.asarray(list(data_dir.rglob(f"**/*.V1A")), dtype=str)
    avail_record_ids = np.asarray(
        [gmc.records.get_record_id(record_ffp) for record_ffp in avail_record_ffps],
        dtype=str,
    )

    val_record_ffps = avail_record_ffps[np.isin(avail_record_ids, plot_val_ids)]
    train_record_ffps = avail_record_ffps[np.isin(avail_record_ids, plot_train_ids)]

    # Load the konno matrices
    konno_matrices = {
        matrix_id: np.load(os.path.join(ko_matrices_dir, f"KO_{matrix_id}.npy"))
        for matrix_id in [1024, 2048, 4096, 8192, 16384, 32768]
    }

    val_output_dir = output_dir / "val"
    val_output_dir.mkdir()
    for cur_record_ffp in val_record_ffps:
        gmc.plots.plot_record_full(
            cur_record_ffp, konno_matrices, val_output_dir, orig_results_df
        )

    train_output_dir = output_dir / "train"
    train_output_dir.mkdir()
    for cur_record_ffp in train_record_ffps:
        gmc.plots.plot_record_full(
            cur_record_ffp, konno_matrices, train_output_dir, orig_results_df
        )


    return


if __name__ == "__main__":
    typer.run(main)
