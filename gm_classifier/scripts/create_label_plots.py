"""Script for generating plots for labelling of records
Have to set 'export CUDA_VISIBLE_DEVICES=-1'
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
from gm_classifier.src.console import console


def main(
    data_dir: Path,
    record_list_ffp: Path,
    output_dir: Path,
    ko_matrices_dir: Path,
    results_ffp: Path = None,
    labels_ffp: Path = None,
):
    results_df = None if results_ffp is None else pd.read_csv(results_ffp, index_col=0)
    label_df = None if labels_ffp is None else pd.read_csv(labels_ffp, index_col=0)

    # Get the record ids of interest
    with open(record_list_ffp, "r") as f:
        record_ids = f.readlines()

    # Strip and drop empty lines
    record_ids = np.unique(
        np.asarray(
            [
                record_id.strip()
                for record_id in record_ids
                if len(record_id.strip()) > 0 and record_id.strip()[0] != "#"
            ],
            dtype=str,
        )
    )

    print(f"Searching for record files")
    avail_record_ffps_v1a = np.asarray(list(data_dir.rglob(f"**/*.V1A")), dtype=str)
    avail_record_ffps_mseed = np.asarray(list(data_dir.rglob(f"**/*.mseed")), dtype=str)
    avail_record_ffps = np.concatenate([avail_record_ffps_v1a, avail_record_ffps_mseed])
    avail_record_ids = np.asarray(
        [gmc.records.get_record_id(record_ffp) for record_ffp in avail_record_ffps],
        dtype=str,
    )

    # Remove duplicates (just take the first file)
    avail_record_ids, indices = np.unique(avail_record_ids, return_index=True)
    avail_record_ffps = avail_record_ffps[indices]

    # Filter
    record_ffps = np.unique(avail_record_ffps[np.isin(avail_record_ids, record_ids)])

    if record_ffps.size == 0:
        console.print("[red]No record files corresponding to specified ids were found. Quitting![/]")
        return

    if record_ffps.size < record_ids.size:
        missing_records_str = '\n'.join(record_ids[~np.isin(record_ids, avail_record_ids)])
        console.print(f"[orange1]No record files were found for the following ids:\n{missing_records_str}[/]")

    # Load the konno matrices
    console.print("Loading Konno matrices")
    konno_matrices = {
        matrix_id: np.load(os.path.join(ko_matrices_dir, f"konno_{matrix_id}.npy"))
        for matrix_id in [1024, 2048, 4096, 8192, 16384, 32768]
    }

    # Create an empty dataframe for it
    sort_ind = np.argsort(record_ids)
    record_ids, record_ffps = record_ids[sort_ind], record_ffps[sort_ind]

    empty_df = pd.DataFrame(
        data=np.full((record_ids.size, 6), fill_value=np.nan),
        columns=[
            "Man_Score_X",
            "Man_Score_Y",
            "Man_Score_Z",
            "Min_Freq_X",
            "Min_Freq_Y",
            "Min_Freq_Z",
        ],
        index=record_ids,
    )
    empty_df.to_csv(output_dir / "labels.csv", index_label="Record_ID")

    # Process
    with typer.progressbar(record_ffps) as progress:
        for cur_record_ffp in progress:
            try:
                gmc.plots.plot_record_full(
                    cur_record_ffp, konno_matrices, output_dir, results_df=results_df, label_df=label_df
                )
            except:
                console.print("\nFailed to process record")


if __name__ == "__main__":
    typer.run(main)
