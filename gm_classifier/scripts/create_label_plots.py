"""Script for generating plots for labelling of records
Have to set 'export CUDA_VISIBLE_DEVICES=-1'
"""
import os
from pathlib import Path
from typing import Dict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import typer

import matplotlib.pyplot as plt
import gm_classifier as gmc


def process_record(record_ffp: Path, ko_matrices: Dict, output_dir: Path):
    try:
        record = gmc.records.Record.load_v1a(str(record_ffp))
    except gmc.records.RecordError as ex:
        typer.echo(
            f"\n{gmc.records.get_record_id(str(record_ffp))}: "
            f"Failed to load record. Due to the error - {ex.error_type}, ffp: {record_ffp}",
            color="red",
        )
        return

    # Create the time vector
    t = np.arange(record.size) * record.dt
    p_wave_ix, s_wave_ix, p_prob_series, s_prob_series = gmc.features.run_phase_net(
        np.stack((record.acc_1, record.acc_2, record.acc_v), axis=1)[np.newaxis, ...],
        record.dt,
        t,
        return_prob_series=True,
    )

    if p_wave_ix == 0:
        typer.echo(
            f"\n{record.id}: P-wave ix == 0, SNR can therefore not be calculated. Skipped.",
            color="red",
        )
        return

    freq_arrays, snr_arrays = [], []
    for cur_acc in record.acc_arrays:
        # Compute the fourier transform
        ft_data = gmc.features.comp_fourier_data(
            np.copy(cur_acc), t, record.dt, p_wave_ix, ko_matrices
        )

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

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)

    fig.savefig(output_dir / f"{record.id}.png")
    plt.close(fig)

    return


def main(
    data_dir: Path, record_list_ffp: Path, output_dir: Path, ko_matrices_dir: Path
):
    # Get the record ids of interest
    with open(record_list_ffp, "r") as f:
        record_ids = f.readlines()

    # Strip and drop empty lines
    record_ids = np.unique(np.asarray(
        [
            record_id.strip()
            for record_id in record_ids
            if len(record_id.strip()) > 0 and record_id.strip()[0] != "#"
        ],
        dtype=str,
    ))

    print(f"Searching for record files")
    avail_record_ffps = np.asarray(list(data_dir.rglob(f"**/*.V1A")), dtype=str)
    avail_record_ids = np.asarray(
        [gmc.records.get_record_id(record_ffp) for record_ffp in avail_record_ffps],
        dtype=str,
    )

    # Remove duplicates (just take the first file)
    avail_record_ids, indices = np.unique(avail_record_ids, return_index=True)
    avail_record_ffps = avail_record_ffps[indices]

    # Filter
    record_ffps = np.unique(avail_record_ffps[np.isin(avail_record_ids, record_ids)])

    # Load the konno matrices
    typer.echo("Loading Konno matrices")
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
                process_record(cur_record_ffp, konno_matrices, output_dir)
            except:
                typer.echo("\nFailed to process record")


if __name__ == "__main__":
    typer.run(main)
