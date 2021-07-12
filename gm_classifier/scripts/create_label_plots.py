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
from gm_classifier.src.console import console


def process_record(
    record_ffp: Path,
    ko_matrices: Dict,
    output_dir: Path,
    results_df: pd.DataFrame = None,
):
    try:
        record = gmc.records.Record.load(str(record_ffp))
        record.record_preprocesing()
    except gmc.records.RecordError as ex:
        console.print(
            f"[red]\n{gmc.records.get_record_id(str(record_ffp))}: "
            f"Failed to load record. Due to the error - {ex.error_type}, ffp: {record_ffp}[/]")
        return

    # Create the time vector
    t = np.arange(record.size) * record.dt
    p_wave_ix, s_wave_ix, p_prob_series, s_prob_series = gmc.features.run_phase_net(
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
                ft_data = gmc.features.comp_fourier_data(
                    np.copy(cur_acc), t, record.dt, p_wave_ix, ko_matrices
                )
            except KeyError as ex:
                console.print(f"[red]\nRecord {record.id} - No konno matrix found for size {ex.args[0]}. Skipping![/]")
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
            console.print(f"[orange1]\nRecord {record.id} - "
                          f"Results missing for some components, skipping result labels[/]")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)

    fig.savefig(output_dir / f"{record.id}.png")
    plt.close(fig)

    return


def main(
    data_dir: Path,
    record_list_ffp: Path,
    output_dir: Path,
    ko_matrices_dir: Path,
    results_ffp: Path = None,
):
    results_df = None if results_ffp is None else pd.read_csv(results_ffp, index_col=0)

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
            # try:
            process_record(
                cur_record_ffp, konno_matrices, output_dir, results_df=results_df
            )
            # except:
            #     console.print("\nFailed to process record")


if __name__ == "__main__":
    typer.run(main)
