import os
import glob
import argparse
from pathlib import Path
from typing import Union, Dict

import numpy as np
import pandas as pd

import gm_classifier as gm


def get_series_data(record_ffp: str, ko_matrices: Union[str, Dict[int, np.ndarray]]):
    """Retrieves the acceleration time series and fourier transform"""
    gf = gm.GeoNet_File(record_ffp)

    # Error checking & pre-processing
    gf = gm.records.record_preprocesing(gf)

    p_wave_ix = gm.features.get_p_wave_ix(gf.comp_1st.acc, gf.comp_2nd.acc, gf.comp_up.acc, gf.comp_1st.dt)

    t = np.arange(gf.comp_1st.acc.shape[0]) * gf.comp_1st.delta_t
    ft_data_X = gm.features.comp_fourier_data(gf.comp_1st.acc, t, gf.comp_1st.delta_t, p_wave_ix, ko_matrices)
    ft_data_Y = gm.features.comp_fourier_data(gf.comp_2nd.acc, t, gf.comp_2nd.delta_t, p_wave_ix, ko_matrices)
    ft_data_Z = gm.features.comp_fourier_data(gf.comp_up.acc, t, gf.comp_up.delta_t, p_wave_ix, ko_matrices)

    # Combine acceleration time-series, shape [n_timesteps, 3 components]
    acc = np.concatenate(
        (
            gf.comp_1st.acc[:, np.newaxis],
            gf.comp_2nd.acc[:, np.newaxis],
            gf.comp_up.acc[:, np.newaxis],
        ),
        axis=1,
    )

    # Combine fourier transform data, shape [n_frequencies, 3 components]
    ft = np.concatenate(
        (ft_data_X.ft[:, np.newaxis], ft_data_Y.ft[:, np.newaxis], ft_data_Z.ft[:, np.newaxis]), axis=1
    )
    ft_smooth = np.concatenate(
        (ft_data_X.smooth_ft[:, np.newaxis], ft_data_Y.smooth_ft[:, np.newaxis], ft_data_Z.smooth_ft[:, np.newaxis]), axis=1
    )

    snr = np.concatenate(
        (ft_data_X.snr[:, np.newaxis], ft_data_Y.snr[:, np.newaxis], ft_data_Z.snr[:, np.newaxis]), axis=1
    )

    ft_freq = ft_data_X.fr_freq
    ft_fre_pe = ft_data_Y.ft_freq_pe

    assert np.isclose(np.isclose(ft_data_X.fr_freq, ft_data_Y.fr_freq) & np.isclose(ft_data_X.fr_freq, ft_data_Z.fr_freq))
    assert np.isclose(np.isclose(ft_data_X.ft_freq_pe, ft_data_Y.ft_freq_pe) & np.isclose(ft_data_X.ft_freq_pe, ft_data_Z.ft_freq_pe))


    # Collect some meta data
    meta_data = {
        "acc_length": gf.comp_1st.acc.size,
        "acc_dt": gf.comp_1st.delta_t,
        "acc_duration": gf.comp_1st.acc.size * gf.comp_1st.delta_t,
        "ft_length": ft.shape[0],
        "snr_length": snr.shape[0]
    }

    return acc, ft, ft_smooth, snr, ft_freq, ft_fre_pe, meta_data


def main(
    record_dir: str,
    output_dir: str,
    ko_matrices_dir: str,
    low_mem_usage: bool = False,
    skip_existing: bool = False,
):
    output_dir = Path(output_dir)

    print(f"Searching for record files")
    record_files = np.asarray(
        glob.glob(os.path.join(record_dir, "**/*.V1A"), recursive=True), dtype=str
    )

    # Load the Konno matrices into memory
    if ko_matrices_dir is not None and not low_mem_usage:
        print(f"Loading Konno matrices into memory")
        ko_matrices = {
            matrix_id: np.load(os.path.join(ko_matrices_dir, f"konno_{matrix_id}.npy"))
            for matrix_id in [1024, 2048, 4096, 8192, 16384, 32768]
        }
    else:
        ko_matrices = ko_matrices_dir

    failed_records = {
        gm.records.RecordError: {
            err_type: [] for err_type in gm.records.RecordErrorType
        },
        gm.features.FeatureError: {
            err_type: [] for err_type in gm.features.FeatureErrorType
        },
        "empty_file": [],
        "other": [],
    }
    meta_data = {}
    for ix, record_ffp in enumerate(record_files):
        record_id = gm.records.get_record_id(record_ffp)
        cur_out_dir = output_dir / f"{record_id}"

        print(f"Processing record {record_id}, {ix + 1}/{record_files.size}")
        if cur_out_dir.is_dir() and skip_existing:
            print(f"Result directory already exists, skipping record.")
            continue

        try:
            cur_acc, cur_ft, cur_smooth_ft, cur_snr, cur_ft_freq, cur_ft_freq_pe, cur_meta_data = get_series_data(
                record_ffp, ko_matrices
            )
        except gm.records.RecordError as ex:
            failed_records[gm.records.RecordError][ex.error_type].append(record_id)
        except gm.records.EmptyFile as ex:
            failed_records["empty_file"].append(record_id)
        except Exception as ex:
            failed_records["other"].append(record_id)
        else:
            meta_data[record_id] = cur_meta_data

            cur_out_dir = output_dir / f"{record_id}"
            if not cur_out_dir.is_dir():
                cur_out_dir.mkdir()

            np.save(cur_out_dir / f"{record_id}_acc.npy", cur_acc.astype(np.float32))
            np.save(cur_out_dir / f"{record_id}_raw_ft.npy", cur_ft.astype(np.float32))
            np.save(cur_out_dir / f"{record_id}_smooth_ft.npy", cur_smooth_ft.astype(np.float32))
            np.save(cur_out_dir / f"{record_id}_snr.npy", cur_snr.astype(np.float32))
            np.save(cur_out_dir / f"{record_id}_ft_freq.npy", cur_ft_freq.astype(np.float32))
            np.save(cur_out_dir / f"{record_id}_ft_pe_freq.npy", cur_ft_freq_pe.astype(np.float32))

    meta_df = pd.DataFrame.from_dict(meta_data, orient="index", columns=["acc_length", "acc_dt", "acc_duration", "ft_length", "snr_length"])
    meta_df.to_csv(output_dir / "meta_data.csv")

    gm.records.print_errors(failed_records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "record_dir",
        type=str,
        help="The directory that contains the records, searches recursively",
    )
    parser.add_argument(
        "output_dir", type=str, help="The directoy to which the resulting data is saved"
    )
    parser.add_argument(
        "ko_matrices_dir",
        type=str,
        help="Path to the directory that contains the Konno matrices. "
        "Has to be specified if the --low_memory options is used",
    )
    parser.add_argument(
        "--low_memory",
        action="store_true",
        help="If specified will prioritise low memory usage over performance. "
        "Requires --ko_matrices_dir to be specified. ",
        default=False,
    )
    parser.add_argument(
        "--no_overwrite",
        action="store_true",
        help="If specified, existing results will not be overwritten",
        default=False,
    )

    args = parser.parse_args()

    main(
        args.record_dir,
        args.output_dir,
        args.ko_matrices_dir,
        low_mem_usage=args.low_memory,
        skip_existing=args.no_overwrite
    )
