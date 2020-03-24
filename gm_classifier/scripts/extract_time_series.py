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

    # Combine acceleration time-series, shape [n_timesteps, 3 components]
    acc = np.concatenate(
        (
            gf.comp_1st.acc[:, np.newaxis],
            gf.comp_2nd.acc[:, np.newaxis],
            gf.comp_up.acc[:, np.newaxis],
        ),
        axis=1,
    )

    # Compute fourier transform
    duration = gf.comp_1st.acc.size * gf.comp_1st.delta_t
    ft_1, ft_freq_1 = gm.features.compute_fourier(
        gf.comp_1st.acc, gf.comp_1st.delta_t, duration
    )
    ft_2, ft_freq_2 = gm.features.compute_fourier(
        gf.comp_2nd.acc, gf.comp_2nd.delta_t, duration
    )
    ft_v, ft_freq_v = gm.features.compute_fourier(
        gf.comp_up.acc, gf.comp_up.delta_t, duration
    )

    # Combine fourier transform data, shape [n_frequencies, 3 components, 2]
    # Where the last axis is [ft, ft_freq]
    ft = np.concatenate(
        (ft_1[:, np.newaxis], ft_2[:, np.newaxis], ft_v[:, np.newaxis]), axis=1
    )
    ft_freq = np.concatenate(
        (ft_freq_1[:, np.newaxis], ft_freq_2[:, np.newaxis], ft_freq_v[:, np.newaxis]),
        axis=1,
    )

    ft_comb = np.concatenate((ft[:, :, np.newaxis], ft_freq[:, :, np.newaxis]), axis=2)

    # Get appropriate smoothing matrix
    if isinstance(ko_matrices, dict):
        smooth_matrix = ko_matrices[ft_freq_1.size - 1]
    elif isinstance(ko_matrices, str) and os.path.isdir(ko_matrices):
        smooth_matrix = np.load(
            os.path.join(
                ko_matrices,
                gm.features.KONNO_MATRIX_FILENAME_TEMPLATE.format(ft_freq_1.size - 1),
            )
        )
    else:
        raise ValueError(
            "The ko_matrices parameter has to either be a "
            "dictionary with the Konno matrices or a directory "
            "path that contains the Konno matrices files."
        )

    # Apply smoothing
    smooth_ft_1 = np.dot(np.abs(ft_1), smooth_matrix)
    smooth_ft_2 = np.dot(np.abs(ft_2), smooth_matrix)
    smooth_ft_v = np.dot(np.abs(ft_v), smooth_matrix)

    # Combine smoothed fourier transform data, same shape as non-smoothed ft
    smooth_ft = np.concatenate(
        (
            smooth_ft_1[:, np.newaxis],
            smooth_ft_2[:, np.newaxis],
            smooth_ft_v[:, np.newaxis],
        ),
        axis=1,
    )
    smooth_ft_comp = np.concatenate(
        (smooth_ft[:, :, np.newaxis], ft_freq[:, :, np.newaxis]), axis=2
    )

    # Collect some meta data
    meta_data = {
        "acc_length": gf.comp_1st.acc.size,
        "acc_dt": gf.comp_1st.delta_t,
        "acc_duration": duration,
        "ft_length": ft.shape[0],
    }

    return acc, ft_comb, smooth_ft_comp, meta_data


def main(
    record_dir: str, output_dir: str, ko_matrices_dir: str, low_mem_usage: bool = False
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
        print(f"Processing record {record_id}, {ix + 1}/{record_files.size}")
        try:
            cur_acc, cur_ft, cur_smooth_ft, cur_meta_data = get_series_data(
                record_ffp, ko_matrices
            )
        except gm.records.RecordError as ex:
            failed_records[gm.records.RecordError][ex.error_type].append(record_id)
        except gm.records.EmptyFile as ex:
            failed_records["empty_file"].append(record_id)
        # except Exception as ex:
        #     failed_records["other"].append(record_id)
        else:
            meta_data[record_id] = cur_meta_data

            cur_out_dir = output_dir / f"{record_id}"
            if not cur_out_dir.is_dir():
                cur_out_dir.mkdir()

            np.save(cur_out_dir / f"{record_id}_acc.npy", cur_acc)
            np.save(cur_out_dir / f"{record_id}_ft.npy", cur_ft)
            np.save(cur_out_dir / f"{record_id}_smooth_ft.npy", cur_ft)

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

    args = parser.parse_args()

    main(
        args.record_dir,
        args.output_dir,
        args.ko_matrices_dir,
        low_mem_usage=args.low_memory,
    )
