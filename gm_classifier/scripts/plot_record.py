from pathlib import Path
import argparse
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

import gm_classifier as gmc


def main(
    record_ffps: Sequence[Path],
    ko_matrices_dir: Path,
    output_dir: Path = None,
):
    # Load the konno matrices
    konno_matrices = {
        matrix_id: np.load(str(ko_matrices_dir / f"KO_{matrix_id}.npy"))
        for matrix_id in [1024, 2048, 4096, 8192, 16384, 32768]
    }

    for record_ffp in record_ffps:
        gmc.plots.plot_record_full(record_ffp, konno_matrices, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("record_ffps", type=Path, help="Path to record file to plot", nargs="+")
    parser.add_argument("ko_matrices_dir", type=Path, help="Path to the directory "
                                                              "that contains the saved konno matrices")
    parser.add_argument("output_dir", type=Path, help="Directory for saving the plot", default=None)

    args = parser.parse_args()

    main(args.record_ffps, args.ko_matrices_dir, args.output_dir)
