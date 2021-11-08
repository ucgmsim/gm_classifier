"""Script for generating the Konno matrices"""
import os
import argparse
import time

import numpy as np

import gm_classifier as gmc

def main(output_dir):
    # Generate the Konno matrices
    dt = 0.005

    ko_matrix_sizes = (
        gmc.records.KO_MATRIX_SIZES
        if os.environ.get("KO_MATRIX_SIZES") is None
        else [
            int(cur_size.strip())
            for cur_size in os.environ.get("KO_MATRIX_SIZES").split(",")
        ]
    )
    ft_lens = np.asarray(ko_matrix_sizes) * 2

    for ft_len in ft_lens:
        print(f"Computing konno {int(ft_len / 2)}")
        start_time = time.time()
        cur_konno = gmc.features.features.get_konno_matrix(ft_len, dt=dt)
        print(f"Took {time.time() - start_time}\n")

        np.save(os.path.join(output_dir, f"konno_{int(ft_len / 2)}.npy"), cur_konno)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    main(args.output_dir)
