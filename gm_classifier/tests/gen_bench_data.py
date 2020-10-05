"""Generates the benchmark data for the V1A files in the bench_data folder"""
from pathlib import Path
import pickle
import argparse

import gm_classifier as gmc

def gen_bench_data(record_ffp: Path, ko_matrices_dir: str, bench_data_dir: Path):
    input_data, add_data = gmc.records.process_record(str(record_ffp), ko_matrices_dir)

    with open(bench_data_dir / f"{record_ffp.name.split('.')[0]}.pickle", "wb") as f:
        pickle.dump({"input_data": input_data, "add_data": add_data}, f)

def main(ko_matrices_dir: str):
    bench_data_dir = Path(__file__).parent / "bench_data"

    record_files_v1a = bench_data_dir.glob("*.V1A")

    for record_file in record_files_v1a:
        gen_bench_data(record_file, ko_matrices_dir, bench_data_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("ko_matrices_dir", type=str, help="Directory that contains the konno matrices")

    args = parser.parse_args()

    main(args.ko_matrices_dir)



