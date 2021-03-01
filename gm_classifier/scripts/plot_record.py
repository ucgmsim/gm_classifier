from pathlib import Path
import argparse

import matplotlib.pyplot as plt

import gm_classifier as gmc

def main(record_ffp: Path):
    record = gmc.records.Record.load_v1(record_ffp)

    fig = gmc.plots.plot_record(record)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("record_ffp", type=Path, help="Path to record file to plot")

    args = parser.parse_args()

