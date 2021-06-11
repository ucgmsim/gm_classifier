from pathlib import Path
import argparse

import matplotlib.pyplot as plt

import gm_classifier as gmc


def main(
    record_ffp: Path,
    output_dir: Path = None,
):
    record = gmc.records.Record.load(str(record_ffp))

    fig = gmc.plots.plot_record(record)

    if output_dir is None:
        plt.show()
    else:
        fig.savefig(str(output_dir / record_ffp.stem))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("record_ffp", type=Path, help="Path to record file to plot")
    parser.add_argument("--output_dir", type=Path, help="Directory for saving the plot", default=None)

    args = parser.parse_args()

    main(args.record_ffp, output_dir=args.output_dir)
