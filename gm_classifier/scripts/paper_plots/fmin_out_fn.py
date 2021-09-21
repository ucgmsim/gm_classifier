from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import gm_classifier as gmc

import typer

def main(output_ffp: Path):
    z = np.linspace(-7.5, 7.5, 1000)

    fig = plt.figure(figsize=(9, 6), dpi=200)
    fmin_fn = gmc.model.get_fmin_sigmoid()

    plt.plot(z, fmin_fn(z))

    plt.xlim(z.min(), z.max())

    plt.xlabel(r"Weighted output neuron sum, $z$")
    plt.ylabel("Minimum usable frequency")

    fig.tight_layout()
    fig.savefig(output_ffp)


if __name__ == '__main__':
    typer.run(main)