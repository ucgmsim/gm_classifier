"""Script that selects the next set of records to be labelled based
on the MCDropout uncertainty"""
from pathlib import Path

import numpy as np
import pandas as pd
import typer

import gm_classifier as gmc
import ml_tools


def main(results_ffp: Path, labels_dir: Path, output_ffp: Path):
    results_df = pd.read_csv(results_ffp, index_col=0)

    # Get rid of records that have already been labelled
    label_df = gmc.utils.load_labels_from_dir(
        str(labels_dir),
        drop_na=False,
        drop_f_min_101=False,
        malf_score_value=0.0,
        multi_eq_score_value=0.0,
    )
    test_labels_df = pd.read_csv(
        labels_dir / "testing_labels_20210226.csv", index_col=0
    )
    labelled_records = np.concatenate(
        (label_df.index.values.astype(str), test_labels_df.index.values.astype(str))
    )
    results_df = results_df.loc[~np.isin(results_df.record.values, labelled_records)]

    # Sort by uncertainty
    results_df.sort_values("score_std", ascending=False, inplace=True)

    # Select records
    new_record_ids = np.unique(results_df.iloc[:300].record.astype(str))

    # Save
    ml_tools.utils.write_to_txt(new_record_ids, output_ffp)


if __name__ == "__main__":
    typer.run(main)
