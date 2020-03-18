import pathlib
import argparse

import pandas as pd

import gm_classifier as gm


def main(
    input_ffp: pathlib.Path, output_ffp: pathlib.Path, label_ffp: pathlib.Path = None
):
    df = pd.read_csv(input_ffp, index_col="record_id")

    name_split = output_ffp.name.split(".")
    meta_ffp = output_ffp.parents[0] / (name_split[0] + "_meta." + name_split[1])

    if label_ffp is not None:
        label_df = pd.read_csv(label_ffp)
        merged_df = pd.merge(df, label_df, left_index=True, right_on="record_id")

        merged_df.loc[:, ["score", "record_id", "event_id", "station_x"]].to_csv(
            meta_ffp, sep="\t", index=False, header=True
        )
        df = merged_df
    else:
        df.loc[:, ["event_id", "station"]].to_csv(meta_ffp, sep="\t", header=True)

    # Standardise the data
    feature_df = df.loc[:, gm.features.FEATURE_NAMES].copy()
    feature_df = (feature_df - feature_df.mean()) / feature_df.std()

    feature_df.to_csv(output_ffp, sep="\t", index=False, header=False)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_ffp",
        type=str,
        help="Input csv file that contains the feature data for each record",
    )
    parser.add_argument("output_tsv_ffp", type=str, help="Output tsv path")
    parser.add_argument(
        "--label_ffp", type=str, help="Path to file that contains labels", default=None
    )

    args = parser.parse_args()

    main(
        pathlib.Path(args.input_ffp),
        pathlib.Path(args.output_tsv_ffp),
        label_ffp=pathlib.Path(args.label_ffp)
        if args.label_ffp is not None
        else args.label_ffp,
    )
