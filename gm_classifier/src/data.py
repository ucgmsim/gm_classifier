from typing import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from . import utils


def load_dataset(
    features_dir: Path,
    label_ffp: Path,
    labels: Sequence[str] = None,
    features: Sequence[str] = None,
):
    label_df = pd.read_csv(label_ffp, index_col=0)
    label_df = label_df.loc[
        (label_df.processed == True)
        & (label_df.investigate == False)
        & (label_df.good_drop == False)
        & (label_df.multi_drop == False)
        & (label_df.other_drop == False)
    ]

    label_dfs = []
    for cur_comp in ["x", "y", "z"]:
        cur_df = label_df.loc[:, [f"score_{cur_comp}", f"fmin_{cur_comp}", "multi", "malf"]]
        cur_df.columns = ["score", "fmin", "multi", "malf"]
        label_dfs.append(cur_df)

    feature_dfs = utils.load_features_from_dir(features_dir, concat=False)

    avail_ids = np.intersect1d(
        label_dfs[0].index.values.astype(str), feature_dfs[0].index.values.astype(str)
    )
    feature_dfs = [cur_df.loc[avail_ids] for cur_df in feature_dfs]
    label_dfs = [cur_df.loc[avail_ids] for cur_df in label_dfs]

    label_df = pd.concat(label_dfs, axis=0)
    feature_df = pd.concat(feature_dfs, axis=0)

    if labels is not None:
        label_df = label_df.loc[:, labels]

    if features is not None:
        feature_df = feature_df.loc[:, features]

    assert np.all(feature_df.index == label_df.index)
    return feature_df, label_df
