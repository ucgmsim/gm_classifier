from typing import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from . import utils


def load_dataset(
    features_dir: Path,
    label_dir: Path,
    ignore_ids_ffp: Path,
    labels: Sequence[str] = None,
    features: Sequence[str] = None,
):
    label_dfs = utils.load_labels_from_dir(
        str(label_dir),
        f_min_100_value=10,
        drop_na=True,
        drop_f_min_101=True,
        malf_score_value=0.0,
        multi_eq_score_value=0.0,
        merge=False,
        ignore_ids_ffp=ignore_ids_ffp,
    )
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

    return feature_df, label_df


