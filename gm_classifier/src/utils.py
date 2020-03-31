import os
import glob
from typing import Dict, Union, Tuple

import pandas as pd
import numpy as np


def load_features_from_dir(feature_dir: str, glob_filter: str = "*comp*.csv"):
    """Loads all features for each component and generates a single
    component based dataframe

    Each sample is identified by a unique sample_id (see get_sample_id)

    Note: All specified files are expected to have the same feature columns,
    along with the columns [record_id, event_id, station]!

    Parameters
    ----------
    feature_dir: string
        The component feature files are expected to have the
         format *_X.csv (extension can be something else as well)
    glob_filter: string, optional
        Glob filter that allows filtering which files to use
        (in the specified directory)

    Returns
    -------
    dataframe
    """
    feature_files = glob.glob(os.path.join(feature_dir, glob_filter))

    columns, dfs = None, []
    for cur_ffp in feature_files:
        cur_comp = os.path.basename(cur_ffp).split(".")[0].split("_")[-1]

        # Load
        cur_df = pd.read_csv(cur_ffp, index_col="record_id")

        # Change the index to sample_id instead of record_id
        cur_df["record_id"] = cur_df.index.values
        cur_df["sample_id"] = get_sample_id(
            cur_df.record_id.values.astype(str), cur_comp
        )
        cur_df.set_index("sample_id", drop=True, inplace=True)

        if columns is None:
            columns = cur_df.columns.values.astype(str).sort()
        else:
            if not np.all(cur_df.colums.values.astype(str) == columns):
                raise ValueError(
                    f"The columns of the specified feature files "
                    f"do not match, current file {cur_ffp}"
                )

        dfs.append(cur_df)

    df = pd.concat(dfs, axis=0)
    return df


def load_labels_from_dir(label_dir: str, glob_filter: str = "labels_*.csv", drop_invalid: bool = True):
    """Loads all labels into a single dataframe and creates
    component based sample labels

    Each sample is identified by a unique sample_id (see get_sample_id)

    Expected columns are:
    [Record_ID, Source_ID, Site_ID, Man_Score_X, Man_Score_Y,
    Man_Score_Z, Min_Freq_X, Min_Freq_Y, Min_Freq_Z]

    Parameters
    ----------
    label_dir: str
        Directory that contains the label files
    glob_filter: str, optional
        Glob filter that allows filtering which files to use
        (in the specified label_dir)

    Returns
    -------
    dataframe
    """
    # Load and combine
    label_files = glob.glob(os.path.join(label_dir, glob_filter))
    dfs = [pd.read_csv(cur_file, index_col="Record_ID") for cur_file in label_files]
    df = pd.concat(dfs)


    # Generate record-component based labels
    df = gen_comp_labels(df)

    if drop_invalid:
        inv_mask = (df.score > 1.0) | df.score.isna() | (df.f_min >= 100.0) | df.f_min.isna()
        df = df.loc[~inv_mask]

    return df




def gen_comp_labels(df: pd.DataFrame):
    """Creates component based labels and gives each
    sample a unique ID of the format "{record_id}_{component}"
    where the component is either X, Y or Z

    Parameters
    ----------
    df: dataframe
        Same column expectation as load_labels_from_dir

    Returns
    -------
    dataframe
        Resulting dataframe will have 3x number of entries of original df
    """
    result_dict = {}
    for record_id, row in df.iterrows():
        for cur_comp in ["X", "Y", "Z"]:
            result_dict[get_sample_id(record_id, cur_comp)] = [
                record_id,
                row.Source_ID,
                row.Site_ID,
                cur_comp,
                row[f"Man_Score_{cur_comp}"],
                row[f"Min_Freq_{cur_comp}"],
            ]

    return pd.DataFrame.from_dict(
        result_dict,
        orient="index",
        columns=["record_id", "event_id", "station", "component", "score", "f_min"],
    )


def get_sample_id(record_id: Union[str, np.ndarray], component: [str, np.ndarray]):
    """Creates the sample id, which has the format
    "{record_id}_{component}", where the component is
    either X, Y or Z

    Parameters
    ----------
    record_id: str
    component: str

    Returns
    -------
    sample_id: str
    """
    if isinstance(record_id, str) and isinstance(component, str):
        return f"{record_id}_{component}"
    else:
        return np.char.add(np.char.add(record_id, "_"), component)


def score_weighting(
    train_df: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    ids_train: np.ndarray,
):
    """Weighs the records based on their score,
    score in:
        [0.0, 1.0]      => 1.0
        [0.25, 0.75]    => 0.75
        [0.5]           => 0.5

    Parameters
    ----------
    train_df: Dataframe
    X_train: numpy array of floats
    y_train: numpy array of floats
    ids_train: numpy array of strings

    Returns
    -------
    numpy array of floats
        The sample weights for each record in X_train (and y_train obviously)
    """
    scores = train_df.loc[ids_train, "score"].values

    weights = np.full(X_train.shape[0], np.nan, dtype=float)
    weights[np.isin(scores, [0.0, 1.0])] = 1.0
    weights[np.isin(scores, [0.25, 0.75])] = 0.75
    weights[scores == 0.5] = 0.5

    assert np.all(~np.isnan(weights))

    return weights
