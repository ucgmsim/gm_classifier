import datetime
import os
import glob
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Union, Tuple, List, Iterable


import pandas as pd
import numpy as np
from obspy.signal.trigger import pk_baer
from scipy.signal import detrend

import phase_net as ph
from . import constants as const


def create_run_id() -> str:
    """Creates a run ID based on the month, day & time"""
    id = datetime.datetime.now().strftime("%m%d_%H%M")
    return id


def load_features_from_dir(
    feature_dir: Union[str, Path],
    glob_filter: str = "*comp*.csv",
    merge: bool = True,
    drop_duplicates: bool = True,
    drop_nan: bool = True,
):
    """
    Loads the features dataframes for each component, does not combine them

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
    single dataframe or triplet of dataframes:
        If merged:
            Single dataframe of shape [n_records, 3 x n_features]
        Else:
            In the order _X, _Y, _Z
    """
    feature_files = glob.glob(os.path.join(feature_dir, glob_filter))

    assert (
        len(feature_files) == 3
    ), f"Expected 3 feature files, found {len(feature_files)} instead"

    result_ix = {"X": 0, "Y": 1, "Z": 2}
    result = [None, None, None]
    for cur_ffp in feature_files:
        print(f"Processing file {os.path.basename(cur_ffp)}")
        cur_comp = os.path.basename(cur_ffp).split(".")[0].split("_")[-1]

        cur_df = pd.read_csv(cur_ffp, index_col="record_id")

        if drop_duplicates:
            dup_mask = cur_df.index.duplicated(keep=False)
            cur_df = cur_df.loc[~dup_mask]
            print(f"Dropped {np.count_nonzero(dup_mask)} duplicates")

        if drop_nan:
            feature_cols = cur_df.columns.values[~np.isin(cur_df.columns, ["event_id", "station"])].astype(str)
            nan_mask = np.any(cur_df.loc[:, feature_cols].isna(), axis=1)
            cur_df = cur_df.loc[~nan_mask]
            print(f"Dropped {np.count_nonzero(nan_mask)} samples due to nan-values")

        if merge:
            cur_df.columns = [f"{cur_col}_{cur_comp}" for cur_col in cur_df.columns]

        result[result_ix[cur_comp]] = cur_df

    if merge:
        result_df = result[0].merge(
            result[1], left_index=True, right_index=True, suffixes=(False, False)
        )
        result_df = result_df.merge(
            result[2], left_index=True, right_index=True, suffixes=(False, False)
        )
        return result_df

    return result


def load_labels_from_dir(
    label_dir: str,
    glob_filter: str = "labels_*.csv",
    drop_na: bool = True,
    drop_f_min_101: bool = True,
    multi_eq_value: float = None,
    malf_value: float = None,
    f_min_100_value: float = None,
    drop_duplicates: bool = True,
    merge: bool = True
):
    """
    Loads all labels, single row per record

    Parameters
    ----------
    label_dir: str
        Directory that contains the label files
    glob_filter: str, optional
        Glob filter that allows filtering which files to use
        (in the specified label_dir)
    drop_na: bool, optional
        If true, drops samples with invalid/bad scores or f_fmin

    Returns
    -------
    dataframe
    """
    # Load and combine
    label_files = glob.glob(os.path.join(label_dir, glob_filter))
    dfs = [pd.read_csv(cur_file, index_col=0) for cur_file in label_files]
    df = pd.concat(dfs)

    # Rename
    df = df.rename(
        columns={
            "Record_ID": "record_id",
            "Source_ID": "source_id",
            "Site_ID": "station",
            "Man_Score_X": "score_X",
            "Man_Score_Y": "score_Y",
            "Man_Score_Z": "score_Z",
            "Min_Freq_X": "f_min_X",
            "Min_Freq_Y": "f_min_Y",
            "Min_Freq_Z": "f_min_Z",
        }
    )
    df.index.name = "record_id"

    # Apply the f_min max value limit
    if f_min_100_value is not None:
        df.loc[(df.f_min_X > f_min_100_value) & (df.f_min_X <= 100), "f_min_X"] = f_min_100_value
        df.loc[(df.f_min_Y > f_min_100_value) & (df.f_min_Y <= 100), "f_min_Y"] = f_min_100_value
        df.loc[(df.f_min_Z > f_min_100_value) & (df.f_min_Z <= 100), "f_min_Z"] = f_min_100_value

    # Drop invalid
    if drop_na:
        na_mask = (df.score_X.isna()
            | df.f_min_X.isna()
            | df.score_Y.isna()
            | df.f_min_Y.isna()
            | df.score_Z.isna()
            | df.f_min_Z.isna()
        )
        df = df.loc[~na_mask]

    if drop_f_min_101:
        f_min_101_mask = (df.f_min_X >= 100.0) | (df.f_min_Y >= 100.0) | (df.f_min_Z >= 100.0)
        print(f"Dropped {np.count_nonzero(na_mask)} na records records")
        df = df.loc[~f_min_101_mask]

    multi_eq_mask = (df.score_X == 2.0) | (df.score_Y == 2.0) | (df.score_Z == 2.0)
    if multi_eq_value is not None:
        df.loc[multi_eq_mask, ["score_X", "score_Y", "score_Z"]] = multi_eq_value
        print(f"Set {np.count_nonzero(multi_eq_mask)} malfunctioned records score to {multi_eq_value}")
    else:
        print(f"Dropped {np.count_nonzero(multi_eq_mask)} malfunctioned records")
        df = df.loc[~multi_eq_mask]

    malf_mask = (df.score_X == 3.0) | (df.score_Y == 3.0) | (df.score_Z == 3.0)
    if malf_value is not None:
        df.loc[malf_mask, ["score_X", "score_Y", "score_Z"]] = malf_value
        print(f"Set {np.count_nonzero(malf_mask)} multiple earthquake records score to {malf_value}")
    else:
        print(f"Dropped {np.count_nonzero(malf_mask)} multiple earthquake records")
        df = df.loc[~malf_mask]

    # Drop duplicates
    if drop_duplicates:
        dup_mask = df.index.duplicated(keep=False)
        df = df.loc[~dup_mask]
        print(f"Dropped {np.count_nonzero(dup_mask)} duplicates")

    if merge:
        return df
    else:
        label_dfs = [df.loc[:, ["score_X", "f_min_X"]], df.loc[:, ["score_Y", "f_min_Y"]], df.loc[:, ["score_Z", "f_min_Z"]]]
        return [cur_df.rename(columns={f"score_{cur_comp}": "score", f"f_min_{cur_comp}": "f_min"}) for cur_comp, cur_df in  zip(const.COMPONENTS, label_dfs)]


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


def get_full_feature_config(feature_config: Dict) -> Tuple[List[str], Dict]:
    """Generates the full feature config, i.e. for all 3 components

    Parameters
    ----------
    feature_config: dictionary
        The general feature config, i.e. not component specific, this will
        be extended for all 3 components

    Returns
    -------
    feature_names: list of strings
        The feature names
    feature_config: dictionary
        The component based feature config (i.e. 3x as many entries
        as original feature config)

    """
    feature_config_X = {f"{key}_X": val for key, val in feature_config.items()}
    feature_config_Y = {f"{key}_Y": val for key, val in feature_config.items()}
    feature_config_Z = {f"{key}_Z": val for key, val in feature_config.items()}
    feature_config = {**feature_config_X, **{**feature_config_Y, **feature_config_Z}}
    feature_names = [key for key in feature_config.keys() if not "*" in key]

    return feature_names, feature_config


def get_feature_details(
    feature_config: Dict, snr_freq_values: np.ndarray
) -> Tuple[Iterable[str], Dict, List[str]]:
    """Retrieves the feature details required for training of a
    record-component multi-output model

    Parameters
    ----------
    feature_config: dictionary
        The general feature config, i.e. not component specific, this will
        be extended for all 3 components
    snr_freq_values: numpy array of floats
        The SNR frequencies to use in for the SNR series input

    Returns
    -------
    feature_names: list of strings
        The feature names
    feature_config: dictionary
        The component based feature config (i.e. 3x as many entries
        as original feature config)
    snr_feature_keys: list of strings
        The SNR series keys into the feature dataframe
    """
    feature_names, feature_config = get_full_feature_config(feature_config)
    snr_feature_keys = [f"snr_value_{freq:.3f}" for freq in snr_freq_values]

    return feature_names, feature_config, snr_feature_keys


def load_record_ts_data(
    record_ts_data_dir: Path, record_dt: float = None, dt: float = None
):
    """Loads the time series data for a single record,
    that was extracted using the extract_time_series.py script
    """
    record_id = record_ts_data_dir.name
    if record_ts_data_dir.is_dir():
        acc_ffp = record_ts_data_dir / f"{record_id}_acc.npy"
        ft_ffp_signal = record_ts_data_dir / f"{record_id}_smooth_ft_signal.npy"
        ft_freq_signal = record_ts_data_dir / f"{record_id}_ft_freq_signal.npy"
        snr_ffp = record_ts_data_dir / f"{record_id}_snr.npy"

        acc_ts = np.load(acc_ffp)
        acc_ts = detrend(acc_ts).astype(np.float32)

        # Down/Up sample if required
        if record_dt is not None and dt is not None and not np.isclose(record_dt, dt):
            t = np.arange(acc_ts.shape[0]) * record_dt
            t_new = np.arange(acc_ts.shape[0]) * dt
            acc_new = np.full((t_new.shape[0], 3), np.nan)

            acc_new[:, 0] = np.interp(t_new, t, acc_ts[:, 0])
            acc_new[:, 1] = np.interp(t_new, t, acc_ts[:, 1])
            acc_new[:, 2] = np.interp(t_new, t, acc_ts[:, 2])

            acc_ts = acc_new

        ft_ts_signal = np.load(ft_ffp_signal)
        ft_freq_signal = np.load(ft_freq_signal)

        snr_ts = np.load(snr_ffp)

        return record_id, acc_ts, ft_ts_signal, ft_freq_signal, snr_ts

    return record_id, None, None, None, None


def load_ts_data(
    record_ids: Iterable[str],
    ts_data_dir: Union[Path, str],
    meta_df: pd.DataFrame,
    dt: float = None,
    n_procs: int = 3,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Loads the timeseries data for the specified records

    Parameters
    ----------
    record_ids: iterable of strings
        Record ids for which to load the ts data
    ts_data_dir: sting or path
        Root timeseries data directory (as generated by
        the extract_time_series.py script)
    meta_df: dataframe
        The meta data,required for the dt of each record
    dt: float
        The target dt for all records afer loading

    Returns
    -------
    record_ids: list of strings
    acc_ts: list of numpy arrays
    snr: list of numpy arrays
    ft_freq: list of numpy arrays
    """
    ts_data_dir = ts_data_dir if isinstance(ts_data_dir, Path) else Path(ts_data_dir)

    with mp.Pool(n_procs) as p:
        results = p.starmap(
            load_record_ts_data,
            [
                (ts_data_dir / cur_record_id, meta_df.loc[cur_record_id, "acc_dt"], dt)
                for cur_record_id in record_ids
            ],
        )

    acc_ts = [result[1] for result in results if result[1] is not None]
    snr = [result[4] for result in results if result[1] is not None]
    record_ids = np.asarray([result[0] for result in results if result[1] is not None])
    ft_freq = [result[3] for result in results if result[1] is not None]

    return record_ids, acc_ts, snr, ft_freq


def run_phase_net(input_data: np.ndarray, dt: float, t: np.ndarray = None, return_prob_series: bool = False):
    """Uses PhaseNet to get the p- & s-wave pick"""
    # Only supports a single record
    assert input_data.shape[0] == 1

    t = t if t is not None else np.arange(input_data.shape[1]) * dt

    # Have to re-sample
    if not np.isclose(dt, 1 / 100):
        dt_new = 1 / 100
        t_new = np.arange(t.max() / dt_new) * dt_new
        input_resampled = np.full((1, t_new.shape[0], 3), np.nan)
        input_resampled[0, :, 0] = np.interp(t_new, t, input_data[0, :, 0])
        input_resampled[0, :, 1] = np.interp(t_new, t, input_data[0, :, 1])
        input_resampled[0, :, 2] = np.interp(t_new, t, input_data[0, :, 2])

        assert np.all(~np.isnan(input_resampled))

        probs = ph.predict(input_resampled)
        p_wave_ix, s_wave_ix = np.argmax(probs[0, :, 1]), np.argmax(probs[0, :, 2])

        # Adjust for original dt
        p_wave_ix = int(np.round((dt_new / dt) * p_wave_ix))
        s_wave_ix = int(np.round((dt_new / dt) * s_wave_ix))
    else:
        probs = ph.predict(input_data)
        p_wave_ix, s_wave_ix = np.argmax(probs[0, :, 1]), np.argmax(probs[0, :, 2])

    if return_prob_series:
        return p_wave_ix, s_wave_ix, probs[0, :, 1], probs[0, :, 2]

    return p_wave_ix, s_wave_ix


def get_p_wave_ix(
    acc_X: np.ndarray,
    acc_Y: np.ndarray,
    acc_Z: np.ndarray,
    dt: float,
    t: np.ndarray = None,
) -> Tuple[int, Union[None, int]]:
    """Performs the p-wave"""

    def p_wave_test(p_wave: float, acc_t_threshold: float):
        """Tests if a specific p-wave pick is 'good', based on
        the picked position exceeding the specified time in the record"""
        return p_wave >= acc_t_threshold

    def get_ix(pick, cur_sample_rate) -> int:
        return int(np.floor(np.multiply(pick, cur_sample_rate)))

    t = t if t is not None else np.arange(acc_X.shape[0]) * dt
    sample_rate = 1.0 / dt

    # PhaseNet
    p_wave_ix, s_wave_ix = run_phase_net(
        np.stack((acc_X, acc_Y, acc_Z), axis=1)[np.newaxis, ...], dt, t=t
    )
    if p_wave_test(p_wave_ix * dt, 3):
        return p_wave_ix, s_wave_ix

    # Obspy picker 2
    cur_p_pick, _ = pk_baer(
        reltrc=acc_Z,
        samp_int=sample_rate,
        tdownmax=20,
        tupevent=60,
        thr1=7.0,
        thr2=12.0,
        preset_len=100,
        p_dur=100,
    )
    p_pick = cur_p_pick * dt
    if p_wave_test(p_pick, 3):
        return get_ix(p_pick, sample_rate), s_wave_ix

    # If all algorithms fail, pick p-wave at 5% record duration, but not less than 3s in.
    p_pick = np.max([3, 0.05 * t[-1]])
    return get_ix(p_pick, sample_rate), s_wave_ix


def to_path(input: Union[str, List[str], List[Path]] = None):
    """Converts a string or list of string to
    Path object/s if required
    """
    if isinstance(input, str):
        return Path(input)
    elif isinstance(input, List):
        return [
            Path(cur_input) if isinstance(cur_input, str) else cur_input
            for cur_input in input
        ]

    return input


def qqqfff_to_binary(
    est_df: pd.DataFrame,
    score_x_th: float = 0.5,
    score_y_th: float = 0.5,
    score_z_th: float = 0.5,
    f_x_th=0.3,
    f_y_th=0.3,
    f_z_th=0.3,
):
    """
    Mapping function for model output to binary classification
    using simple thresholds for each of the outputs

    Parameters
    ----------
    est_df: Dataframe
        Model output dataframe, expected to have columns
        [score_X, score_Y, score_Z, f_min_X, f_min_Y, f_min_Z]
    score_x_th: float
    score_y_th: float
    score_z_th: float
    f_x_th: float
    f_y_th: float
    f_z_th: float
        Thresholds a record has to meet in order to be classified as good.
        Records have to be below f-min thresholds and above the score thresholds

    Returns
    -------
    Series
        Binary classification (True/False) for each record
    """
    quality_mask = (
        (est_df.loc[:, "score_X"].values > score_x_th)
        & (est_df.loc[:, "score_Y"].values > score_y_th)
        & (est_df.loc[:, "score_Z"].values > score_z_th)
    )
    f_min_mask = (
        (est_df.loc[:, "f_min_X"].values > f_x_th)
        & (est_df.loc[:, "f_min_Y"].values > f_y_th)
        & (est_df.loc[:, "f_min_Z"].values > f_z_th)
    )

    usable_mask = quality_mask & f_min_mask

    result_df = pd.Series(index=est_df.index, data=usable_mask)
    return result_df
