import os
import json
import glob
import pickle
import datetime
from pathlib import Path
from typing import Union, Any, Dict, Sequence

import yaml
import pandas as pd
import numpy as np
import tensorflow as tf




def create_run_id() -> str:
    """Creates a run ID based on the month, day & time"""
    id = datetime.datetime.now().strftime("%m%d_%H%M")
    return id


def load_features_from_dir(
    feature_dir: Union[str, Path],
    glob_filter: str = "*comp*.csv",
    concat: bool = True,
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
    concat: bool
        If True, combine the different components along the index
        adding _X, _Y, and _Z to each record id accordingly

    Returns
    -------
    single dataframe or triplet of dataframes:
        If concat:
            Single dataframe of shape [3 x n_records, n_features]
        Else:
            In the order _X, _Y, _Z
    """
    print("Loading feature files")
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
            # drop_duplicates ignores the index, so have to make
            # it a column..
            cur_df["record_id"] = cur_df.index
            cur_df.drop_duplicates(inplace=True)
            cur_df.drop(columns=["record_id"], inplace=True)
            print("Dropped duplicated rows, with same values (& index)")

            dup_mask = cur_df.index.duplicated(keep="first")
            cur_df = cur_df.loc[~dup_mask]
            print(
                f"Dropped {np.count_nonzero(dup_mask)} index duplicate."
                f"Kept the first occurrence. Note values of duplicates are note equal."
            )

        if drop_nan:
            feature_cols = cur_df.columns.values[
                ~np.isin(cur_df.columns, ["event_id", "station"])
            ].astype(str)
            nan_mask = np.any(cur_df.loc[:, feature_cols].isna(), axis=1)
            cur_df = cur_df.loc[~nan_mask]
            print(f"Dropped {np.count_nonzero(nan_mask)} samples due to nan-values")

        if concat:
            cur_df.index = [f"{cur_id}_{cur_comp}" for cur_id in cur_df.index]

        result[result_ix[cur_comp]] = cur_df

    if concat:
        result_df = pd.concat(result, axis=0)
        return result_df

    return result


def run_phase_net(
    input_data: np.ndarray,
    dt: float,
    t: np.ndarray = None,
    return_prob_series: bool = False,
):
    """Uses PhaseNet to get the p- & s-wave pick"""
    import phase_net as ph
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


def load_yaml(ffp: Union[Path, str]):
    """Loads data from a yaml file"""
    with open(ffp, "r") as f:
        return yaml.safe_load(f)


def load_picke(ffp: Union[Path, str]):
    """Loads the pickle file"""
    with open(ffp, "rb") as f:
        return pickle.load(f)


def save_print_data(output_dir: Path, wandb_save: bool = True, **kwargs):
    """Save and if specified prints the given data
    To print, set the value to a tuple, with the 2nd entry a
    boolean that indicates whether to print or not,
    i.e. hyperparams=(hyperparams_dict, True) will save and print
    the hyperparameter dictionary
    """
    for key, value in kwargs.items():
        if isinstance(value, tuple) and value[1] is True:
            _print(key, value[0])
            _save(output_dir, key, value[0], wandb_save)
        else:
            _save(output_dir, key, value, wandb_save)


def _print(key: str, data: Union[pd.DataFrame, pd.Series, dict]):
    print(f"{key}:")
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        print(data)
    elif isinstance(data, dict):
        print_dict(data)


def _save(
    output_dir: Path,
    key: str,
    data: Union[pd.DataFrame, pd.Series, dict],
    wandb_save: bool,
):
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        out_ffp = output_dir / f"{key}.csv"
        data.to_csv(out_ffp)
    elif isinstance(data, dict):
        out_ffp = output_dir / f"{key}.json"
        write_to_json(data, out_ffp)
    elif isinstance(data, list):
        out_ffp = output_dir / f"{key}.txt"
        write_to_txt(data, out_ffp)
    elif isinstance(data, np.ndarray):
        out_ffp = output_dir / f"{key}.npy"
        write_np_array(data, out_ffp)
    elif isinstance(data, tf.keras.Model):
        out_ffp = output_dir / f"{key}.png"
        tf.keras.utils.plot_model(
            data,
            to_file=str(out_ffp),
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            expand_nested=True,
        )
    else:
        raise NotImplementedError()

    if wandb_save:
        import wandb
        wandb.save(str(out_ffp))


def write_to_txt(data: Sequence[Any], ffp: Path, clobber: bool = False):
    """Writes each entry in the list on a newline in a text file"""
    if ffp.is_dir() or (not clobber and ffp.exists()):
        raise FileExistsError(f"File {ffp} already exists, failed to save the data!")

    with open(ffp, "w") as f:
        f.writelines([f"{cur_line}\n" for cur_line in data])


def write_to_json(data: Dict, ffp: Path, clobber: bool = False):
    """Writes the data to the specified file path in the json format"""
    if ffp.is_dir() or (not clobber and ffp.exists()):
        raise FileExistsError(f"File {ffp} already exists, failed to save the data!")

    with open(ffp, "w") as f:
        json.dump(data, f, cls=GenericObjJSONEncoder, indent=4)


class GenericObjJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError as ex:
            return str(obj)


def print_dict(data_dict: Dict):
    """Pretty prints a dictionary"""
    print(json.dumps(data_dict, cls=GenericObjJSONEncoder, sort_keys=False, indent=4))


def write_np_array(array: np.ndarray, ffp: Path, clobber: bool = False):
    """Saves the array in the numpy binary format .npy"""
    if ffp.is_dir() or (not clobber and ffp.exists()):
        raise FileExistsError(f"File {ffp} already exists, failed to save the data!")

    np.save(str(ffp), array)


def write_pickle(obj: Any, ffp: Path, clobber: bool = False):
    """Saves the object as a pickle file"""
    if ffp.is_dir() or (not clobber and ffp.exists()):
        raise FileExistsError(f"File {ffp} already exists, failed to save the data!")

    with open(ffp, "wb") as f:
        pickle.dump(obj, f)


