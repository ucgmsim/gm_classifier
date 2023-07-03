from typing import Sequence, Dict
from pathlib import Path

import numpy as np
import pandas as pd

from . import utils
from .console import console
from .records import Record, RecordError, get_record_id
from . import features



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

def compute_record_snr(record: Record, ko_matrices: Dict):
    # Create the time vector
    t = np.arange(record.size) * record.dt
    p_wave_ix, s_wave_ix, p_prob_series, s_prob_series = features.run_phase_net(
        np.stack((record.acc_1, record.acc_2, record.acc_v), axis=1)[np.newaxis, ...],
        record.dt,
        t,
        return_prob_series=True,
    )

    freq_arrays, snr_arrays = [], []
    if p_wave_ix == 0:
        console.print(
            f"[orange1]\n{record.id}: P-wave ix == 0, SNR can therefore not be calculated[/]"
        )
        freq_arrays = snr_arrays = [None, None, None]
    else:
        for cur_acc in record.acc_arrays:
            # Compute the fourier transform
            try:
                ft_data = features.comp_fourier_data(
                    np.copy(cur_acc), t, record.dt, p_wave_ix, ko_matrices
                )
            except KeyError as ex:
                console.print(
                    f"[red]\nRecord {record.id} - No konno matrix found for size {ex.args[0]}. Skipping![/]"
                )
                freq_arrays.append(None)
                snr_arrays.append(None)
            else:
                freq_arrays.append(ft_data.ft_freq_signal)
                snr_arrays.append(ft_data.smooth_ft_signal / ft_data.smooth_ft_pe)

    return freq_arrays, snr_arrays, t, p_wave_ix, s_wave_ix, p_prob_series, s_prob_series