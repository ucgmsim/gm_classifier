"""Contains code for feature/metric extraction from GeoNet ground motion records.
Based on existing implemention
from Xavier Bellagamba (https://github.com/xavierbellagamba/GroundMotionRecordClassifier)
"""
import os
import math
import h5py
import pandas as pd
from enum import Enum
from typing import Tuple, Dict, Union, Any
from collections import namedtuple

import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from obspy.signal.konnoohmachismoothing import calculate_smoothing_matrix

from ..utils import run_phase_net
from ..records import Record
from . import malfunction_features as mal_features
from . import multiple_eq_features as multi_eq_features


class FeatureErrorType(Enum):
    # PGA is zero
    PGA_zero = 1

    # P-wave pick is to early,
    # which means no decent pre-event noise,
    # breaking most features used
    early_p_pick = 2

    # Signal duration is too short for
    # accurate FAS -> SNR, hence breaking features
    short_signal_duration = 3

    missing_ko_matrix = 4


class FeatureError(Exception):
    def __init__(
        self, message: Union[str, None], error_type: FeatureErrorType, **kwargs
    ):
        super(Exception, self).__init__(message)

        self.error_type = error_type
        self.kwargs = kwargs


SNR_FREQ_BINS = [
    (0.1, 0.2),
    (0.2, 0.5),
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, 5.0),
    (5.0, 10.0),
]


KONNO_MATRIX_FILENAME_TEMPLATE = "KO_{}.npy"

FourierData = namedtuple(
    "FourierFeatures",
    [
        "ft",
        "ft_freq",
        "ft_signal",
        "ft_freq_signal",
        "ft_pe",
        "ft_freq_pe",
        "smooth_ft",
        "smooth_ft_signal",
        "smooth_ft_pe",
        "snr",
        "snr_min",
    ],
)


def get_konno_matrix(ft_len, dt: float = 0.005):
    """Computes the Konno matrix"""
    ft_freq = (np.arange(0, ft_len / 2 + 1) * (1.0 / (ft_len * dt))).astype(np.float32)
    return calculate_smoothing_matrix(ft_freq, bandwidth=30, normalize=True)


def get_husid(acc: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the Husid vector, defined as \int{acceleration ** 2.}

    Parameters
    ----------
    acc: numpy array of floats
        The acceleration time-series
    t: numpy array of floats
        The time values for the acc time-series

    Returns
    -------
    husid: numpy array of floats
        The Husid vector
    AI: numpy array of floats
        Normalised Husid vector?
    """
    husid = np.hstack([0.0, cumtrapz(acc ** 2.0, t)])
    AI = husid / max(husid)
    return husid, AI


def compute_husid(acc, t) -> Tuple[np.ndarray, np.ndarray, float, int, int, int]:
    """
    Computes the Husid & AI vector, the Arias IM
    and the Husid 0.05, 0.75, 0.95 indices

    Parameters
    ----------
    acc: numpy array of floats
        The acceleration time-series
    t: numpy array of floats
        The time values for the acc time-series

    Returns
    -------
    husid: numpy array of floats
    AI: numpy array of floats
    Arias: float
    husid_index_5: int
    husid_index_75: int
    husid_index_95: int
    """
    husid, AI = get_husid(acc, t)
    Arias = max(husid)
    husid_index_5 = np.flatnonzero(AI > 0.05)[0]
    husid_index_75 = np.flatnonzero(AI > 0.75)[0]
    husid_index_95 = np.flatnonzero(AI > 0.95)[0]
    return husid, AI, Arias, husid_index_5, husid_index_75, husid_index_95


def compute_fourier(
    acc: np.ndarray, dt: float, duration: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes fourier spectra for acceleration time series

    Parameters
    ----------
    acc: numpy array of floats
        Acceleration time series
    dt: float
        Time step size
    duration: float
        Time series duration in seconds

    Returns
    -------
    ft: numpy array of floats
        Fourier transform
    ft_freq: numpy array of floats
        Frequencies at which fourier amplitudes are computed
    """
    # Create a copy since the acc series is modified in this function..
    acc = acc.copy()

    # Computes fourier spectra for acceleration time series
    npts = len(acc)
    npts_FFT = int(math.ceil(duration) / dt)

    # Compute number of points for efficient FFT
    ft_len = int(2.0 ** math.ceil(math.log(npts_FFT) / math.log(2.0)))
    if npts > ft_len:
        acc = acc[:ft_len]
        npts = len(acc)

    # Apply hanning taper to last 5% of motion
    ntap = int(npts * 0.05)
    acc[npts - ntap :] *= np.hanning(ntap * 2 + 1)[ntap + 1 :]

    # Increase time series length with zeroes for FFT
    accForFFT = np.pad(acc, (0, ft_len - len(acc)), "constant", constant_values=(0, 0))
    ft = np.fft.rfft(accForFFT)

    # Compute frequencies at which fourier amplitudes are computed
    ft_freq = np.arange(0, ft_len / 2 + 1) * (1.0 / (ft_len * dt))
    return ft, ft_freq


def get_freq_ix(ft_freq: np.ndarray, lower: float, upper: float) -> (int, int):
    lower_index = min(np.flatnonzero(ft_freq > lower))
    upper_index = max(np.flatnonzero(ft_freq < upper))
    return lower_index, upper_index


def load_konno_matrix(
    ko_matrices: Union[str, Dict[int, np.ndarray]], ft_freq: np.ndarray
):
    try:
        if isinstance(ko_matrices, dict):
            smooth_matrix = ko_matrices[ft_freq.size - 1]
        elif isinstance(ko_matrices, str) and os.path.isdir(ko_matrices):
            matrix_name = KONNO_MATRIX_FILENAME_TEMPLATE.format(ft_freq.size - 1)
            print(f"Loading Konno matrix {matrix_name}")
            smooth_matrix = np.load(os.path.join(ko_matrices, matrix_name))
        else:
            raise ValueError(
                "The ko_matrices parameter has to either be a "
                "dictionary with the Konno matrices or a directory "
                "path that contains the Konno matrices files."
            )
    except KeyError as ex:
        raise FeatureError(
            f"No Konno matrix of size {ex.args[0]} available",
            FeatureErrorType.missing_ko_matrix,
            key=ex.args[0],
        )

    return smooth_matrix


def comp_fourier_data(
    acc: np.ndarray,
    t: np.ndarray,
    dt: float,
    p_wave_ix: int,
    ko_matrices: Union[str, Dict[int, np.ndarray]],
) -> FourierData:
    """
    Computes the Fourier transform featuers

    Parameters
    ----------
    acc: numpy array of floats
        Acceleration time series
    t: numpy array of floats
        Time values for the time series
    dt: float
        Step size
    p_wave_ix: int
        P-wave arrival index
    ko_matrices: dictionary or str
        Either dictionary of Konno matrices or
       directory path, which contains stored konno matrices
       for on-the-fly loading

    Returns
    -------
    FourierFeatuers
        Named tuple which contains the FT features
    """
    # Calculate fourier spectra
    signal_acc, noise_acc = acc[p_wave_ix:], acc[:p_wave_ix]
    signal_duration, noise_duration = t[-1] - t[p_wave_ix], t[p_wave_ix]

    # ft, ft_freq = compute_fourier(acc, dt, t[-1])
    ft_signal, ft_freq_signal = compute_fourier(signal_acc, dt, signal_duration)
    ft_pe, ft_freq_pe = compute_fourier(noise_acc, dt, signal_duration)

    # Noise scaling, scale factor = (signal length / noise length)**(1/2)
    ft_pe = np.abs(ft_pe) * np.sqrt(signal_acc.size / noise_acc.size)

    # Apply smoothing
    smooth_signal_matrix = load_konno_matrix(ko_matrices, ft_freq_signal)

    # Smooth ft with konno ohmachi matrix
    smooth_ft_signal = np.dot(np.abs(ft_signal).astype(np.float32), smooth_signal_matrix).astype(np.float64)
    smooth_ft_pe = np.dot(np.abs(ft_pe).astype(np.float32), smooth_signal_matrix).astype(np.float64)

    # Calculate SNR
    snr = smooth_ft_signal / smooth_ft_pe
    lower_index, upper_index = get_freq_ix(ft_freq_signal, 0.1, 20)
    snr_min = np.round(np.min(snr[lower_index:upper_index]), 5)

    return FourierData(
        None,
        None,
        ft_signal,
        ft_freq_signal,
        ft_pe,
        ft_freq_pe,
        None,
        smooth_ft_signal,
        smooth_ft_pe,
        snr,
        snr_min,
    )


def compute_pga(acc: np.ndarray):
    """Computes PGA for the given acceleration time series"""
    return np.max(np.abs(acc))


def compute_pe_max_amp(acc: np.ndarray, p_wave_ix: int):
    """Computes the pre-event max amplitude"""
    return np.max(np.abs(acc[0:p_wave_ix]))


def compute_avg_pn(acc: np.ndarray, p_wave_ix: int):
    """Computes the arthmetic pre-event noise average"""
    return np.average(np.abs(acc[0:p_wave_ix]))


def compute_tail_avg(acc: np.ndarray, tail_length: int):
    """Computes the tail average"""
    return np.mean(np.abs(acc[-tail_length:]))


def compute_tail_max(acc: np.ndarray, tail_length: int):
    """Computes the maximum tail amplitude"""
    return np.max(np.abs(acc[-tail_length:]))


def compute_head_avg(acc: np.ndarray, head_length: int):
    """Computes the head average"""
    return np.max(np.abs(acc[0:head_length]))


def compute_bracketed_pga_dur(
    acc: np.ndarray, pga: float, pga_factor: float, dt: float
):
    """Computes the bracketed PGA duraction for the specified PGA factor"""
    pga_excd_ix = np.flatnonzero(np.abs(acc) >= (pga_factor * pga))
    return (np.max(pga_excd_ix) - np.min(pga_excd_ix)) * dt


def compute_sig_duration(husid_ix_low: int, husid_ix_high: int, dt: float):
    """Computes the significant duration"""
    return (husid_ix_high - husid_ix_low) * dt


def compute_snr_min(
    snr: np.ndarray, ft_freq: np.ndarray, lower_freq: float, upper_freq: float
):
    """Computes SNR min over the specified frequency range"""
    lower_ix, upper_ix = get_freq_ix(ft_freq, lower_freq, upper_freq)
    return np.min(snr[lower_ix:upper_ix])


def compute_snr_max(
    snr: np.ndarray, ft_freq: np.ndarray, lower_freq: float, upper_freq: float
):
    """Computes SNR max over the specified frequency range"""
    lower_ix, upper_ix = get_freq_ix(ft_freq, lower_freq, upper_freq)
    return np.max(snr[lower_ix:upper_ix])


def compute_snr_avg(
    snr: np.ndarray, ft_freq: np.ndarray, lower_freq: float, upper_freq: float
):
    """Computes the SNR average"""
    lower_ix, upper_ix = get_freq_ix(ft_freq, lower_freq, upper_freq)
    return np.trapz(snr[lower_ix:upper_ix], ft_freq[lower_ix:upper_ix]) / (
        ft_freq[upper_ix] - ft_freq[lower_ix]
    )


# def compute_snr(
#     snr: np.ndarray, ft_freq: np.ndarray, lower_freq: float, upper_freq: float
# ):
#     """Computes the SNR for the specified frequency range"""
#     lower_ix, upper_ix = get_freq_ix(ft_freq, lower_freq, upper_freq)
#
#     # Can occur in certain cases (20161113_131547_NNZ_10.V1A)
#     # TODO: This is silly.., should just interpolate all the time?
#     if lower_ix == upper_ix:
#         return snr[lower_ix]
#
#     return np.trapz(snr[lower_ix:upper_ix], ft_freq[lower_ix:upper_ix]) / (
#         ft_freq[upper_ix] - ft_freq[lower_ix]
#     )


def compute_fas(
    ft_smooth: np.ndarray, ft_freq: np.ndarray, lower_freq: float, upper_freq: float
):
    """Computes the Fourier amplitude spectra (FAS) for the specified frequency range"""
    lower_ix, upper_ix = get_freq_ix(ft_freq, lower_freq, upper_freq)

    # Can occur in certain cases (20161113_131547_NNZ_10.V1A)
    # TODO: This is silly.., should just interpolate all the time?
    if lower_ix == upper_ix:
        return ft_smooth[lower_ix], None

    fas = np.trapz(ft_smooth[lower_ix:upper_ix], ft_freq[lower_ix:upper_ix],) / (
        ft_freq[upper_ix] - ft_freq[lower_ix]
    )
    ft_s = (ft_smooth[upper_ix] - ft_smooth[lower_ix]) / (
        ft_freq[upper_ix] / ft_freq[lower_ix]
    )

    return fas, ft_s


def compute_low_freq_fas(
    smooth_ft: np.ndarray, ft_freq: np.ndarray, low_freq: float
) -> float:
    """Computes the low frequency FAS"""
    low_ix = min(np.flatnonzero(ft_freq > low_freq))
    return np.trapz(smooth_ft[1:low_ix], ft_freq[1:low_ix]) / (
        ft_freq[low_ix] - ft_freq[1]
    )


def log_interpolate(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
    fill_value: Tuple[float, float] = (0, 0),
):
    """Performs log interpolation"""
    ln_x, ln_y, ln_x_new = np.log(x), np.log(y), np.log(x_new)
    return np.exp(
        interp1d(ln_x, ln_y, kind="linear", bounds_error=False, fill_value=fill_value)(
            ln_x_new
        )
    )


def compute_channel_features(
    acc: np.ndarray,
    t: np.ndarray,
    dt: float,
    p_wave_ix: int,
    s_wave_ix: int,
    prob_series: Tuple[np.ndarray, np.ndarray],
    ko_matrices: Dict[int, np.ndarray] = None,
):
    """
    Computes the features for the acceleration time-series

    Parameters
    ----------
    acc: array of floats
    t: array of floats
    dt: float
    p_wave_ix: int
    s_wave_ix: int
    prob_series: pair of array of floats
        The probability series from PhaseNet
        (p_wave_prob_series, s_wave_prob_series)
    ko_matrices: dictionary

    Returns
    -------

    """
    sample_rate = 1.0 / dt

    husid, AI, arias, husid_5_ix, husid_75_ix, husid_95_ix = compute_husid(acc, t)

    # Calculate max amplitude of acc time serie
    pga = compute_pga(acc)
    if np.isclose(pga, 0):
        raise FeatureError(
            None, FeatureErrorType.PGA_zero,
        )

    # Calculate pre-event max amplitude (noise)
    max_pn = compute_pe_max_amp(acc, p_wave_ix)

    # Compute Peak Noise to Peak Ground Acceleration Ratio
    pn_pga_ratio = max_pn / pga

    # Compute average tail ratio
    tail_duration = min(5.0, 0.1 * t[-1])
    tail_length = int(tail_duration * sample_rate)
    avg_tail_ratio = compute_tail_avg(acc, tail_length) / pga

    # Compute Maximum Tail Ratio
    max_tail_duration = min(2.0, 0.1 * t[-1])
    max_tail_length = int(max_tail_duration * sample_rate)
    max_tail_ratio_1 = compute_tail_max(acc, max_tail_length) / pga

    # Compute Maximum Head Ratio
    head_duration = 1.0
    head_length = int(head_duration * sample_rate)
    max_head_ratio_1 = compute_head_avg(acc, head_length) / pga

    # Calculate Ds575 and Ds595
    ds_575 = compute_sig_duration(husid_5_ix, husid_75_ix, dt)
    ds_595 = compute_sig_duration(husid_5_ix, husid_95_ix, dt)

    # Compute the fourier transform
    ft_data = comp_fourier_data(np.copy(acc), t, dt, p_wave_ix, ko_matrices)
    ft_freq_signal = ft_data.ft_freq_signal

    # SNR metrics - min, max and averages
    snr = ft_data.smooth_ft_signal / ft_data.smooth_ft_pe
    snr_min = compute_snr_min(snr, ft_freq_signal, 0.1, 20)
    snr_max = compute_snr_max(snr, ft_freq_signal, 0.1, 20)
    snr_avg = compute_snr_avg(snr, ft_freq_signal, 0.1, 20)

    # Compute SNR for a range of different frequency values
    snr_freq = np.logspace(np.log(0.01), np.log(25), 100, base=np.e)
    snr_values = log_interpolate(ft_freq_signal + 1e-17, snr, snr_freq)

    # Compute the Fourier amplitude ratio
    fas_0p1_0p2, ft_s1 = compute_fas(ft_data.smooth_ft_signal, ft_freq_signal, 0.1, 0.2)
    fas_0p2_0p5, ft_s2 = compute_fas(ft_data.smooth_ft_signal, ft_freq_signal, 0.2, 0.5)
    fas_0p5_1p0, ft_s3 = compute_fas(ft_data.smooth_ft_signal, ft_freq_signal, 0.5, 1.0)
    fas_ratio_low = fas_0p1_0p2 / fas_0p2_0p5
    fas_ratio_high = fas_0p2_0p5 / fas_0p5_1p0

    # Compute low frequency (both event & pre-event) FAS to maximum signal FAS ratio
    fas_max = np.max(ft_data.smooth_ft_signal)
    lf_fas = compute_low_freq_fas(ft_data.smooth_ft_signal, ft_freq_signal, 0.1)
    lf_pe_fas = compute_low_freq_fas(ft_data.smooth_ft_pe, ft_freq_signal, 0.1)

    signal_ratio_max = lf_fas / fas_max
    signal_pe_ratio_max = lf_pe_fas / fas_max

    p_wave_prob_series, s_wave_prob_series = prob_series

    features_dict = {
        "pn_pga_ratio": pn_pga_ratio,
        "average_tail_ratio": avg_tail_ratio,
        "max_tail_ratio": max_tail_ratio_1,
        "max_head_ratio": max_head_ratio_1,
        "ds_575": ds_575,
        "ds_595": ds_595,
        "signal_pe_ratio_max": signal_pe_ratio_max,
        "signal_ratio_max": signal_ratio_max,
        "fas_ratio_low": fas_ratio_low,
        "fas_ratio_high": fas_ratio_high,
        "snr_min": snr_min,
        "snr_max": snr_max,
        "snr_average": snr_avg,
        "snr_average_0.1_0.2": snr_values[0],
        "snr_average_0.2_0.5": snr_values[1],
        "snr_average_0.5_1.0": snr_values[2],
        "snr_average_1.0_2.0": snr_values[3],
        "snr_average_2.0_5.0": snr_values[4],
        "snr_average_5.0_10.0": snr_values[5],
        # Malfunction record features
        "spike_detector": mal_features.spike_detector(acc),
        "jerk_detector": mal_features.jerk_detector(acc),
        "lowres_detector": mal_features.lowres_detector(acc),
        "gainjump_detector": mal_features.gainjump_detector(acc),
        "flatline_detector": mal_features.flatline_detector(acc),
        # Multiple earthquake record features
        "p_numpeaks_detector": multi_eq_features.numpeaks_detector(p_wave_prob_series),
        "p_multimax_detector": multi_eq_features.multimax_detector(p_wave_prob_series),
        "p_multidist_detector": multi_eq_features.multidist_detector(
            p_wave_prob_series
        ),
        "s_numpeaks_detector": multi_eq_features.numpeaks_detector(s_wave_prob_series),
        "s_multimax_detector": multi_eq_features.multimax_detector(s_wave_prob_series),
        "s_multidist_detector": multi_eq_features.multidist_detector(
            s_wave_prob_series
        ),
    }
    features_dict = {
        **features_dict,
        **{f"snr_value_{freq:.3f}": val for freq, val in zip(snr_freq, snr_values)},
    }
    return features_dict


def get_features(
    record: Record, ko_matrices: Dict[int, np.ndarray] = None, phase_row: pd.Series = None, prob_series_ffp: str = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Create the time vector
    t = np.arange(record.size) * record.dt

    if phase_row is not None and not phase_row.empty:
        p_wave_ix, s_wave_ix, = phase_row["p_wave_ix"].values[0], phase_row["s_wave_ix"].values[0]
        with h5py.File(prob_series_ffp, 'r') as f:
            try:
                p_prob_series = f[record.id]["p_prob_series"][:]
                s_prob_series = f[record.id]["s_prob_series"][:]
                return p_prob_series, s_prob_series
            except KeyError:
                raise KeyError(f"Record ID {record.id} not found in the file.")
    else:
        p_wave_ix, s_wave_ix, p_prob_series, s_prob_series = run_phase_net(
            np.stack((record.acc_1, record.acc_2, record.acc_v), axis=1)[np.newaxis, ...],
            record.dt,
            t,
            return_prob_series=True,
        )

    if (record.size - p_wave_ix) * record.dt <= 10.24:
        raise FeatureError(
            "Signal duration is less than 10.24 seconds,"
            "preventing accurate feature generation",
            FeatureErrorType.short_signal_duration,
        )

    if p_wave_ix * record.dt < 2.5:
        raise FeatureError(
            "P-wave pick is less than 2.5s from start of record,"
            "preventing accurate feature generation",
            FeatureErrorType.early_p_pick,
        )

    # Compute the ratio (t_swave - t_pwave) / (t_end - t_swave)
    # For detecting records truncated to early
    s_wave_ratio = ((s_wave_ix * record.dt) - (p_wave_ix * record.dt)) / (
        t[-1] - (s_wave_ix * record.dt)
    )

    features_dict = {}
    for cur_key, cur_acc in zip(
        ["1", "2", "v"], [record.acc_1, record.acc_2, record.acc_v]
    ):
        try:
            features_dict[cur_key] = compute_channel_features(
                cur_acc,
                t,
                record.dt,
                p_wave_ix,
                s_wave_ix,
                (p_prob_series, s_prob_series),
                ko_matrices=ko_matrices,
            )
            features_dict[cur_key]["is_vertical"] = 1 if cur_key == "v" else 0

            # HACK: Fix this in the future..
            if cur_key == "1":
                features_dict[cur_key]["s_wave_ratio"] = s_wave_ratio
        except FeatureError as ex:
            if ex.error_type is FeatureErrorType.PGA_zero:
                raise FeatureError(
                    f"Record {record.id} - PGA is zero"
                    f" for one (or more) of the components",
                    FeatureErrorType.PGA_zero,
                )
            else:
                raise ex

    additional_data = {
        "p_wave_ix": p_wave_ix,
        "s_wave_ix": s_wave_ix,
    }

    return features_dict, additional_data
