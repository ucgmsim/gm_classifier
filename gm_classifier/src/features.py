"""Contains code for feature/metric extraction from GeoNet ground motion records.
Based on existing implemention
from Xavier Bellagamba (https://github.com/xavierbellagamba/GroundMotionRecordClassifier)
"""
import gc
import os
import math
from enum import Enum
from typing import Tuple, Dict, Union, Any
from collections import namedtuple
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from obspy.signal.trigger import ar_pick
from obspy.signal.konnoohmachismoothing import calculate_smoothing_matrix

from . import GeoNet_File


class FeatureErrorType(Enum):
    # PGA is zero
    PGA_zero = 1


class FeatureError(Exception):
    def __init__(self, message: str, error_type: FeatureErrorType):
        super(Exception, self).__init__(message)

        self.error_type = error_type


KONNO_MATRIX_FILENAME_TEMPLATE = "konno_{}.npy"

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
    ft_freq = np.arange(0, ft_len / 2 + 1) * (1.0 / (ft_len * dt))
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
    if isinstance(ko_matrices, dict):
        smooth_matrix = ko_matrices[ft_freq.size - 1]
    elif isinstance(ko_matrices, str) and os.path.isdir(ko_matrices):
        matrix_name = KONNO_MATRIX_FILENAME_TEMPLATE.format(ft_freq.size - 1)
        print(f"Loading Konno matrix {matrix_name}")
        smooth_matrix = np.load(
            os.path.join(
                ko_matrices, matrix_name
            )
        )
    else:
        raise ValueError(
            "The ko_matrices parameter has to either be a "
            "dictionary with the Konno matrices or a directory "
            "path that contains the Konno matrices files."
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

    # Apply smoothing, might require two different smoothing matrices due to different number of frequencies
    # smooth_matrix = load_konno_matrix(ko_matrices, ft_freq)
    # smooth_ft = np.dot(np.abs(ft), smooth_matrix)

    # if ft_freq.size == ft_freq_signal.size:
    #     smooth_signal_matrix = smooth_matrix
    # else:
    # del smooth_matrix
    smooth_signal_matrix = load_konno_matrix(ko_matrices, ft_freq_signal)

    # Smooth ft with konno ohmachi matrix
    smooth_ft_signal = np.dot(np.abs(ft_signal), smooth_signal_matrix)
    smooth_ft_pe = np.dot(np.abs(ft_pe), smooth_signal_matrix)

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


def compute_snr(
    snr: np.ndarray, ft_freq: np.ndarray, lower_freq: float, upper_freq: float
):
    """Computes the SNR for the specified frequency range"""
    lower_ix, upper_ix = get_freq_ix(ft_freq, lower_freq, upper_freq)
    return np.trapz(snr[lower_ix:upper_ix], ft_freq[lower_ix:upper_ix]) / (
        ft_freq[upper_ix] - ft_freq[lower_ix]
    )


def compute_fas(
    ft_smooth: np.ndarray, ft_freq: np.ndarray, lower_freq: float, upper_freq: float
):
    """Computes the Fourier amplitude spectra (FAS) for the specified frequency range
    Not sure, what ft_s actually is?
    """
    lower_index, upper_index_average = get_freq_ix(ft_freq, lower_freq, upper_freq)
    fas = np.trapz(
        ft_smooth[lower_index:upper_index_average],
        ft_freq[lower_index:upper_index_average],
    ) / (ft_freq[upper_index_average] - ft_freq[lower_index])
    ft_s = (ft_smooth[upper_index_average] - ft_smooth[lower_index]) / (
        ft_freq[upper_index_average] / ft_freq[lower_index]
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


def log_interpolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray):
    """Performs log interpolation"""
    ln_x, ln_y, ln_x_new = np.log(x), np.log(y), np.log(x_new)
    return np.exp(interp1d(ln_x, ln_y, kind="linear", bounds_error=True)(ln_x_new))


def get_p_wave_ix(acc_X, acc_Y, acc_Z, dt):
    sample_rate = 1.0 / dt

    p_pick, s_pick = ar_pick(a = acc_Z,
                b = acc_X,
                c = acc_Y,
                samp_rate=sample_rate,  # sample_rate
                f1=dt*20.0,  # low_pass
                f2=sample_rate/20.0,  # high_pass
                lta_p=1.0,  # P-LTA
                sta_p=0.2,  # P-STA,
                lta_s=2.0,  # S-LTA
                sta_s=0.4,  # S-STA
                m_p=8,  # P-AR coefficients
                m_s=8,  # S-coefficients
                l_p=0.4,  # P-length
                l_s=0.2,  # S-length
                s_pick=True,  # S-pick
            )

    p_wave_ix = int(np.floor(np.multiply(p_pick, sample_rate)))
    return p_wave_ix

# def get_p_wave_ix(acc_X, acc_Y, acc_Z, dt):
#     # Set up some modified time series for p- and s-wave picking.
#     # These are multiplied by an additional 10 because it seems to make the P-wave picking better
#     # Also better if the vertical component is sign squared (apparently better)
#     tr1 = acc_X * 9806.7 * 10.0
#     tr2 = acc_Y * 9806.7 * 10.0
#     tr3 = np.multiply(np.abs(acc_Z), acc_Z) * np.power(9806.7 * 10.0, 2)
#
#     sample_rate = 1.0 / dt
#
#     low_pass = dt * 20.0
#     high_pass = sample_rate / 20.0
#
#     p_s_pick = partial(
#         ar_pick,
#         samp_rate=sample_rate,  # sample_rate
#         f1=low_pass,  # low_pass
#         f2=high_pass,  # high_pass
#         lta_p=1.0,  # P-LTA
#         sta_p=0.2,  # P-STA,
#         lta_s=2.0,  # S-LTA
#         sta_s=0.4,  # S-STA
#         m_p=8,  # P-AR coefficients
#         m_s=8,  # S-coefficients
#         l_p=0.4,  # P-length
#         l_s=0.2,  # S-length
#         s_pick=True,  # S-pick
#     )
#
#     # Get p-wave arrival and s-wave arrival picks
#     p_pick, s_pick = p_s_pick(tr3, tr1, tr2)
#
#     if p_pick < 5.0:
#         # NEXT THING TO DO IS TO TEST CHANGING THE PARAMETERS FOR THE FAKE PICK
#         tr3_fake1 = np.multiply(np.abs(acc_X), acc_Z) * np.power(9806.7 * 10.0, 2)
#         p_pick_fake1, s_pick_fake1 = p_s_pick(tr3_fake1, tr1, tr2)
#
#         tr3_fake2 = np.multiply(np.abs(acc_Y), acc_Z) * np.power(9806.7 * 10.0, 2)
#         p_pick_fake2, s_pick_fake2 = p_s_pick(tr3_fake2, tr1, tr2)
#
#         p_pick = np.max([p_pick_fake1, p_pick_fake2, p_pick])
#         s_pick = np.max([s_pick_fake1, s_pick_fake2, s_pick])
#     p_wave_ix = int(np.floor(np.multiply(p_pick, sample_rate)))
#     return p_wave_ix


def get_features(
    gf: GeoNet_File, ko_matrices: Dict[int, np.ndarray] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    # Create the time vector
    t = np.arange(gf.comp_1st.acc.size) * gf.comp_1st.delta_t

    # Get acceleration time series
    acc_1, acc_2 = gf.comp_1st.acc, gf.comp_2nd.acc
    acc_v = gf.comp_up.acc

    # Compute husid and Arias intensities
    husid_1, AI_1, arias_1, husid_5_ix_1, husid_75_ix_1, husid_95_ix_1 = compute_husid(
        acc_1, t
    )
    husid_2, AI_2, arias_2, husid_5_ix_2, husid_75_ix_2, husid_95_ix_2 = compute_husid(
        acc_2, t
    )
    husid_v, AI_v, arias_v, husid_5_ix_v, husid_75_ix_v, husid_95_ix_v = compute_husid(
        acc_v, t
    )
    arias_gm = np.sqrt(arias_1 * arias_2)

    assert np.isclose(gf.comp_1st.delta_t, gf.comp_2nd.delta_t) and np.isclose(
        gf.comp_1st.delta_t, gf.comp_up.delta_t
    )

    sample_rate = 1.0 / gf.comp_1st.delta_t
    p_wave_ix = get_p_wave_ix(acc_1, acc_2, acc_v, gf.comp_1st.delta_t)

    # Calculate max amplitudes of acc time series
    pga_1, pga_2 = compute_pga(acc_1), compute_pga(acc_2)
    pga_v = compute_pga(acc_v)
    pga_gm = np.sqrt(pga_1 * pga_2)

    if np.isclose(pga_1, 0) or np.isclose(pga_2, 0) or np.isclose(pga_v, 0):
        raise FeatureError(
            f"Record {os.path.basename(gf.record_ffp)} - PGA is zero"
            f" for one (or more) of the components",
            FeatureErrorType.PGA_zero,
        )

    # Calculate pre-event max amplitude (noise)
    max_pn_1 = compute_pe_max_amp(acc_1, p_wave_ix)
    max_pn_2 = compute_pe_max_amp(acc_2, p_wave_ix)
    max_pn_v = compute_pe_max_amp(acc_v, p_wave_ix)
    max_pn_gm = np.sqrt(max_pn_1 * max_pn_2)

    # Compute Peak Noise to Peak Ground Acceleration Ratio
    pn_pga_ratio_1 = max_pn_1 / pga_1
    pn_pga_ratio_2 = max_pn_2 / pga_2
    pn_pga_ratio_v = max_pn_v / pga_v
    pn_pga_ratio_gm = max_pn_gm / pga_gm

    # Compute average tail ratio
    tail_duration = min(5.0, 0.1 * t[-1])
    tail_length = int(tail_duration * sample_rate)

    avg_tail_ratio_1 = compute_tail_avg(acc_1, tail_length) / pga_1
    avg_tail_ratio_2 = compute_tail_avg(acc_2, tail_length) / pga_2
    avg_tail_ratio_v = compute_tail_avg(acc_v, tail_length) / pga_v
    tail_ratio_avg_gm = np.sqrt(avg_tail_ratio_1 * avg_tail_ratio_2)

    # Compute Maximum Tail Ratio
    max_tail_duration = min(2.0, 0.1 * t[-1])
    max_tail_length = int(max_tail_duration * sample_rate)

    max_tail_ratio_1 = compute_tail_max(acc_1, max_tail_length) / pga_1
    max_tail_ratio_2 = compute_tail_max(acc_2, max_tail_length) / pga_2
    max_tail_ratio_v = compute_tail_max(acc_v, max_tail_length) / pga_v
    max_tail_ratio_gm = np.sqrt(max_tail_ratio_1 * max_tail_ratio_2)

    # Compute Maximum Head Ratio
    head_duration = 1.0
    head_length = int(head_duration * sample_rate)

    max_head_ratio_1 = compute_head_avg(acc_1, head_length) / pga_1
    max_head_ratio_2 = compute_head_avg(acc_2, head_length) / pga_2
    max_head_ratio_v = compute_head_avg(acc_v, head_length) / pga_v
    max_head_ratio_gm = np.sqrt(max_head_ratio_1 * max_head_ratio_2)

    # Calculate Ds575 and Ds595
    ds_575_1 = compute_sig_duration(husid_5_ix_1, husid_75_ix_1, gf.comp_1st.delta_t)
    ds_595_1 = compute_sig_duration(husid_5_ix_1, husid_95_ix_1, gf.comp_1st.delta_t)

    ds_575_2 = compute_sig_duration(husid_5_ix_2, husid_75_ix_2, gf.comp_2nd.delta_t)
    ds_595_2 = compute_sig_duration(husid_5_ix_2, husid_95_ix_2, gf.comp_2nd.delta_t)

    ds_575_v = compute_sig_duration(husid_5_ix_v, husid_75_ix_v, gf.comp_up.delta_t)
    ds_595_v = compute_sig_duration(husid_5_ix_v, husid_95_ix_v, gf.comp_up.delta_t)

    ds_575_gm = np.sqrt(ds_575_1 * ds_575_2)
    ds_595_gm = np.sqrt(ds_595_1 * ds_595_2)

    # Compute the fourier transform
    ft_data_1 = comp_fourier_data(
        np.copy(acc_1), t, gf.comp_1st.delta_t, p_wave_ix, ko_matrices
    )
    ft_data_2 = comp_fourier_data(
        np.copy(acc_2), t, gf.comp_2nd.delta_t, p_wave_ix, ko_matrices
    )
    ft_data_v = comp_fourier_data(
        np.copy(acc_v), t, gf.comp_up.delta_t, p_wave_ix, ko_matrices
    )

    # Compute geomean of fourier spectra
    ft_gm = np.sqrt(np.abs(ft_data_1.ft_signal) * np.abs(ft_data_2.ft_signal))
    ft_pe_gm = np.sqrt(np.abs(ft_data_1.ft_pe) * np.abs(ft_data_2.ft_pe))

    smooth_matrix = load_konno_matrix(ko_matrices, ft_data_1.ft_freq_signal)
    ft_smooth_gm = np.dot(ft_gm, smooth_matrix)
    ft_smooth_pe_gm = np.dot(ft_pe_gm, smooth_matrix)

    # Same for all components
    ft_freq_signal = ft_data_1.ft_freq_signal
    assert np.all(np.isclose(ft_data_1.ft_freq_signal, ft_data_2.ft_freq_signal)) and np.all(
        np.isclose(ft_data_1.ft_freq_signal, ft_data_v.ft_freq_signal)
    )

    # SNR metrics - min, max and averages
    snr_1 = ft_data_1.smooth_ft_signal / ft_data_1.smooth_ft_pe
    snr_min_1 = compute_snr_min(snr_1, ft_freq_signal, 0.1, 20)
    snr_max_1 = compute_snr_max(snr_1, ft_freq_signal, 0.1, 20)
    snr_avg_1 = compute_snr_avg(snr_1, ft_freq_signal, 0.1, 20)

    snr_2 = ft_data_2.smooth_ft_signal / ft_data_2.smooth_ft_pe
    snr_min_2 = compute_snr_min(snr_2, ft_freq_signal, 0.1, 20)
    snr_max_2 = compute_snr_max(snr_2, ft_freq_signal, 0.1, 20)
    snr_avg_2 = compute_snr_avg(snr_2, ft_freq_signal, 0.1, 20)

    snr_v = ft_data_v.smooth_ft_signal / ft_data_v.smooth_ft_pe
    snr_min_v = compute_snr_min(snr_v, ft_freq_signal, 0.1, 20)
    snr_max_v = compute_snr_max(snr_v, ft_freq_signal, 0.1, 20)
    snr_avg_v = compute_snr_avg(snr_2, ft_freq_signal, 0.1, 10)

    snr_gm = np.divide(ft_smooth_gm, ft_smooth_pe_gm)
    snr_min = compute_snr_min(snr_gm, ft_freq_signal, 0.1, 20)
    snr_max = compute_snr_max(snr_gm, ft_freq_signal, 0.1, 20)
    snr_avg_gm = compute_snr_avg(snr_gm, ft_freq_signal, 0.1, 20)

    # Computing average SNR for the different frequency ranges
    snr_freq_bins = [
        (0.1, 0.2),
        (0.2, 0.5),
        (0.5, 1.0),
        (1.0, 2.0),
        (2.0, 5.0),
        (5.0, 10.0),
    ]
    snr_values_1, snr_values_2 = [], []
    snr_values_v, snr_values_gm = [], []
    for lower_freq, upper_freq in snr_freq_bins:
        snr_values_1.append(compute_snr(snr_1, ft_freq_signal, lower_freq, upper_freq))
        snr_values_2.append(compute_snr(snr_2, ft_freq_signal, lower_freq, upper_freq))
        snr_values_v.append(compute_snr(snr_v, ft_freq_signal, lower_freq, upper_freq))
        snr_values_gm.append(compute_snr(snr_gm, ft_freq_signal, lower_freq, upper_freq))

    # Compute SNR for a range of different frequency values
    snr_freq = np.logspace(np.log(0.01), np.log(25), 100, base=np.e)
    snr_values_1 = log_interpolate(ft_freq_signal + 1e-17, snr_1, snr_freq)
    snr_values_2 = log_interpolate(ft_freq_signal + 1e-17, snr_2, snr_freq)
    snr_values_v = log_interpolate(ft_freq_signal + 1e-17, snr_v, snr_freq)

    # Compute the Fourier amplitude ratio
    fas_0p1_0p2_1, ft_s1_1 = compute_fas(ft_data_1.smooth_ft_signal, ft_freq_signal, 0.1, 0.2)
    fas_0p1_0p2_2, ft_s1_2 = compute_fas(ft_data_2.smooth_ft_signal, ft_freq_signal, 0.1, 0.2)
    fas_0p1_0p2_v, ft_s1_v = compute_fas(ft_data_v.smooth_ft_signal, ft_freq_signal, 0.1, 0.2)
    fas_0p1_0p2_gm, ft_s1_gm = compute_fas(ft_smooth_gm, ft_freq_signal, 0.1, 0.2)

    fas_0p2_0p5_1, ft_s2_1 = compute_fas(ft_data_1.smooth_ft_signal, ft_freq_signal, 0.2, 0.5)
    fas_0p2_0p5_2, ft_s2_2 = compute_fas(ft_data_2.smooth_ft_signal, ft_freq_signal, 0.2, 0.5)
    fas_0p2_0p5_v, ft_s2_v = compute_fas(ft_data_v.smooth_ft_signal, ft_freq_signal, 0.2, 0.5)
    fas_0p2_0p5_gm, ft_s2_gm = compute_fas(ft_smooth_gm, ft_freq_signal, 0.2, 0.5)

    fas_0p5_1p0_1, ft_s3_1 = compute_fas(ft_data_1.smooth_ft_signal, ft_freq_signal, 0.5, 1.0)
    fas_0p5_1p0_2, ft_s3_2 = compute_fas(ft_data_2.smooth_ft_signal, ft_freq_signal, 0.5, 1.0)
    fas_0p5_1p0_v, ft_s3_v = compute_fas(ft_data_v.smooth_ft_signal, ft_freq_signal, 0.5, 1.0)
    fas_0p5_1p0_gm, ft_s3_gm = compute_fas(ft_smooth_gm, ft_freq_signal, 0.5, 1.0)

    fas_ratio_low_1 = fas_0p1_0p2_1 / fas_0p2_0p5_1
    fas_ratio_low_2 = fas_0p1_0p2_2 / fas_0p2_0p5_2
    fas_ratio_low_v = fas_0p1_0p2_v / fas_0p2_0p5_v
    fas_ratio_low_gm = fas_0p1_0p2_gm / fas_0p2_0p5_gm

    fas_ratio_high_1 = fas_0p2_0p5_1 / fas_0p5_1p0_1
    fas_ratio_high_2 = fas_0p2_0p5_2 / fas_0p5_1p0_2
    fas_ratio_high_v = fas_0p2_0p5_v / fas_0p5_1p0_v
    fas_ratio_high_gm = fas_0p2_0p5_gm / fas_0p5_1p0_gm

    # What is this?
    ft_s1_s2_1 = ft_s1_1 / ft_s2_1
    ft_s1_s2_2 = ft_s1_2 / ft_s2_2
    ft_s1_s2_v = ft_s1_v / ft_s2_v
    ft_s1_s2_gm = ft_s1_gm / ft_s2_gm

    ft_s2_s3_1 = ft_s1_1 / ft_s2_1
    ft_s2_s3_2 = ft_s1_2 / ft_s2_2
    ft_s2_s3_v = ft_s1_v / ft_s2_v
    ft_s2_s3_gm = ft_s1_gm / ft_s2_gm

    # Compute low frequency (both event & pre-event) FAS to maximum signal FAS ratio
    fas_max_1 = np.max(ft_data_1.smooth_ft_signal)
    lf_fas_1 = compute_low_freq_fas(ft_data_1.smooth_ft_signal, ft_freq_signal, 0.1)
    lf_pe_fas_1 = compute_low_freq_fas(ft_data_1.smooth_ft_pe, ft_freq_signal, 0.1)

    fas_max_2 = np.max(ft_data_2.smooth_ft_signal)
    lf_fas_2 = compute_low_freq_fas(ft_data_2.smooth_ft_signal, ft_freq_signal, 0.1)
    lf_pe_fas_2 = compute_low_freq_fas(ft_data_2.smooth_ft_pe, ft_freq_signal, 0.1)

    fas_max_v = np.max(ft_data_v.smooth_ft_signal)
    lf_fas_v = compute_low_freq_fas(ft_data_v.smooth_ft_signal, ft_freq_signal, 0.1)
    lf_pe_fas_v = compute_low_freq_fas(ft_data_v.smooth_ft_pe, ft_freq_signal, 0.1)

    signal_ratio_max_1 = lf_fas_1 / fas_max_1
    signal_ratio_max_2 = lf_fas_2 / fas_max_2
    signal_ratio_max_v = lf_fas_v / fas_max_v
    signal_ratio_max = max([lf_fas_1 / fas_max_1, lf_fas_2 / fas_max_2])

    signal_pe_ratio_max_1 = lf_pe_fas_1 / fas_max_1
    signal_pe_ratio_max_2 = lf_pe_fas_2 / fas_max_2
    signal_pe_ratio_max_v = lf_pe_fas_v / fas_max_v
    signal_pe_ratio_max = max([lf_pe_fas_1 / fas_max_1, lf_pe_fas_2 / fas_max_2])

    # TODO: Do this better at some point...
    features_dict = {
        "1": {
            "pn_pga_ratio": pn_pga_ratio_1,
            "average_tail_ratio": avg_tail_ratio_1,
            "max_tail_ratio": max_tail_ratio_1,
            "max_head_ratio": max_head_ratio_1,
            "ds_575": ds_575_1,
            "ds_595": ds_595_1,
            "signal_pe_ratio_max": signal_pe_ratio_max_1,
            "signal_ratio_max": signal_ratio_max_1,
            "fas_ratio_low": fas_ratio_low_1,
            "fas_ratio_high": fas_ratio_high_1,
            "snr_min": snr_min_1,
            "snr_max": snr_max_1,
            "snr_average": snr_avg_1,
            "snr_average_0.1_0.2": snr_values_1[0],
            "snr_average_0.2_0.5": snr_values_1[1],
            "snr_average_0.5_1.0": snr_values_1[2],
            "snr_average_1.0_2.0": snr_values_1[3],
            "snr_average_2.0_5.0": snr_values_1[4],
            "snr_average_5.0_10.0": snr_values_1[5],
            "is_vertical": 0,
        },
        "2": {
            "pn_pga_ratio": pn_pga_ratio_2,
            "average_tail_ratio": avg_tail_ratio_2,
            "max_tail_ratio": max_tail_ratio_2,
            "max_head_ratio": max_head_ratio_2,
            "ds_575": ds_575_2,
            "ds_595": ds_595_2,
            "signal_pe_ratio_max": signal_pe_ratio_max_2,
            "signal_ratio_max": signal_ratio_max_2,
            "fas_ratio_low": fas_ratio_low_2,
            "fas_ratio_high": fas_ratio_high_2,
            "snr_min": snr_min_2,
            "snr_max": snr_max_2,
            "snr_average": snr_avg_2,
            "snr_average_0.1_0.2": snr_values_2[0],
            "snr_average_0.2_0.5": snr_values_2[1],
            "snr_average_0.5_1.0": snr_values_2[2],
            "snr_average_1.0_2.0": snr_values_2[3],
            "snr_average_2.0_5.0": snr_values_2[4],
            "snr_average_5.0_10.0": snr_values_2[5],
            "is_vertical": 0,
        },
        "v": {
            "pn_pga_ratio": pn_pga_ratio_v,
            "average_tail_ratio": avg_tail_ratio_v,
            "max_tail_ratio": max_tail_ratio_v,
            "max_head_ratio": max_head_ratio_v,
            "ds_575": ds_575_v,
            "ds_595": ds_595_v,
            "signal_pe_ratio_max": signal_pe_ratio_max_v,
            "signal_ratio_max": signal_ratio_max_v,
            "fas_ratio_low": fas_ratio_low_v,
            "fas_ratio_high": fas_ratio_high_v,
            "snr_min": snr_min_v,
            "snr_max": snr_max_v,
            "snr_average": snr_avg_v,
            "snr_average_0.1_0.2": snr_values_v[0],
            "snr_average_0.2_0.5": snr_values_v[1],
            "snr_average_0.5_1.0": snr_values_v[2],
            "snr_average_1.0_2.0": snr_values_v[3],
            "snr_average_2.0_5.0": snr_values_v[4],
            "snr_average_5.0_10.0": snr_values_v[5],
            "is_vertical": 1,
        },
        "gm": {
            "pn_pga_ratio": pn_pga_ratio_gm,
            "average_tail_ratio": tail_ratio_avg_gm,
            "max_tail_ratio": max_tail_ratio_gm,
            "max_head_ratio": max_head_ratio_gm,
            "ds_575": ds_575_gm,
            "ds_595": ds_595_gm,
            "signal_pe_ratio_max": signal_pe_ratio_max,
            "signal_ratio_max": signal_ratio_max,
            "fas_ratio_low": fas_ratio_low_gm,
            "fas_ratio_high": fas_ratio_high_gm,
            "snr_min": snr_min,
            "snr_max": snr_max,
            "snr_average": snr_avg_gm,
            "snr_average_0.1_0.2": snr_values_gm[0],
            "snr_average_0.2_0.5": snr_values_gm[1],
            "snr_average_0.5_1.0": snr_values_gm[2],
            "snr_average_1.0_2.0": snr_values_gm[3],
            "snr_average_2.0_5.0": snr_values_gm[4],
            "snr_average_5.0_10.0": snr_values_gm[5],
        },
    }

    # Add the SNR values
    features_dict["1"] = {
        **features_dict["1"],
        **{f"snr_value_{freq:.3f}": val for freq, val in zip(snr_freq, snr_values_1)},
    }
    features_dict["2"] = {
        **features_dict["2"],
        **{f"snr_value_{freq:.3f}": val for freq, val in zip(snr_freq, snr_values_2)},
    }
    features_dict["v"] = {
        **features_dict["v"],
        **{f"snr_value_{freq:.3f}": val for freq, val in zip(snr_freq, snr_values_v)},
    }

    additional_data = {
        "p_wave_ix": p_wave_ix,
        "gm": {
            "fas_0p1_0p2": fas_0p1_0p2_gm,
            "fas_0p2_0p5": fas_0p2_0p5_gm,
            "fas_0p5_1p0": fas_0p5_1p0_gm,
            "ft_s1": ft_s1_gm,
            "ft_s2": ft_s2_gm,
            "ft_s3": ft_s3_gm,
            "ft_s1_s2": ft_s1_s2_gm,
            "ft_s2_s3": ft_s2_s3_gm,
            "pga": pga_gm,
            "pn": max_pn_gm,
            "arias": arias_gm,
        },
        "1": {
            "fas_0p1_0p2": fas_0p1_0p2_1,
            "fas_0p2_0p5": fas_0p2_0p5_1,
            "fas_0p5_1p0": fas_0p5_1p0_1,
            "ft_s1": ft_s1_1,
            "ft_s2": ft_s2_1,
            "ft_s3": ft_s3_1,
            "ft_s1_s2": ft_s1_s2_1,
            "ft_s2_s3": ft_s2_s3_1,
            "pga": pga_1,
            "pn": max_pn_1,
            "arias": arias_1,
        },
        "2": {
            "fas_0p1_0p2": fas_0p1_0p2_2,
            "fas_0p2_0p5": fas_0p2_0p5_2,
            "fas_0p5_1p0": fas_0p5_1p0_2,
            "ft_s1": ft_s1_2,
            "ft_s2": ft_s2_2,
            "ft_s3": ft_s3_2,
            "ft_s1_s2": ft_s1_s2_2,
            "ft_s2_s3": ft_s2_s3_2,
            "pga": pga_2,
            "pn": max_pn_2,
            "arias": arias_2,
        },
        "v": {
            "fas_0p1_0p2": fas_0p1_0p2_v,
            "fas_0p2_0p5": fas_0p2_0p5_v,
            "fas_0p5_1p0": fas_0p5_1p0_v,
            "ft_s1": ft_s1_v,
            "ft_s2": ft_s2_v,
            "ft_s3": ft_s3_v,
            "ft_s1_s2": ft_s1_s2_v,
            "ft_s2_s3": ft_s2_s3_v,
            "pga": pga_v,
            "pn": max_pn_v,
            "arias": arias_v,
        },
    }

    return features_dict, additional_data


def generate_plots():
    plt.figure(figsize=(21, 14), dpi=75)
    plt.suptitle(stat_code, fontsize=20)
    ax = plt.subplot(431)
    plt.plot(t, acc1, color="k")
    ymin, ymax = ax.get_ylim()
    plt.vlines(p_pick, ymin, ymax, color="r", linewidth=2)
    plt.vlines(s_pick, ymin, ymax, color="b", linewidth=2)
    #    plt.vlines(min(hindex1_10)*gf.comp_1st.delta_t, ymin, ymax, color='g', linewidth=2, linestyle='--')
    #    plt.vlines(max(hindex1_10)*gf.comp_1st.delta_t, ymin, ymax, color='g', linewidth=2, linestyle='--')
    #    plt.vlines(min(hindex1_20)*gf.comp_1st.delta_t, ymin, ymax, color='g', linewidth=2, linestyle='--')
    #    plt.vlines(max(hindex1_20)*gf.comp_1st.delta_t, ymin, ymax, color='g', linewidth=2, linestyle='--')
    #    plt.vlines(min(hindex1_30)*gf.comp_1st.delta_t, ymin, ymax, color='g', linewidth=2, linestyle='--')
    #    plt.vlines(max(hindex1_30)*gf.comp_1st.delta_t, ymin, ymax, color='g', linewidth=2, linestyle='--')
    #    plt.vlines(min(hindex1_40)*gf.comp_1st.delta_t, ymin, ymax, color='g', linewidth=2, linestyle='--')
    #    plt.vlines(max(hindex1_40)*gf.comp_1st.delta_t, ymin, ymax, color='g', linewidth=2, linestyle='--')
    #    plt.vlines(min(hindex1_50)*gf.comp_1st.delta_t, ymin, ymax, color='g', linewidth=2, linestyle='--')
    #    plt.vlines(max(hindex1_50)*gf.comp_1st.delta_t, ymin, ymax, color='g', linewidth=2, linestyle='--')
    #    plt.vlines(husid_index1_75*gf.comp_1st.delta_t, ymin, ymax, color='grey', linewidth=2, linestyle='--')
    #    plt.vlines(husid_index1_95*gf.comp_1st.delta_t, ymin, ymax, color='grey', linewidth=2, linestyle='--')
    #    plt.hlines([0.10*PGA1], 0, t[-1], color=['g'], linestyle='--')
    #    plt.hlines([-0.10*PGA1], 0, t[-1], color=['g'], linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (g)")
    ax.text(
        0.99,
        0.97,
        PGA1,
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.99,
        0.90,
        [round(p_pick, 3), round(s_pick, 3)],
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.99,
        0.03,
        [
            round(average_tail_ratio1, 5),
            round(max_tail_ratio1, 5),
            round(max_head_ratio1, 5),
        ],
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )
    ax.set_ylim([ymin, ymax])
    plt.title("Comp 1")
    ax = plt.subplot(432)
    plt.plot(t, acc2, color="k")
    ymin, ymax = ax.get_ylim()
    plt.vlines(p_pick, ymin, ymax, color="r", linewidth=2)
    plt.vlines(s_pick, ymin, ymax, color="b", linewidth=2)

    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (g)")
    ax.text(
        0.99,
        0.97,
        PGA2,
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.99,
        0.90,
        [round(p_pick, 3), round(s_pick, 3)],
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.99,
        0.03,
        [
            round(average_tail_ratio2, 5),
            round(max_tail_ratio2, 5),
            round(max_head_ratio2, 5),
        ],
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )
    ax.set_ylim([ymin, ymax])
    plt.title("Comp 2")
    ax = plt.subplot(433)
    plt.plot(t, accv, color="k")
    ymin, ymax = ax.get_ylim()
    plt.vlines(p_pick, ymin, ymax, color="r", linewidth=2)
    plt.vlines(s_pick, ymin, ymax, color="b", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (g)")
    ax.text(
        0.99,
        0.97,
        PGAv,
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.99,
        0.90,
        [round(p_pick, 3), round(s_pick, 3)],
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.99,
        0.03,
        [
            round(average_tail_ratiov, 5),
            round(max_tail_ratiov, 5),
            round(max_head_ratiov, 5),
        ],
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )
    plt.title("Comp Up")

    ax = plt.subplot(434)
    plt.plot(t, AI1, "k")
    plt.vlines(p_pick, 0, 1, color="r", linewidth=2)
    plt.vlines(
        husid_index1_75 * gf.comp_1st.delta_t,
        0,
        1,
        color="grey",
        linewidth=2,
        linestyle="--",
    )
    plt.vlines(
        husid_index1_95 * gf.comp_1st.delta_t,
        0,
        1,
        color="grey",
        linewidth=2,
        linestyle="--",
    )
    plt.hlines([0.05], 0, t[-1], color=["r"], linestyle="--")
    plt.hlines([0.75], 0, t[-1], color=["r"], linestyle="--")
    plt.hlines([0.95], 0, t[-1], color=["r"], linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized cumulative Arias Intensity")
    ax = plt.subplot(435)
    plt.plot(t, AI2, "k")
    plt.vlines(p_pick, 0, 1, color="r", linewidth=2)
    plt.vlines(
        husid_index2_75 * gf.comp_1st.delta_t,
        0,
        1,
        color="grey",
        linewidth=2,
        linestyle="--",
    )
    plt.vlines(
        husid_index2_95 * gf.comp_1st.delta_t,
        0,
        1,
        color="grey",
        linewidth=2,
        linestyle="--",
    )
    plt.hlines([0.05], 0, t[-1], color=["r"], linestyle="--")
    plt.hlines([0.75], 0, t[-1], color=["r"], linestyle="--")
    plt.hlines([0.95], 0, t[-1], color=["r"], linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized cumulative Arias Intensity")
    ax = plt.subplot(436)
    plt.plot(t, AIv, "k")
    plt.vlines(p_pick, 0, 1, color="r", linewidth=2)
    plt.hlines([0.05], 0, t[-1], color=["r"], linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized cumulative Arias Intensity")
    #    ax.text(0.99, 0.97, t_husid_threshold, horizontalalignment='right', verticalalignment='top', transform = ax.transAxes)

    ax = plt.subplot(437)
    plt.loglog(ft1_freq, np.abs(ft1), label="Full motion", color="blue")
    plt.loglog(ft1_freq_pe, np.abs(ft1_pe), label="Pre-event trace", color="red")
    plt.loglog(ft1_freq, smooth_ft1, label="Smoothed fm", color="green", linewidth=1.5)
    plt.loglog(
        ft1_freq, smooth_ft1_pe, label="Smoothed pe", color="grey", linewidth=1.5
    )
    ymin, ymax = ax.get_ylim()
    plt.vlines(0.1, ymin, ymax, color="r", linewidth=1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Fourier amplitude")
    plt.legend(loc=3, prop={"size": 10})
    ax.set_xlim([0.001, 100])
    #    ax.text(0.01, 0.97, snr1_min, horizontalalignment='left', verticalalignment='top', transform = ax.transAxes)
    ax.text(
        0.01,
        0.97,
        [round(lf1_pe / signal1_max, 5), round(lf1_pe, 4), round(signal1_max, 4)],
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.01,
        0.90,
        [round(lf1 / signal1_max, 5), round(lf1, 4), round(signal1_max, 4)],
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax = plt.subplot(438)
    plt.loglog(ft2_freq, np.abs(ft2), label="Full motion", color="blue")
    plt.loglog(ft2_freq_pe, np.abs(ft2_pe), label="Pre-event trace", color="red")
    plt.loglog(ft2_freq, smooth_ft2, label="Smoothed fm", color="green", linewidth=1.5)
    plt.loglog(
        ft2_freq, smooth_ft2_pe, label="Smoothed pe", color="grey", linewidth=1.5
    )
    ymin, ymax = ax.get_ylim()
    plt.vlines(0.1, ymin, ymax, color="r", linewidth=1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Fourier amplitude")
    plt.legend(loc=3, prop={"size": 10})
    ax.set_xlim([0.001, 100])
    #    ax.text(0.01, 0.97, snr2_min, horizontalalignment='left', verticalalignment='top', transform = ax.transAxes)
    ax.text(
        0.01,
        0.97,
        [round(lf2_pe / signal2_max, 5), round(lf2_pe, 4), round(signal2_max, 4)],
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.01,
        0.90,
        [round(lf2 / signal2_max, 5), round(lf2, 4), round(signal2_max, 4)],
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax = plt.subplot(439)
    plt.loglog(ftv_freq, np.abs(ftv), label="Full motion", color="blue")
    plt.loglog(ftv_freq_pe, np.abs(ftv_pe), label="Pre-event trace", color="red")
    plt.loglog(ftv_freq, smooth_ftv, label="Smoothed fm", color="green", linewidth=1.5)
    plt.loglog(
        ftv_freq, smooth_ftv_pe, label="Smoothed pe", color="grey", linewidth=1.5
    )
    ymin, ymax = ax.get_ylim()
    plt.vlines(0.1, ymin, ymax, color="r", linewidth=1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Fourier amplitude")
    plt.legend(loc=3, prop={"size": 10})
    ax.set_xlim([0.001, 100])
    #    ax.text(0.01, 0.97, snr3_min, horizontalalignment='left', verticalalignment='top', transform = ax.transAxes)
    ax.text(
        0.01,
        0.97,
        [round(lfv_pe / signalv_max, 5), round(lfv_pe, 4), round(signalv_max, 4)],
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.01,
        0.90,
        [round(lfv / signalv_max, 5), round(lfv, 4), round(signalv_max, 4)],
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )

    ax = plt.subplot(4, 3, 10)
    plt.loglog(ft1_freq, np.divide(smooth_ft1, smooth_ft1_pe), c="k")
    ymin, ymax = ax.get_ylim()
    plt.hlines([2.0], ft1_freq[1], ft1_freq[-1], color=["r"], linestyle="--")
    plt.vlines(0.1, ymin, ymax, color="r", linewidth=1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Signal to noise ratio")
    ax.set_xlim([0.001, 100])
    ax.text(
        0.01,
        0.97,
        snr1_min,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax = plt.subplot(4, 3, 11)
    plt.loglog(ft2_freq, np.divide(smooth_ft2, smooth_ft2_pe), c="k")
    ymin, ymax = ax.get_ylim()
    plt.hlines([2.0], ft1_freq[1], ft1_freq[-1], color=["r"], linestyle="--")
    plt.vlines(0.1, ymin, ymax, color="r", linewidth=1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Signal to noise ratio")
    ax.set_xlim([0.001, 100])
    ax.text(
        0.01,
        0.97,
        snr2_min,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax = plt.subplot(4, 3, 12)
    plt.loglog(ftv_freq, np.divide(smooth_gm_ft, smooth_gm_ft_pe), c="k")
    ymin, ymax = ax.get_ylim()
    plt.hlines([2.0], ft1_freq[1], ft1_freq[-1], color=["r"], linestyle="--")
    plt.vlines([0.1, fmin], ymin, ymax, color=["r", "b"], linewidth=1, linestyle="--")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Geomean Signal to noise ratio")
    ax.set_xlim([0.001, 100])
    ax.text(
        0.01,
        0.97,
        snr_min,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.01,
        0.90,
        [round(fmin, 2), round(1 / fmin, 1)],
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.01,
        0.83,
        [snr_max],
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    plt.savefig(loc_plot, dpi=75)
    plt.clf()
    plt.close("all")
    gc.collect()
