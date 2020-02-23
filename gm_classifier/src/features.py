"""Contains code for feature/metric extraction from GeoNet ground motion records.
Based on existing implemention
from Xavier Bellagamba (https://github.com/xavierbellagamba/GroundMotionRecordClassifier)
"""
import os
import math
from typing import Tuple, Dict, Union, Any
from collections import namedtuple
from functools import partial

import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from obspy.signal.trigger import ar_pick, pk_baer
from obspy.signal.konnoohmachismoothing import calculate_smoothing_matrix

KONNO_MATRIX_FILENAME_TEMPLATE = "konno_{}.npy"

FourierData = namedtuple(
    "FourierFeatures",
    [
        "ft",
        "ft_freq",
        "ft_pe",
        "ft_freq_pe",
        "smooth_ft",
        "smooth_ft_pe",
        "snr",
        "snr_min",
    ],
)


def get_konno_matrix(ft_len, dt: float = 0.005):
    """Computes the Konno matrix"""
    ft_freq = np.arange(0, ft_len / 2 + 1) * (1.0 / (ft_len * dt))
    return calculate_smoothing_matrix(ft_freq, bandwidth=20, normalize=True)


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


def comp_fourier_data(
    acc: np.ndarray,
    t: np.ndarray,
    dt: float,
    index: int,
    ko_matrices: Union[str, Dict[int, np.ndarray]],
) -> Tuple[FourierData, np.ndarray]:
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
    index: int
        ?
    ko_matrices: dictionary or str
        Either dictionary of Konno matrices or
       directory path, which contains stored konno matrices
       for on-the-fly loading

    Returns
    -------
    FourierFeatuers
        Named tuple which contains the FT features
    np.ndarray
        The smoothing matrix used
    """
    # Calculate fourier spectra
    ft, ft_freq = compute_fourier(acc, dt, t[-1])
    ft_pe, ft_freq_pe = compute_fourier(acc[0:index], dt, t[-1])

    # Get appropriate smoothing matrix
    if isinstance(ko_matrices, dict):
        smooth_matrix = ko_matrices[ft_freq.size - 1]
    elif isinstance(ko_matrices, str) and os.path.isdir(ko_matrices):
        smooth_matrix = np.load(
            os.path.join(
                ko_matrices, KONNO_MATRIX_FILENAME_TEMPLATE.format(ft_freq.size - 1)
            )
        )
    else:
        raise ValueError(
            "The ko_matrices parameter has to either be a "
            "dictionary with the Konno matrices or a directory "
            "path that contains the Konno matrices files."
        )

    # Smooth ft with konno ohmachi matrix
    smooth_ft = np.dot(np.abs(ft), smooth_matrix)
    smooth_ft_pe = np.dot(np.abs(ft_pe), smooth_matrix)

    # Calculate snr of frequency intervals
    lower_index, upper_index = get_freq_ix(ft_freq, 0.1, 20)
    snr = np.divide(smooth_ft, smooth_ft_pe)
    snr_min = np.round(np.min(snr[lower_index:upper_index]), 5)

    return (
        FourierData(
            ft, ft_freq, ft_pe, ft_freq_pe, smooth_ft, smooth_ft_pe, snr, snr_min
        ),
        smooth_matrix,
    )


def get_features(
    gf,
    ko_matrices: Dict[int, np.ndarray] = None,
) -> Tuple[Dict[str, float], Dict[str, float ]]:

    # Create the time vector
    t = np.arange(gf.comp_1st.acc.size) * gf.comp_1st.delta_t

    # Set up a copy of accelerations for plotting later (they get changed by window/taper in the ft step)
    acc1 = copy.deepcopy(gf.comp_1st.acc)
    acc2 = copy.deepcopy(gf.comp_2nd.acc)
    accv = copy.deepcopy(gf.comp_up.acc)

    # check number of zero crossings by
    zarray1 = np.multiply(acc1[0:-2], acc1[1:-1])
    zindices1 = [i for (i, z) in enumerate(zarray1) if z < 0]
    zeroc1 = len(zindices1)
    zarray2 = np.multiply(acc2[0:-2], acc2[1:-1])
    zindices2 = [i for (i, z) in enumerate(zarray2) if z < 0]
    zeroc2 = len(zindices2)
    zarray3 = np.multiply(accv[0:-2], accv[1:-1])
    zindices3 = [i for (i, z) in enumerate(zarray3) if z < 0]
    zeroc3 = len(zindices3)
    zeroc = 10 * np.min([zeroc1, zeroc2, zeroc3]) / t[-1]

    # calculate husid and Arias intensities
    husid1, AI1, Arias1, husid_index1_5, husid_index1_75, husid_index1_95 = compute_husid(
        acc1, t
    )
    husid2, AI2, Arias2, husid_index2_5, husid_index2_75, husid_index2_95 = compute_husid(
        acc2, t
    )
    husidv, AIv, Ariasv, husid_indexv_5, husid_indexv_75, husid_indexv_95 = compute_husid(
        accv, t
    )
    arias = np.sqrt(Arias1 * Arias2)

    # Set up some modified time series for p- and s-wave picking.
    # These are multiplied by an additional 10 because it seems to make the P-wave picking better
    # Also better if the vertical component is sign squared (apparently better)
    tr1 = acc1 * 9806.7 * 10.0
    tr2 = acc2 * 9806.7 * 10.0
    tr3 = np.multiply(np.abs(accv), accv) * np.power(9806.7 * 10.0, 2)
    sample_rate = 1.0 / gf.comp_1st.delta_t

    low_pass = gf.comp_1st.delta_t * 20.0
    high_pass = sample_rate / 20.0

    p_s_pick = partial(
        ar_pick,
        samp_rate=sample_rate,  # sample_rate
        f1=low_pass,  # low_pass
        f2=high_pass,  # high_pass
        lta_p=0.0,  # P-LTA
        sta_p=0.2,  # P-STA,
        lta_s=2.0,  # S-LTA
        sta_s=0.4,  # S-STA
        m_p=8,  # P-AR coefficients
        m_s=8,  # S-coefficients
        l_p=0.4,  # P-length
        l_s=0.2,  # S-length
        s_pick=True,  # S-pick
    )

    # Get p-wave arrival and s-wave arrival picks
    p_pick, s_pick = p_s_pick(tr3, tr1, tr2)

    if p_pick < 5.0:
        # NEXT THING TO DO IS TO TEST CHANGING THE PARAMETERS FOR THE FAKE PICK
        tr3_fake1 = np.multiply(np.abs(acc1), accv) * np.power(9806.7 * 10.0, 2)
        p_pick_fake1, s_pick_fake1 = p_s_pick(tr3_fake1, tr1, tr2)

        tr3_fake2 = np.multiply(np.abs(acc2), accv) * np.power(9806.7 * 10.0, 2)
        p_pick_fake2, s_pick_fake2 = p_s_pick(tr3_fake2, tr1, tr2)

        # p_pick_fake = np.max([p_pick_fake1, p_pick_fake2])
        # s_pick_fake = np.max([s_pick_fake1, s_pick_fake2])
        # p_pick = max(p_pick, p_pick_fake)
        # s_pick = max(s_pick, s_pick_fake)
        p_pick = np.max([p_pick_fake1, p_pick_fake2, p_pick])
        s_pick = np.max([s_pick_fake1, s_pick_fake2, s_pick])
    index = int(np.floor(np.multiply(p_pick, sample_rate)))

    # Calculate max amplitudes of acc time series
    PGA1 = np.round(np.max(np.abs(acc1)), 7)
    PGA2 = np.round(np.max(np.abs(acc2)), 7)
    PGAv = np.round(np.max(np.abs(accv)), 7)
    amp1_pe = np.max(np.abs(acc1[0:index]))
    amp2_pe = np.max(np.abs(acc2[0:index]))

    # Compute PGA and Peak Noise (PN)
    pga = np.sqrt(PGA1 * PGA2)
    pn = np.sqrt(amp1_pe * amp2_pe)
    PN_average = np.sqrt(
        np.average(np.abs(acc1[0:index])) * np.average(np.abs(acc2[0:index]))
    )

    # Compute Peak Noise to Peak Ground Acceleration Ratio
    pn_pga_ratio = pn / pga

    # Compute Average Tail Ratio and Average Tail Noise Ratio
    tail_duration = min(5.0, 0.1 * t[-1])
    tail_length = math.ceil(tail_duration * sample_rate)
    tail_average1 = np.mean(np.abs(acc1[-tail_length:]))
    tail_average2 = np.mean(np.abs(acc2[-tail_length:]))
    tail_averagev = np.mean(np.abs(accv[-tail_length:]))
    if PGA1 != 0 and PGA2 != 0:
        average_tail_ratio1 = tail_average1 / PGA1
        average_tail_ratio2 = tail_average2 / PGA2
        average_tail_ratiov = tail_averagev / PGAv
        average_tail_ratio = np.sqrt(average_tail_ratio1 * average_tail_ratio2)
        average_tail_noise_ratio = average_tail_ratio / PN_average
    else:
        average_tail_ratio1 = 1.0
        average_tail_ratio2 = 1.0
        average_tail_ratiov = 1.0
        average_tail_ratio = 1.0

    # Compute Maximum Tail Ratio and Maximum Tail Noise Ratio
    max_tail_duration = min(2.0, 0.1 * t[-1])
    max_tail_length = math.ceil(max_tail_duration * sample_rate)
    tail_max1 = np.max(np.abs(acc1[-max_tail_length:]))
    tail_max2 = np.max(np.abs(acc2[-max_tail_length:]))
    tail_maxv = np.max(np.abs(accv[-max_tail_length:]))
    if PGA1 != 0 and PGA2 != 0:
        max_tail_ratio1 = tail_max1 / PGA1
        max_tail_ratio2 = tail_max2 / PGA2
        max_tail_ratiov = tail_maxv / PGAv
        max_tail_ratio = np.sqrt(max_tail_ratio1 * max_tail_ratio2)
        max_tail_noise_ratio = max_tail_ratio / pn
    else:
        max_tail_ratio1 = 1.0
        max_tail_ratio2 = 1.0
        max_tail_ratiov = 1.0
        max_tail_ratio = 1.0

    # Compute Maximum Head Ratio
    head_duration = 1.0
    head_length = math.ceil(head_duration * sample_rate)
    head_average1 = np.max(np.abs(acc1[0:head_length]))
    head_average2 = np.max(np.abs(acc2[0:head_length]))
    head_averagev = np.max(np.abs(accv[0:head_length]))
    if PGA1 != 0 and PGA2 != 0:
        max_head_ratio1 = head_average1 / PGA1
        max_head_ratio2 = head_average2 / PGA2
        max_head_ratiov = head_averagev / PGAv
        max_head_ratio = np.sqrt(max_head_ratio1 * max_head_ratio2)
    else:
        max_head_ratio1 = 1.0
        max_head_ratio2 = 1.0
        max_head_ratiov = 1.0
        max_head_ratio = 1.0

    # Bracketed durations between 10%, 20%, 30%, 40% and 50% of PGA
    # First get all vector indices where abs max acc is greater than or equal,
    # and less than or equal to x*PGA
    hindex1_10 = np.flatnonzero(np.abs(acc1) >= (0.10 * np.max(np.abs(acc1))))
    hindex2_10 = np.flatnonzero(np.abs(acc2) >= (0.10 * np.max(np.abs(acc2))))
    hindex1_20 = np.flatnonzero(np.abs(acc1) >= (0.20 * np.max(np.abs(acc1))))
    hindex2_20 = np.flatnonzero(np.abs(acc2) >= (0.20 * np.max(np.abs(acc2))))

    # Get bracketed duration (from last and first time the index is exceeded)
    if len(hindex1_10) != 0 and len(hindex2_10) != 0:
        bracketed_pga_10 = np.sqrt(
            ((max(hindex1_10) - min(hindex1_10)) * gf.comp_1st.delta_t)
            * ((max(hindex2_10) - min(hindex2_10)) * gf.comp_1st.delta_t)
        )
    else:
        bracketed_pga_10 = 9999.0

    if len(hindex1_20) != 0 and len(hindex2_20) != 0:
        bracketed_pga_20 = np.sqrt(
            ((max(hindex1_20) - min(hindex1_20)) * gf.comp_1st.delta_t)
            * ((max(hindex2_20) - min(hindex2_20)) * gf.comp_1st.delta_t)
        )
    else:
        bracketed_pga_20 = 9999.0

    bracketed_pga_10_20 = bracketed_pga_10 / bracketed_pga_20

    # Calculate Ds575 and Ds595
    ds_575 = np.sqrt(
        ((husid_index1_75 - husid_index1_5) * gf.comp_1st.delta_t)
        * ((husid_index2_75 - husid_index2_5) * gf.comp_1st.delta_t)
    )
    ds_595 = np.sqrt(
        ((husid_index1_95 - husid_index1_5) * gf.comp_1st.delta_t)
        * ((husid_index2_95 - husid_index2_5) * gf.comp_1st.delta_t)
    )

    ft_data_1, smooth_matrix = comp_fourier_data(
        gf.comp_1st.acc, t, gf.comp_1st.delta_t, index, ko_matrices
    )
    ft_data_2, smooth_matrix = comp_fourier_data(
        gf.comp_2nd.acc, t, gf.comp_2nd.delta_t, index, ko_matrices
    )
    ft_data_v, smooth_matrix = comp_fourier_data(
        gf.comp_up.acc, t, gf.comp_up.delta_t, index, ko_matrices
    )

    # Compute geomean of fourier spectra
    gm_ft = np.sqrt(np.multiply(np.abs(ft_data_1.ft), np.abs(ft_data_2.ft)))
    gm_ft_pe = np.sqrt(np.multiply(np.abs(ft_data_1.ft_pe), np.abs(ft_data_2.ft_pe)))
    smooth_gm_ft = np.dot(gm_ft, smooth_matrix)
    smooth_gm_ft_pe = np.dot(gm_ft_pe, smooth_matrix)

    # Same for all components
    ft_freq = ft_data_1.ft_freq
    assert np.all(np.isclose(ft_data_1.ft_freq, ft_data_2.ft_freq)) and np.all(
        np.isclose(ft_data_1.ft_freq, ft_data_v.ft_freq)
    )

    # snr metrics - min, max and averages
    lower_index, upper_index = get_freq_ix(ft_freq, 0.1, 20)
    snrgm = np.divide(smooth_gm_ft, smooth_gm_ft_pe)
    snr_min = np.round(np.min(snrgm[lower_index:upper_index]), 5)
    snr_max = np.round(np.max(snrgm), 5)

    fmin = 1.0

    # Compute SNR average across all FT frequencies
    lower_index_average, upper_index_average = get_freq_ix(ft_freq, 0.1, 10)
    snr_average = np.round(
        np.trapz(
            snrgm[lower_index_average:upper_index_average],
            ft_freq[lower_index_average:upper_index_average],
        )
        / (ft_freq[upper_index_average] - ft_freq[lower_index_average]),
        5,
    )

    # Compute the Fourier amplitude ratio
    lower_index_average, upper_index_average = get_freq_ix(ft_freq, 0.1, 0.5)
    fas_0p1_0p5 = np.round(
        np.trapz(
            smooth_gm_ft[lower_index_average:upper_index_average],
            ft_freq[lower_index_average:upper_index_average],
        )
        / (ft_freq[upper_index_average] - ft_freq[lower_index_average]),
        5,
    )
    ft_s1 = (smooth_gm_ft[upper_index_average] - smooth_gm_ft[lower_index_average]) / (
        ft_freq[upper_index_average] / ft_freq[lower_index_average]
    )

    lower_index_average, upper_index_average = get_freq_ix(ft_freq, 0.5, 1.0)
    fas_0p5_1p0 = np.round(
        np.trapz(
            smooth_gm_ft[lower_index_average:upper_index_average],
            ft_freq[lower_index_average:upper_index_average],
        )
        / (ft_freq[upper_index_average] - ft_freq[lower_index_average]),
        5,
    )
    ft_s2 = (smooth_gm_ft[upper_index_average] - smooth_gm_ft[lower_index_average]) / (
        ft_freq[upper_index_average] / ft_freq[lower_index_average]
    )

    fas_ratio = fas_0p1_0p5 / fas_0p5_1p0
    ft_s1_s2 = ft_s1 / ft_s2

    # Computing SNR for the different frequency ranges
    snr_freq_ranges = [
        (0.1, 0.5),
        (0.5, 1.0),
        (1.0, 2.0),
        (2.0, 5.0),
        (5.0, 10.0),
    ]
    snr_values = []
    for freq_min, freq_max in snr_freq_ranges:
        cur_lower_index, cur_upper_index = get_freq_ix(ft_freq, freq_min, freq_max)
        cur_snr = np.round(
            np.trapz(
                snrgm[cur_lower_index:cur_upper_index],
                ft_freq[cur_lower_index:cur_upper_index],
            )
            / (ft_freq[cur_upper_index] - ft_freq[cur_lower_index]),
            5,
        )
        snr_values.append(cur_snr)

    # Compute low frequency (both event & pre-event) FAS to maximum signal FAS ratio
    signal1_max = np.max(ft_data_1.smooth_ft)
    lf1 = np.round(
        np.trapz(ft_data_1.smooth_ft[1:lower_index], ft_freq[1:lower_index])
        / (ft_freq[lower_index] - ft_freq[1]),
        5,
    )
    lf1_pe = np.round(
        np.trapz(ft_data_1.smooth_ft_pe[1:lower_index], ft_freq[1:lower_index])
        / (ft_freq[lower_index] - ft_freq[1]),
        5,
    )

    signal2_max = np.max(ft_data_2.smooth_ft)
    lf2 = np.round(
        np.trapz(ft_data_2.smooth_ft[1:lower_index], ft_freq[1:lower_index])
        / (ft_freq[lower_index] - ft_freq[1]),
        5,
    )
    lf2_pe = np.round(
        np.trapz(ft_data_2.smooth_ft_pe[1:lower_index], ft_freq[1:lower_index])
        / (ft_freq[lower_index] - ft_freq[1]),
        5,
    )

    signalv_max = np.max(ft_data_v.smooth_ft)
    lfv = np.round(
        np.trapz(ft_data_v.smooth_ft[1:lower_index], ft_freq[1:lower_index])
        / (ft_freq[lower_index] - ft_freq[1]),
        5,
    )
    lfv_pe = np.round(
        np.trapz(ft_data_v.smooth_ft_pe[1:lower_index], ft_freq[1:lower_index])
        / (ft_freq[lower_index] - ft_freq[1]),
        5,
    )

    signal_ratio_max = max([lf1 / signal1_max, lf2 / signal2_max])
    signal_pe_ratio_max = max([lf1_pe / signal1_max, lf2_pe / signal2_max])

    features_dict = {
        "pn_pga_ratio": pn_pga_ratio,
        "average_tail_ratio": average_tail_ratio,
        "max_tail_ratio": max_tail_ratio,
        "average_tail_noise_ratio": average_tail_noise_ratio,
        "max_tail_noise_ratio": max_tail_noise_ratio,
        "max_head_ratio": max_head_ratio,
        "bracketed_pga_10_20": bracketed_pga_10_20,
        "signal_pe_ratio_max": signal_pe_ratio_max,
        "signal_ratio_max": signal_ratio_max,
        "fas_ratio": fas_ratio,
        "snr_min": snr_min,
        "snr_max": snr_max,
        "snr_average": snr_average,
        "snr_average_0.1_0.5": snr_values[0],
        "snr_average_0.5_1.0": snr_values[1],
        "snr_average_1.0_2.0": snr_values[2],
        "snr_average_2.0_5.0": snr_values[3],
        "snr_average_5.0_10.0": snr_values[4]
    }

    additional_data = {
        "p_pick": p_pick,
        "s_pick": s_pick,
        "fmin": fmin,
        "fas_0p1_0p5": fas_0p1_0p5,
        "fas_0p5_1p0": fas_0p5_1p0,
        "ft_s1": ft_s1,
        "ft_s2": ft_s2,
        "ft_s1_s2": ft_s1_s2,
        "pga": pga,
        "pn": pn,
        "arias": arias,
        "zeroc": zeroc
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
    plt.loglog(
        ft1_freq, smooth_ft1, label="Smoothed fm", color="green", linewidth=1.5
    )
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
    plt.loglog(
        ft2_freq, smooth_ft2, label="Smoothed fm", color="green", linewidth=1.5
    )
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
    plt.loglog(
        ftv_freq, smooth_ftv, label="Smoothed fm", color="green", linewidth=1.5
    )
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
    plt.vlines(
        [0.1, fmin], ymin, ymax, color=["r", "b"], linewidth=1, linestyle="--"
    )
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
