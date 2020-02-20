import math
from typing import Tuple, Dict
from collections import namedtuple
from functools import partial

import copy
import numpy as np
from scipy.integrate import cumtrapz
from obspy.signal.trigger import ar_pick, pk_baer
from obspy.signal.konnoohmachismoothing import calculate_smoothing_matrix


def get_konno_matrix(ft_len, dt: float = 0.005):
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


def get_freq_ix(ft_freq, lower, upper):
    lower_indices = [i for i, x in enumerate(ft_freq) if x > lower]
    upper_indices = [i for i, x in enumerate(ft_freq) if x < upper]
    lower_index = min(lower_indices)
    upper_index = max(upper_indices)
    return lower_index, upper_index


def comp_fourier_features(
    acc: np.ndarray,
    t: np.ndarray,
    dt: float,
    index: int,
    ko_matrices: Dict[int, np.ndarray] = None,
):
    # Calculate fourier spectra
    ft, ft_freq = compute_fourier(acc, dt, t[-1])
    ft_pe, ft_freq_pe = compute_fourier(acc[0:index], dt, t[-1])

    # Get appropriate smoothing matrix
    if ko_matrices is not None:
        smooth_matrix = ko_matrices[ft_freq.size - 1]
    else:
        raise NotImplementedError

    # Smooth ft with konno ohmachi matrix
    smooth_ft = np.dot(np.abs(ft), smooth_matrix)
    smooth_ft_pe = np.dot(np.abs(ft_pe), smooth_matrix)

    # Calculate snr of frequency intervals
    lower_index, upper_index = get_freq_ix(ft_freq, 0.1, 20)
    snr = np.divide(smooth_ft, smooth_ft_pe)
    snr_min = round(min(snr[lower_index:upper_index]), 5)

    return (
        ft,
        ft_freq,
        ft_pe,
        ft_freq_pe,
        smooth_ft,
        smooth_ft_pe,
        snr,
        snr_min,
        smooth_matrix,
    )


def get_features(
    loc_plot,
    loc_fas,
    gf,
    stat_code,
    sta_sample,
    lta_sample,
    ko_matrices: Dict[int, np.ndarray] = None,
    plot_active: bool = False,
):
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
    Arias = np.sqrt(Arias1 * Arias2)

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
    PGA = np.sqrt(PGA1 * PGA2)
    PN = np.sqrt(amp1_pe * amp2_pe)
    PN_average = np.sqrt(
        np.average(np.abs(acc1[0:index])) * np.average(np.abs(acc2[0:index]))
    )

    # Compute Peak Noise to Peak Ground Acceleration Ratio
    PNPGA = PN / PGA

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
        max_tail_noise_ratio = max_tail_ratio / PN
    else:
        max_tail_ratio1 = 1.0
        max_tail_ratio2 = 1.0
        max_tail_ratiov = 1.0
        max_tail_ratio = 1.0

    # Compute Maximum Head Ratio
    head_duration = 1.0
    head_length = math.ceil(head_duration * sample_rate)
    head_average1 = np.max(abs(acc1[0:head_length]))
    head_average2 = np.max(abs(acc2[0:head_length]))
    head_averagev = np.max(abs(accv[0:head_length]))
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
    # First get all vector indices where abs max acc is greater than or equal, and less than or equal to x*PGA
    hindex1_10 = np.flatnonzero(np.abs(acc1) >= (0.10 * np.max(np.abs(acc1))))
    hindex2_10 = np.flatnonzero(np.abs(acc2) >= (0.10 * np.max(np.abs(acc2))))
    hindex1_20 = np.flatnonzero(np.abs(acc1) >= (0.20 * np.max(np.abs(acc1))))
    hindex2_20 = np.flatnonzero(np.abs(acc2) >= (0.20 * np.max(np.abs(acc2))))

    # Get bracketed duration (from last and first time the index is exceeded)
    if len(hindex1_10) != 0 and len(hindex2_10) != 0:
        bracketedPGA_10 = np.sqrt(
            ((max(hindex1_10) - min(hindex1_10)) * gf.comp_1st.delta_t)
            * ((max(hindex2_10) - min(hindex2_10)) * gf.comp_1st.delta_t)
        )
    else:
        bracketedPGA_10 = 9999.0
    if len(hindex1_20) != 0 and len(hindex2_20) != 0:
        bracketedPGA_20 = np.sqrt(
            ((max(hindex1_20) - min(hindex1_20)) * gf.comp_1st.delta_t)
            * ((max(hindex2_20) - min(hindex2_20)) * gf.comp_1st.delta_t)
        )
    else:
        bracketedPGA_20 = 9999.0

    bracketedPGA_10_20 = bracketedPGA_10 / bracketedPGA_20

    # Calculate Ds575 and Ds595
    Ds575 = np.sqrt(
        ((husid_index1_75 - husid_index1_5) * gf.comp_1st.delta_t)
        * ((husid_index2_75 - husid_index2_5) * gf.comp_1st.delta_t)
    )
    Ds595 = np.sqrt(
        ((husid_index1_95 - husid_index1_5) * gf.comp_1st.delta_t)
        * ((husid_index2_95 - husid_index2_5) * gf.comp_1st.delta_t)
    )

    ft1, ft1_freq, ft1_pe, ft1_freq_pe, smooth_ft1, smooth_ft1_pe, snr1, snr1_min, smooth_matrix = comp_fourier_features(
        gf.comp_1st.acc,
        t,
        gf.comp_1st.delta_t,
        index,
        konno1024,
        konno2048,
        konno4096,
        konno8192,
        konno16384,
        konno32768,
    )
    ft2, ft2_freq, ft2_pe, ft2_freq_pe, smooth_ft2, smooth_ft2_pe, snr2, snr2_min, smooth_matrix = comp_fourier_features(
        gf.comp_2nd.acc,
        t,
        gf.comp_2nd.delta_t,
        index,
        konno1024,
        konno2048,
        konno4096,
        konno8192,
        konno16384,
        konno32768,
    )
    ftv, ftv_freq, ftv_pe, ftv_freq_pe, smooth_ftv, smooth_ftv_pe, snrv, snrv_min, smooth_matrix = comp_fourier_features(
        gf.comp_up.acc,
        t,
        gf.comp_up.delta_t,
        index,
        konno1024,
        konno2048,
        konno4096,
        konno8192,
        konno16384,
        konno32768,
    )

    # geomean of fourier spectra
    ftgm = np.sqrt(np.multiply(abs(ft1), abs(ft2)))
    ftgm_pe = np.sqrt(np.multiply(abs(ft1_pe), abs(ft2_pe)))
    smooth_ftgm = np.dot(ftgm, smooth_matrix)
    smooth_ftgm_pe = np.dot(ftgm_pe, smooth_matrix)

    # snr metrics - min, max and averages
    lower_index, upper_index = get_freq_ix(ft1_freq, 0.1, 20)
    snrgm = np.divide(smooth_ftgm, smooth_ftgm_pe)
    snr_min = round(min(snrgm[lower_index:upper_index]), 5)
    snr_max = round(max(snrgm), 5)

    ##    calculate minimum usable frequency  around the lower corner frequency (0-20Hz)
    ##    find where snr is less than 2.0 and then take max index + 1
    #    try:
    #        fmin_indices = [i for i,x in enumerate(snrgm[0:upper_index]) if x<2.0]
    #        if not fmin_indices:
    #            #i.e. frequency is always above 2.0
    #            fmin = ft1_freq[1]
    #        else:
    #            fmin_index = max(fmin_indices) + 1
    #            fmin = ft1_freq[0:upper_index][fmin_index]
    #    except Exception as e:
    #        print(e)
    #        fmin = ft1_freq[1]
    fmin = 1.0

    lower_index_average, upper_index_average = get_freq_ix(ft1_freq, 0.1, 10)
    snr_average = round(
        np.trapz(
            snrgm[lower_index_average:upper_index_average],
            ft1_freq[lower_index_average:upper_index_average],
        )
        / (ft1_freq[upper_index_average] - ft1_freq[lower_index_average]),
        5,
    )

    lower_index_average, upper_index_average = get_freq_ix(ft1_freq, 0.1, 0.5)
    ft_a1 = round(
        np.trapz(
            smooth_ftgm[lower_index_average:upper_index_average],
            ft1_freq[lower_index_average:upper_index_average],
        )
        / (ft1_freq[upper_index_average] - ft1_freq[lower_index_average]),
        5,
    )
    snr_a1 = round(
        np.trapz(
            snrgm[lower_index_average:upper_index_average],
            ft1_freq[lower_index_average:upper_index_average],
        )
        / (ft1_freq[upper_index_average] - ft1_freq[lower_index_average]),
        5,
    )
    ft_s1 = (smooth_ftgm[upper_index_average] - smooth_ftgm[lower_index_average]) / (
        ft1_freq[upper_index_average] / ft1_freq[lower_index_average]
    )

    lower_index_average, upper_index_average = get_freq_ix(ft1_freq, 0.5, 1.0)
    ft_a2 = round(
        np.trapz(
            smooth_ftgm[lower_index_average:upper_index_average],
            ft1_freq[lower_index_average:upper_index_average],
        )
        / (ft1_freq[upper_index_average] - ft1_freq[lower_index_average]),
        5,
    )
    snr_a2 = round(
        np.trapz(
            snrgm[lower_index_average:upper_index_average],
            ft1_freq[lower_index_average:upper_index_average],
        )
        / (ft1_freq[upper_index_average] - ft1_freq[lower_index_average]),
        5,
    )
    ft_s2 = (smooth_ftgm[upper_index_average] - smooth_ftgm[lower_index_average]) / (
        ft1_freq[upper_index_average] / ft1_freq[lower_index_average]
    )

    lower_index_average, upper_index_average = get_freq_ix(ft1_freq, 1.0, 2.0)
    snr_a3 = round(
        np.trapz(
            snrgm[lower_index_average:upper_index_average],
            ft1_freq[lower_index_average:upper_index_average],
        )
        / (ft1_freq[upper_index_average] - ft1_freq[lower_index_average]),
        5,
    )

    lower_index_average, upper_index_average = get_freq_ix(ft1_freq, 2.0, 5.0)
    snr_a4 = round(
        np.trapz(
            snrgm[lower_index_average:upper_index_average],
            ft1_freq[lower_index_average:upper_index_average],
        )
        / (ft1_freq[upper_index_average] - ft1_freq[lower_index_average]),
        5,
    )

    lower_index_average, upper_index_average = get_freq_ix(ft1_freq, 5.0, 10.0)
    snr_a5 = round(
        np.trapz(
            snrgm[lower_index_average:upper_index_average],
            ft1_freq[lower_index_average:upper_index_average],
        )
        / (ft1_freq[upper_index_average] - ft1_freq[lower_index_average]),
        5,
    )

    ft_a1_a2 = ft_a1 / ft_a2
    ft_s1_s2 = ft_s1 / ft_s2

    # calculate lf to max signal ratios
    signal1_max = max(smooth_ft1)
    lf1 = round(
        np.trapz(smooth_ft1[1:lower_index], ft1_freq[1:lower_index])
        / (ft1_freq[lower_index] - ft1_freq[1]),
        5,
    )
    lf1_pe = round(
        np.trapz(smooth_ft1_pe[1:lower_index], ft1_freq[1:lower_index])
        / (ft1_freq[lower_index] - ft1_freq[1]),
        5,
    )
    signal2_max = max(smooth_ft2)
    lf2 = round(
        np.trapz(smooth_ft2[1:lower_index], ft2_freq[1:lower_index])
        / (ft2_freq[lower_index] - ft2_freq[1]),
        5,
    )
    lf2_pe = round(
        np.trapz(smooth_ft2_pe[1:lower_index], ft2_freq[1:lower_index])
        / (ft2_freq[lower_index] - ft2_freq[1]),
        5,
    )
    signalv_max = max(smooth_ftv)
    lfv = round(
        np.trapz(smooth_ftv[1:lower_index], ftv_freq[1:lower_index])
        / (ftv_freq[lower_index] - ftv_freq[1]),
        5,
    )
    lfv_pe = round(
        np.trapz(smooth_ftv_pe[1:lower_index], ftv_freq[1:lower_index])
        / (ftv_freq[lower_index] - ftv_freq[1]),
        5,
    )
    #    signal1_max = max(smooth_ft1)
    #    lf1_max = max(smooth_ft1[0:lower_index])
    #    lf1_max_pe = max(smooth_ft1_pe[0:lower_index])
    #    signal2_max = max(smooth_ft2)
    #    lf2_max = max(smooth_ft2[0:lower_index])
    #    lf2_max_pe = max(smooth_ft2_pe[0:lower_index])
    #    signalv_max = max(smooth_ftv)
    #    lfv_max = max(smooth_ftv[0:lower_index])
    #    lfv_max_pe = max(smooth_ftv_pe[0:lower_index])
    signal_ratio_max = max([lf1 / signal1_max, lf2 / signal2_max])
    signal_pe_ratio_max = max([lf1_pe / signal1_max, lf2_pe / signal2_max])

    #    f = open(loc_fas,'w')
    #    for i,x in enumerate(ft1_freq):
    #        f.write("%0.10f %0.10f %0.10f %0.10f %0.10f %0.10f %0.10f\n" %(ft1_freq[i], smooth_ft1[i], smooth_ft1_pe[i], smooth_ft2[i], smooth_ft2_pe[i], smooth_ftv[i], smooth_ftv_pe[i]))
    #    f.close()
    if plot_active:
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
            [round(average_tail_ratio1, 5), round(max_tail_ratio1, 5), round(max_head_ratio1, 5)],
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
        #    plt.vlines(min(hindex2_10)*gf.comp_2nd.delta_t, ymin, ymax, color='g', linewidth=2, linestyle='--')
        #    plt.vlines(max(hindex2_10)*gf.comp_2nd.delta_t, ymin, ymax, color='g', linewidth=2, linestyle='--')
        #    plt.vlines(min(hindex2_20)*gf.comp_2nd.delta_t, ymin, ymax, color='g', linewidth=2, linestyle='--')
        #    plt.vlines(max(hindex2_20)*gf.comp_2nd.delta_t, ymin, ymax, color='g', linewidth=2, linestyle='--')
        #    plt.vlines(min(hindex2_30)*gf.comp_2nd.delta_t, ymin, ymax, color='g', linewidth=2, linestyle='--')
        #    plt.vlines(max(hindex2_30)*gf.comp_2nd.delta_t, ymin, ymax, color='g', linewidth=2, linestyle='--')
        #    plt.vlines(min(hindex2_40)*gf.comp_2nd.delta_t, ymin, ymax, color='g', linewidth=2, linestyle='--')
        #    plt.vlines(max(hindex2_40)*gf.comp_2nd.delta_t, ymin, ymax, color='g', linewidth=2, linestyle='--')
        #    plt.vlines(min(hindex2_50)*gf.comp_2nd.delta_t, ymin, ymax, color='g', linewidth=2, linestyle='--')
        #    plt.vlines(max(hindex2_50)*gf.comp_2nd.delta_t, ymin, ymax, color='g', linewidth=2, linestyle='--')
        #    plt.vlines(husid_index2_75*gf.comp_2nd.delta_t, ymin, ymax, color='grey', linewidth=2, linestyle='--')
        #    plt.vlines(husid_index2_95*gf.comp_2nd.delta_t, ymin, ymax, color='grey', linewidth=2, linestyle='--')
        #    plt.hlines([0.10*PGA2], 0, t[-1], color=['g'], linestyle='--')
        #    plt.hlines([-0.10*PGA2], 0, t[-1], color=['g'], linestyle='--')
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
            [round(average_tail_ratio2, 5), round(max_tail_ratio2, 5), round(max_head_ratio2, 5)],
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
            [round(average_tail_ratiov, 5), round(max_tail_ratiov, 5), round(max_head_ratiov, 5)],
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
        plt.loglog(ftv_freq, np.divide(smooth_ftgm, smooth_ftgm_pe), c="k")
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

    return (
        p_pick,
        s_pick,
        snr_min,
        snr_max,
        snr_average,
        signal_ratio_max,
        signal_pe_ratio_max,
        average_tail_ratio,
        max_tail_ratio,
        average_tail_noise_ratio,
        max_tail_noise_ratio,
        max_head_ratio,
        fmin,
        snr_a1,
        snr_a2,
        snr_a3,
        snr_a4,
        snr_a5,
        ft_a1,
        ft_a2,
        ft_a1_a2,
        ft_s1,
        ft_s2,
        ft_s1_s2,
        PGA,
        PN,
        PNPGA,
        Arias,
        bracketedPGA_10_20,
        Ds575,
        Ds595,
        zeroc,
    )
