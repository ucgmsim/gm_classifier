import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths


def _moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def numpeaks_detector(arr, distance=100):
    """
	Returns numpeaks_factor, which is larger the more peaks are
	detected.

	:param arr:
        (n,) array of probability series; np.array
	:param distance:
        min distance between peaks permitted; int
	:returns numpeaks_factor:
		log of number of peaks detected (0-1); float
	"""
    prob_amp = arr.max()
    numpeaks_factor = 0
    low_bound = 0.1

    if prob_amp > 0:
        max_vals = sorted(
            arr[
                find_peaks(
                    arr,
                    height=low_bound,
                    distance=distance,
                    prominence=0.05 * prob_amp,
                )[0]
            ]
        )
        num_maxs = len(max_vals)

        if num_maxs > 1:
            numpeaks_factor = min(np.log10(num_maxs), 1)

    return numpeaks_factor


def multimax_detector(arr, distance=100):
    """
	Returns multimax_factor, which is a metric for quanitifying the
	presence of multiple probability peaks in the probability array
	returned by PhaseNet. Considers the amplitudes of two largest peaks
	only.

	:param arr:
		(n,) array of probability series; np.array
	:param distance:
		min distance between peaks permitted; int
	:returns multimax_factor:
		Normalized sum of amplitudes of secondary maxima (0-1); float
	"""
    prob_amp = arr.max()
    multimax_factor = 0
    low_bound = min(max(0.25, 0.5 * prob_amp), 0.75 * prob_amp)

    if prob_amp > 0:
        max_vals = sorted(
            arr[
                find_peaks(
                    arr,
                    height=low_bound,
                    distance=distance,
                    prominence=0.25 * prob_amp,
                )[0]
            ]
        )
        num_maxs = len(max_vals)

        if num_maxs > 1:
            multimax_factor = (np.array(max_vals[:-1]).max() - low_bound) / (
                prob_amp - low_bound
            )

    return multimax_factor


def multidist_detector(arr, distance=100):
    """
	Returns multidist_factor, which is a metric for quantifying the
	presence of multiple probability distributions in the probability
	array returned by PhaseNet. Only considers the two largest widths.
	Assumes only positive values.

	:param arr:
		(n,) array of probability series; np.array
	:param distance:
		min distance between peaks permitted; int
	:returns multidist_factor:
		Normalized sum of areas of secondary prob. dist. (0-1); float
	"""
    prob_amp = arr.max()
    multidist_factor = 0
    low_bound = min(max(0.25, 0.5 * prob_amp), 0.75 * prob_amp)

    if prob_amp > 0:
        peaks = find_peaks(
            arr, height=low_bound, distance=distance, prominence=0.25 * prob_amp,
        )

        max_vals = list(arr[peaks[0]])
        max_widths = sorted(peak_widths(arr, peaks[0], rel_height=0.5)[0])
        # max_dists = sorted([m*w for m,w in zip(max_vals,max_widths)])
        num_maxs = len(max_vals)

        if num_maxs > 1:
            multidist_factor = (np.array(max_widths[:-1]).max()) / (
                np.array(max_widths).max()
            )

    return multidist_factor
