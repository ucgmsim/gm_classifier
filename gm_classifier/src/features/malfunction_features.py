import numpy as np


def _moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def spike_detector(arr):
    """
    Returns spike_factor, which is a metric for quanitifying the
    presence of spikes; spikes are single erroneous data point in an
    acceleration series.

    :param arr:
        (n,) array of acceleration series; np.array
    :returns spike_factor:
        Normalized maximum spike amplitude (0-1); float
    """

    record_amp = np.abs(arr.max() - arr.min())

    max_spike_amp = np.abs(np.diff(np.diff(arr))).max() / 2

    spike_factor = max_spike_amp / record_amp

    return spike_factor


def jerk_detector(arr, window=100):
    """
    Returns jerk_factor, which is a metric for quanitifying the
    presence of jerks; Jerks are baseline offsets which may occur in the
    acceleration series.

    :param arr:
        (n,) array of acceleration series; np.array
    :param window:
        width of the smoothing window on either side; int
    :returns jerk_factor:
        Normalized maximum jerk amplitude (0-1); float
    """

    record_amp = np.abs(arr.max() - arr.min())

    diff_arr = np.abs(np.diff(arr))
    smooth_diff_arr = np.concatenate(
        (np.zeros(window), _moving_average(diff_arr, 2 * window), np.zeros(window - 1),)
    )

    max_jerk_amp = np.abs(diff_arr - smooth_diff_arr).max()

    jerk_factor = max_jerk_amp / record_amp

    return jerk_factor


def lowres_detector(arr):
    """
    Returns lowres_factor, which is a metric for quanitfying the
    resolution; the higher this factor, the greater the likelihood
    of resolution malfunction.

    :param arr:
        (n,) array of acceleration series; np.array
    :returns lowres_factor:
        Likelihood of the record being low-resolution (0-1); float
    """

    diff_arr = np.abs(np.diff(arr))

    n_levels = len(np.unique(diff_arr))
    lowres_factor = ((len(diff_arr) - n_levels) / len(diff_arr)) ** 10

    # Apply a linear decaying weighting function, so that acc series
    # with acc > 120 have lowres == 0 regardless of length
    lowres_factor = lowres_factor * max((1.0 - (n_levels - 50) * 0.0125), 0.0)

    return lowres_factor


def flatline_detector(arr):
    """
	Returns flatline_factor, which is a metric for quanitfying the
	likelihood that a record been clipped/saturated, or has otherwise
	flat-lined; the higher this factor, the greater the likelihood
	of resolution malfunction.

	:param arr:
		(n,) array of acceleration series; np.array
	:returns flatline_factor:
		Relative length of interior amplitudes in the record which are
		saturated/clipped (0-1); float
	"""
    record_amp = np.abs(arr.max() - arr.min()) / 2
    strong_idx = np.where(np.abs(arr) > 0.1 * record_amp)
    strong_arr = arr[strong_idx[0].min() : strong_idx[0].max()]

    # Single large spike, will be picked up by spike-detector
    if len(strong_arr) < 2:
        return 0.0

    diff_arr = np.abs(np.diff(strong_arr))
    min_diff = diff_arr[np.nonzero(diff_arr)].min()
    mod_diff_arr = np.where(diff_arr <= 10 * min_diff, 0, 1)

    idx_pairs = np.where(np.diff(np.hstack(([False], mod_diff_arr == 0, [False]))))[
        0
    ].reshape(-1, 2)
    max_streak = np.diff(idx_pairs, axis=1).max()

    flatline_factor = max_streak / len(strong_arr)

    return flatline_factor


def gainjump_detector(arr):
    """
    Returns gainjump_factor, which is a metric for quantifying the
    likelihood that a record contains a jump in the signal gain. This
    instrument malfunction is characterized by a sudden change in
    amplitude of the backgroun noise; the higher this factor, the
    greater the likelihood of gain malfunction.

    :param arr:
        (n,) array of acceleration series; np.array
    :returns gainjump_factor:
        Likelihood of the record having a gain jump (0-1); float
    """

    record_amp = np.abs(arr.max() - arr.min())

    max_jump_amp = np.diff(np.abs(np.diff(arr))).max()

    gainjump_factor = max_jump_amp / record_amp

    return gainjump_factor
