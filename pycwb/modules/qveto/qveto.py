import numpy as np


def _zero_crossing_segment_maxima(x):
    signs = np.sign(x)
    crossings = np.where(np.diff(signs, prepend=signs[0]) != 0)[0]
    if len(crossings) < 2:
        return None

    starts = np.empty(len(crossings), dtype=np.intp)
    starts[0] = 0
    starts[1:] = crossings[1:]
    return np.maximum.reduceat(np.abs(x), starts)


def get_qveto(wf, NTHR=1, ATHR=7.58859):
    """
    Compute Qveto and Qfactor for a given waveform.

    Parameters:
    wf (np.ndarray): Input waveform (1D array).
    NTHR (int): Neighbor threshold for peak inclusion (default=1).
    ATHR (float): Amplitude threshold ratio for external peaks (default=7.58859).

    Returns:
    tuple: (Qveto, Qfactor) metrics.
    """
    wf = np.asarray(wf, dtype=np.float64)

    # Resample waveform to 4 times the original length using FFT.
    n = len(wf)
    x = np.fft.irfft(np.fft.rfft(wf), n=4 * n)

    a = _zero_crossing_segment_maxima(x)
    # Find global maximum and its index
    if a is None or len(a) == 0:
        return (0.0, 0.0)
    imax = np.argmax(a)
    amax = a[imax]

    # Compute Qveto (energy ratio)
    indices = np.arange(len(a))
    # Mask for peaks inside the NTHR window
    mask_in = (np.abs(indices - imax) <= NTHR)
    ein = np.sum(a[mask_in] ** 2)
    # Mask for peaks outside the window and above amplitude threshold
    mask_out = ~mask_in & (a > amax / ATHR)
    eout = np.sum(a[mask_out] ** 2)
    Qveto = eout / ein if ein > 0 else 0.0

    # Compute Qfactor (using neighboring peaks)
    if imax < 1 or imax >= len(a) - 1:
        Qfactor = 0.0
    else:
        R = (a[imax - 1] + a[imax + 1]) / (2.0 * amax)
        if R <= 0:
            Qfactor = 0.0
        else:
            # Ensure the logarithm argument is positive
            Qfactor = np.sqrt(-(np.pi ** 2) / (2 * np.log(R)))

    return (float(Qveto), float(Qfactor))
