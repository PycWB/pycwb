"""cWB Q-veto on reconstructed/whitened waveforms."""
import numpy as np


def _zero_crossing_segment_maxima(x):
    # Original GetQveto updates xmax with sample i before checking prev*x[i]<0.
    # It excludes sample 0 and never appends the unfinished last half-cycle.
    crossings = np.flatnonzero(x[:-1] * x[1:] < 0) + 1
    if len(crossings) < 2:
        return None
    starts = np.r_[1, crossings[:-1] + 1]
    return np.maximum.reduceat(np.abs(x[1:crossings[-1] + 1]), starts - 1)


def _reference_upsample(wf):
    n = len(wf)
    spectrum = np.fft.rfft(wf)
    padded = np.zeros(2 * n + 1, dtype=complex)
    padded[:len(spectrum)] = spectrum
    if n % 2 == 0:
        # wavearray's packed real FFT keeps Nyquist in slot 1. Resizing the
        # packed buffer moves it to the *new* Nyquist, not the old frequency.
        padded[n // 2] = 0
        padded[-1] = spectrum[-1].real
    # Overall amplitude cancels in both statistics.
    return np.fft.irfft(padded, n=4 * n)


def get_qveto(wf, NTHR=1, ATHR=7.58859):
    """Return Qveto and Qfactor using original cWB peak and FFT conventions.

    NTHR is the number of neighbouring peaks included in the central energy;
    ATHR sets the amplitude threshold for external peaks. Original cWB stores
    both outputs and the neighbour ratio for Qfactor as float32.
    """
    wf = np.asarray(wf, dtype=np.float64)
    if len(wf) < 2:
        return (0.0, 0.0)
    a = _zero_crossing_segment_maxima(_reference_upsample(wf))
    if a is None:
        return (0.0, 0.0)
    # The reference intentionally starts its maximum search at peak 1.
    imax = 1 + np.argmax(a[1:])
    amax = a[imax]
    indices = np.arange(len(a))
    inside = np.abs(indices - imax) <= NTHR
    ein = np.sum(a[inside] ** 2)
    eout = np.sum(a[(~inside) & (a > amax / ATHR)] ** 2)
    qveto = eout / ein if ein > 0 else 0.0
    qfactor = 0.0
    # Avoid undefined neighbour access for a peak at the boundary.
    if 0 < imax < len(a) - 1 and amax > 0:
        ratio = float(np.float32((a[imax - 1] + a[imax + 1]) / amax / 2.0))
        if ratio > 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                qfactor = np.sqrt(-np.pi ** 2 / np.log(ratio) / 2.0)
    return float(np.float32(qveto)), float(np.float32(qfactor))
