"""Release waveform summary conventions, independent of waveform synthesis."""

import math
import numpy as np
from numba import njit


@njit(cache=True)
def waveform_rms(data):
    """Compute mean-subtracted RMS in the release grouped reduction order.

    Parameters
    ----------
    data : numpy.ndarray
        One-dimensional float64 waveform samples.

    Returns
    -------
    float
        RMS amplitude, or zero for an empty waveform.
    """
    n = len(data)
    if not n:
        return 0.0
    tail = n % 4
    total = squares = 0.0
    for i in range(tail):
        total += data[i]
        squares += data[i] * data[i]
    for i in range(tail, n, 4):
        total += data[i] + data[i + 1] + data[i + 2] + data[i + 3]
        squares += data[i] * data[i] + data[i + 1] * data[i + 1] + data[i + 2] * data[i + 2] + data[i + 3] * data[i + 3]
    mean = total / n
    # Roundoff can make a zero-variance signal slightly negative.
    return math.sqrt(max(0.0, squares / n - mean * mean))


@njit(cache=True)
def waveform_time(data, start, rate):
    """Compute the energy-weighted waveform time.

    Parameters
    ----------
    data : numpy.ndarray
        One-dimensional waveform samples.
    start : float
        Time of the first sample in seconds.
    rate : float
        Sampling rate in Hz.

    Returns
    -------
    float
        Energy centroid in seconds, or zero for zero energy.
    """
    energy = weighted = 0.0
    for i in range(len(data)):
        value = data[i] * data[i]
        energy += value
        weighted += value * i
    return start + weighted / energy / rate if energy > 0 else 0.0


@njit(cache=True)
def _packed_frequency(real, imag, rate, n):
    """Reduce packed FFT energy with the release Nyquist-at-DC convention."""
    energy = weighted = 0.0
    for i in range(n // 2):
        x = real[i] / n
        y = real[-1] / n if i == 0 else imag[i] / n
        value = x * x + y * y
        energy += value
        weighted += value * (2 * i)
    return 0.5 * weighted * rate / energy / n if energy > 0 else 0.0


def waveform_frequency(data, rate):
    """Match detector::getWFfreq, including its packed Nyquist-at-DC convention.

    Native WDM synthesis returns an even number of samples. The release
    method accesses beyond its storage for odd lengths; reject that unsupported
    case explicitly instead of reproducing an invalid read.

    Parameters
    ----------
    data : numpy.ndarray
        One-dimensional waveform of even length.
    rate : float
        Sampling rate in Hz.

    Returns
    -------
    float
        Energy-weighted frequency in Hz, or zero for an empty/zero-energy waveform.
    """
    n = len(data)
    if not n:
        return 0.0
    if n % 2:
        raise ValueError("Release waveform frequency requires even length")
    spectrum = np.fft.rfft(data)
    return _packed_frequency(spectrum.real, spectrum.imag, rate, n)


@njit(cache=True)
def network_centroids(signal_energy, times, frequencies):
    """Combine detector centroids with release float32 stores.

    Parameters
    ----------
    signal_energy : numpy.ndarray
        Per-detector energies.
    times : numpy.ndarray
        Per-detector time centroids in seconds.
    frequencies : numpy.ndarray
        Per-detector frequency centroids in Hz.

    Returns
    -------
    tuple
        Float32 total energy, network time and network frequency.
    """
    total = np.float32(0.0)
    time = np.float32(0.0)
    frequency = np.float32(0.0)
    for i in range(len(signal_energy)):
        weight = np.float32(signal_energy[i])
        # getWFtime/freq return double; compound assignment stores float.
        time = np.float32(np.float64(time) + np.float64(weight) * times[i])
        frequency = np.float32(np.float64(frequency) + np.float64(weight) * frequencies[i])
        total = np.float32(total + weight)
    if total > 0:
        time = np.float32(time / total)
        frequency = np.float32(frequency / total)
    return total, time, frequency


def compute_sky_posterior_scale(norm, rc, pixel_count, disbalance):
    """Compute the release sky-posterior scale with float32 arithmetic.

    Parameters
    ----------
    norm : float
        Waveform normalization.
    rc : float
        Correlation coefficient.
    pixel_count : float
        Effective selected pixel count.
    disbalance : float
        Network energy disbalance.

    Returns
    -------
    float
        Scale used by sky-localization postprocessing.
    """
    f = np.float32
    return float(f(norm) * f(rc) * np.sqrt(f(pixel_count)) * (f(1) + np.abs(f(1) - f(disbalance))))


def compute_final_detection_statistics(eo, eh, ew, nw, gn, dc, ec, rc, count, nifo, rho, xrho):
    """Float scalar operations from the final network::likelihood2G block.

    Retain native guards for invalid/degenerate energies. Ordinary accepted
    packets follow the release arithmetic and narrowing points.
    """
    f = np.float32
    eo, eh, ew, nw, gn, dc, ec, rc, count, nifo, rho, xrho = map(
        f, (eo, eh, ew, max(nw, 0.0), gn, dc, ec, rc, count, nifo, rho, xrho)
    )
    one = f(1)
    two = f(2)
    ch = (nw + gn) / (count * nifo) if count * nifo > 0 else one
    correction = one + (ch - one) * two * (one - rc) if ch > one else one
    denom_r = ec * rc + (dc + nw + gn) * correction - count * (nifo - one)
    denom_p = ec * rc + (dc + nw + gn) - count * (nifo - one)
    cr = ec * rc / denom_r if denom_r > 0 else f(0)
    cp = ec * rc / denom_p if denom_p > 0 else f(0)
    norm = (eo - eh) / ew if ew > 0 else one
    chi = ch if ch > one else one
    return dict(
        ch=float(ch),
        cr=float(cr),
        cp=float(cp),
        norm=float(norm * two),
        null=float(nw + gn),
        residual=float(nw + gn + dc - count * nifo),
        norm_cor=float(ec * rc),
        rho_reduced=float(rho / np.sqrt(chi)),
        xrho_reduced=float(xrho / np.sqrt(chi)),
    )
