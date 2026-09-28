"""Dominant-polarization-frame transforms and the DPF energy regulator.

compute_dpf and compute_dpf_into return the frame and per-pixel quantities;
compute_dpf_regulator returns the scalar energy regulator used by the scan.
"""

from math import sqrt

import numpy as np
from numba import njit, prange, vectorize, float32, uint32


@njit(parallel=True, cache=True)
def compute_dpf_regulator(FP, FX, noise_weights, n_sky, n_ifo, gamma_regulator, network_energy_threshold, sky_valid_indices):
    """Return the sky-averaged DPF energy regulator, not frame arrays.

    Antenna patterns have shape (n_sky, n_ifo); normalized inverse-noise
    weights have shape (n_pix, n_ifo). Only sky_valid_indices enter the average.
    """
    FP = FP.astype(np.float32)
    FX = FX.astype(np.float32)
    noise_weights = noise_weights.astype(np.float32)

    # check shape of FP, FX, and noise_weights
    if FP.shape != (n_sky, n_ifo) or FX.shape != (n_sky, n_ifo) or noise_weights.shape[1] != n_ifo:
        n1, n2 = FP.shape
        m1, m2 = FX.shape
        p1, p2 = noise_weights.shape

        raise ValueError(
            f"FP and FX must have shape (n_sky, n_ifo) and noise_weights must have shape (n_pix, n_ifo), "
            f"got FP: ({n1}, {n2}), FX: ({m1}, {m2}), noise_weights: ({p1}, {p2})"
        )

    n_valid = len(sky_valid_indices)
    aa = np.zeros(n_valid)

    for k in prange(n_valid):
        i = sky_valid_indices[k]
        aa[k] = compute_dpf(FP[i], FX[i], noise_weights)[0]

    FF = np.float64(n_valid)
    ff = np.float64((aa > gamma_regulator).sum())

    return (FF ** 2 / (ff ** 2 + 1.e-9) - 1) * network_energy_threshold


@vectorize([float32(float32, float32)])
def mul_vec(a, b):
    return a * b


@vectorize([float32(float32, float32)])
def div_vec(a, b):
    _o = float32(1e-9)
    return a / (b + _o)


@vectorize([float32(float32, float32)])
def add_vec(a, b):
    return a + b


@vectorize([float32(float32, float32)])
def sub_vec(a, b):
    return a - b


@vectorize([float32(float32, float32)])
def norm_vec(a, b):
    return sqrt(a * a + b * b)


@vectorize([float32(float32, float32)])
def avg_vec(a, b):
    return (a + b) / float32(2.)


@vectorize([float32(float32)])
def sin_from_cc(a):
    return sqrt((float32(1.) - a) / float32(2.))


@vectorize([float32(float32, float32)])
def cos_from_cc(a, si):
    return sqrt((float32(1.) + a) / float32(2.)) if si > float32(0.) else - sqrt((float32(1.) + a) / float32(2.))


@vectorize([uint32(float32)])
def pos_sign_vec(a):
    return uint32(1) if a > float32(0.) else uint32(0)


@njit(cache=True)
def compute_dpf(Fp0, Fx0, noise_weights):
    """
    Compute the dominant polarization frame (DPF)

    Parameters
    ----------
    Fp0 : np.ndarray
        The Fp0 vector for the current sky location.
    Fx0 : np.ndarray
        The Fx0 vector for the current sky location.
    noise_weights : np.ndarray
        Normalized inverse-noise weights for the pixels, shape (NPIX, NIFO).

    Returns
    -------
    tuple
        - NI : float
            Scalar network index used by the regulator.
        - f: np.ndarray
            The plus polarization component in the DPF.
        - F: np.ndarray
            The cross polarization component in the DPF.
        - fp: np.ndarray
            |f+|^2 
        - fx: np.ndarray
            |fx|^2
        - si: np.ndarray
            The sine component of the DPF.
        - co: np.ndarray
            The cosine component of the DPF.
        - ni: np.ndarray
            The network index for each pixel.
    """
    n_pix, n_ifo = noise_weights.shape
    scratch = (np.empty((n_pix, n_ifo), dtype=np.float32),
               np.empty((n_pix, n_ifo), dtype=np.float32),
               np.empty(n_pix, dtype=np.float32), np.empty(n_pix, dtype=np.float32),
               np.empty(n_pix, dtype=np.float32), np.empty(n_pix, dtype=np.float32),
               np.empty(n_pix, dtype=np.float32))
    return compute_dpf_into(Fp0, Fx0, noise_weights, scratch)


@njit(cache=True)
def compute_dpf_into(Fp0, Fx0, noise_weights, scratch):
    """Compute the dominant polarization frame into caller-owned arrays.

    Parameters
    ----------
    Fp0 : np.ndarray
        The Fp0 vector for the current sky location.
    Fx0 : np.ndarray
        The Fx0 vector for the current sky location.
    noise_weights : np.ndarray
        Normalized inverse-noise weights for the pixels, shape (NPIX, NIFO).

    scratch : tuple of numpy.ndarray
        Writable float32 buffers: (f, F, si, co, fp, fx, ni): f/F have shape (n_pix, n_ifo);
        the remaining arrays have shape (n_pix,). fx/ni are reset before accumulation.
        Every buffer is overwritten before use and must not alias any input
        or another scratch buffer. Returned arrays borrow these buffers and
        remain valid only until the next call with the same scratch tuple.

    Returns
    -------
    tuple
        - NI : float
            Scalar network index used by the regulator.
        - f: np.ndarray
            The plus polarization component in the DPF.
        - F: np.ndarray
            The cross polarization component in the DPF.
        - fp: np.ndarray
            |f+|^2
        - fx: np.ndarray
            |fx|^2
        - si: np.ndarray
            The sine component of the DPF.
        - co: np.ndarray
            The cosine component of the DPF.
        - ni: np.ndarray
            The network index for each pixel.
    """
    NPIX, NIFO = noise_weights.shape
    NPIX = uint32(NPIX)
    NIFO = uint32(NIFO)

    # variables for return
    f, F, si, co, fp, fx, ni = scratch

    fx.fill(0)
    ni.fill(0)

    _o = float32(1e-9)

    # Compute f and F
    for j in range(NIFO):
        for i in range(NPIX):
            f[i, j] = mul_vec(noise_weights[i, j], Fp0[j])
            F[i, j] = mul_vec(noise_weights[i, j], Fx0[j])

    # Compute ff, FF, and fF
    for i in range(NPIX):
        _ff = float32(0.0)
        _FF = float32(0.0)
        _fF = float32(0.0)

        for j in range(NIFO):
            _ff += f[i, j] * f[i, j]
            _FF += F[i, j] * F[i, j]
            _fF += F[i, j] * f[i, j]

        # Compute si, co, AP, nn, fp, and cc
        _si = mul_vec(float32(2.0), _fF)  # rotation 2*sin*cos*norm
        _co = sub_vec(_ff, _FF)  # rotation (cos^2-sin^2)*norm
        _AP = add_vec(_ff, _FF)  # total antenna norm
        _nn = norm_vec(_co, _si)  # co/si norm    np.sqrt(_co * _co + _si * _si)
        _cc = div_vec(_co, _nn)  # cos(2p)       _co / (_nn + 1e-9)
        fp[i] = avg_vec(_AP, _nn)  # |f+|^2        (_AP + _nn) / 2.
        si[i] = sin_from_cc(_cc)  # |sin(p)|      sqrt((1. - _cc) / 2.)
        co[i] = cos_from_cc(_cc, _si)  # cos(p)        (sqrt((1. + _cc) / 2.) if _si > 0.0 else - sqrt((1. + _cc) / 2.))

    # Compute f_new, F_new, fF_new, F_new, fx, ni
    for i in range(NPIX):
        for j in range(NIFO):
            f[i, j], F[i, j] = f[i, j] * co[i] + F[i, j] * si[i], F[i, j] * co[i] - f[i, j] * si[i]

        fF_new = float32(0.0)
        for j in range(NIFO):
            fF_new += f[i, j] * F[i, j]
        fF_new = div_vec(fF_new, fp[i])

        for j in range(NIFO):
            F[i, j] -= f[i, j] * fF_new
            fx[i] += F[i, j] * F[i, j]
            ni[i] += f[i, j] ** 4

    NI, NN = float32(0.0), uint32(0)

    # Compute NI and NN
    for i in range(NPIX):
        ni[i] = div_vec(ni[i], mul_vec(fp[i], fp[i]))
        NI += div_vec(fx[i], ni[i])  # sum of |fx|^2/2/ni
        NN += pos_sign_vec(fp[i])  # pixel count
        # NN += 1 if fp[i] > 0.0 else 0

    return sqrt(NI / (NN + 0.01)), f, F, fp, fx, si, co, ni
