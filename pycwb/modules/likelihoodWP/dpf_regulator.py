"""Allocation-reduced scalar DPF calculation for the regulator pass."""

from math import sqrt

import numpy as np
from numba import njit, prange, float32, uint32

from .dpf import mul_vec, sub_vec, add_vec, norm_vec, div_vec, avg_vec, sin_from_cc, cos_from_cc, pos_sign_vec


@njit(cache=True)
def dpf_index_only(Fp0, Fx0, rms):
    """Compute dpf_np_loops_vec(...)[0], preserving float32 rounding/order.

    The regulator discards the per-pixel output arrays. Two detector-sized
    work arrays suffice here; the sky scan still computes its full DPF.

    Parameters
    ----------
    Fp0, Fx0 : numpy.ndarray
        Float32 plus/cross antenna patterns for one direction, shape (n_ifo,).
    rms : numpy.ndarray
        Float32 pixel noise weights, shape (n_pix, n_ifo).

    Returns
    -------
    float
        Scalar network index, matching element zero of dpf_np_loops_vec.
    """
    n_pix, n_ifo = rms.shape
    f = np.empty(n_ifo, dtype=np.float32)
    F = np.empty(n_ifo, dtype=np.float32)
    NI, NN = float32(0.0), uint32(0)
    for i in range(n_pix):
        ff, FF, fF = float32(0.0), float32(0.0), float32(0.0)
        for j in range(n_ifo):
            f[j] = mul_vec(rms[i, j], Fp0[j])
            F[j] = mul_vec(rms[i, j], Fx0[j])
            ff += f[j] * f[j]
            FF += F[j] * F[j]
            fF += F[j] * f[j]
        si = mul_vec(float32(2.0), fF)
        co = sub_vec(ff, FF)
        AP = add_vec(ff, FF)
        nn = norm_vec(co, si)
        cc = div_vec(co, nn)
        fp = avg_vec(AP, nn)
        rotation_sin = sin_from_cc(cc)
        rotation_cos = cos_from_cc(cc, si)
        for j in range(n_ifo):
            f[j], F[j] = (f[j] * rotation_cos + F[j] * rotation_sin, F[j] * rotation_cos - f[j] * rotation_sin)
        fF_new = float32(0.0)
        for j in range(n_ifo):
            fF_new += f[j] * F[j]
        fF_new = div_vec(fF_new, fp)
        fx, ni = float32(0.0), float32(0.0)
        for j in range(n_ifo):
            F[j] -= f[j] * fF_new
            fx = float32(fx + F[j] * F[j])
            # The original stores this sum into float32 ni[i] each iteration.
            ni = float32(ni + f[j] ** 4)
        ni = div_vec(ni, mul_vec(fp, fp))
        NI += div_vec(fx, ni)
        NN += pos_sign_vec(fp)
    return sqrt(NI / (NN + 0.01))


@njit(parallel=True, cache=True)
def calculate_dpf_scalar(FP, FX, rms, n_sky, n_ifo, gamma_regulator, network_energy_threshold, sky_valid_indices):
    """Compute the regulator without retaining per-pixel DPF output arrays.

    Parameters
    ----------
    FP, FX : numpy.ndarray
        Antenna patterns, shape (n_sky, n_ifo); narrowed to float32.
    rms : numpy.ndarray
        Pixel noise weights, shape (n_pix, n_ifo); narrowed to float32.
    n_sky, n_ifo : int
        Sky-grid and detector dimensions used to validate inputs.
    gamma_regulator : float
        Threshold for counting valid directions by scalar DPF index.
    network_energy_threshold : float
        Multiplier applied to the final regulator.
    sky_valid_indices : numpy.ndarray
        Int64 indices of sky directions to evaluate.

    Returns
    -------
    float
        Regulator using the original float64 count reduction and offset.
    """
    FP = FP.astype(np.float32)
    FX = FX.astype(np.float32)
    rms = rms.astype(np.float32)
    if FP.shape != (n_sky, n_ifo) or FX.shape != (n_sky, n_ifo) or rms.shape[1] != n_ifo:
        raise ValueError("FP/FX must be (n_sky, n_ifo), rms must be (n_pix, n_ifo)")
    n_valid = len(sky_valid_indices)
    aa = np.zeros(n_valid)
    for k in prange(n_valid):
        i = sky_valid_indices[k]
        aa[k] = dpf_index_only(FP[i], FX[i], rms)
    FF = np.float64(n_valid)
    ff = np.float64((aa > gamma_regulator).sum())
    return (FF**2 / (ff**2 + 1.0e-9) - 1) * network_energy_threshold
