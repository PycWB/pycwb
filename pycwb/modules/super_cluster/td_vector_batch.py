"""
Batch time-delay vector extraction for super_cluster.

Replaces the serial per-pixel loop:
    for pixel in pixels:
        for ifo in range(n_ifo):
            td_vec = wdm.get_td_vec(tf_map, pixel_index=idx, K=K, mode="a")

with a single Numba-parallelised call per (layer, ifo) group:
    batch_get_td_vecs(pixel_indices, padded00, padded90, T0, Tx, M, n_coeffs, K, J)

Notes
-----
- L=1 is assumed (set_td_filter called with L=1), so J = M.
- K < J must hold (always true in practice: TDSize << M).
- This replicates core.time_delay.get_pixel_amplitude + get_td_vec(mode="a").
- Numba compiles once per dtype signature (not per array shape), so there is
  no per-lag JIT-trace-cache accumulation and no associated memory leak.
- Call ``prepare_td_inputs`` to extract numpy arrays from the TF map and
  filter bank before invoking the batch function.
"""

import logging
import math

import numpy as np
from numba import njit, prange

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _get_pixel_amplitude_nb(n, m, dT, padded_plane, T0, Tx, M, n_coeffs, J, quad):
    """
    Numba reimplementation of core.time_delay.get_pixel_amplitude.

    Parameters
    ----------
    n, m   : int  — time-bin and frequency-layer of the pixel
    dT     : int  — delay index in [-J, J]
    padded_plane : float64 array (n_time + 2*n_coeffs, M+1)  — zero-padded TF plane
    T0, Tx : float64 arrays (2*J+1, 2*n_coeffs+1)  — TD filter tables
    M, n_coeffs, J : int  — static filter dimensions
    quad   : bool  — False → 00-phase result, True → 90-phase result

    Returns
    -------
    float64 scalar
    """
    is_odd_n   = (n & 1) != 0
    is_odd_mn  = ((m + n) & 1) != 0
    cancel_even = is_odd_n ^ quad  # XOR
    is_edge_layer = (m == 0) or (m == M)

    dT_idx = dT + J
    win_len = 2 * n_coeffs + 1
    dt_val = float(dT)

    # ---- Same-band term (layer m) ----
    sum_even_same = 0.0
    sum_odd_same = 0.0
    for k in range(win_len):
        val = padded_plane[n + k, m] * T0[dT_idx, k]
        if k % 2 == 0:
            sum_even_same += val
        else:
            sum_odd_same += val

    if is_edge_layer and cancel_even:
        sum_even_same = 0.0
    if is_edge_layer and not cancel_even:
        sum_odd_same = 0.0

    same_phase = m * math.pi * dt_val / M
    if is_odd_n:
        result = sum_even_same * math.cos(same_phase) - sum_odd_same * math.sin(same_phase)
    else:
        result = sum_even_same * math.cos(same_phase) + sum_odd_same * math.sin(same_phase)

    # ---- Cross-band low term (layer m-1), active when m > 0 ----
    if m > 0:
        m_low = m - 1 if m > 0 else 0
        low_even = 0.0
        low_odd = 0.0
        for k in range(win_len):
            val = padded_plane[n + k, m_low] * Tx[dT_idx, k]
            if k % 2 == 0:
                low_even += val
            else:
                low_odd += val

        if m == 1 and cancel_even:
            low_even = 0.0
        if m == 1 and not cancel_even:
            low_odd = 0.0

        low_phase = (2.0 * m - 1.0) * dt_val * math.pi / (2.0 * M)
        if is_odd_n:
            low_term = (low_even - low_odd) * math.sin(low_phase)
        else:
            low_term = (low_even + low_odd) * math.sin(low_phase)

        if m == 1 or m == M:
            low_term *= math.sqrt(2.0)

        if is_odd_mn:
            result -= low_term
        else:
            result += low_term

    # ---- Cross-band high term (layer m+1), active when m < M ----
    if m < M:
        m_high = m + 1 if m < M else M
        high_even = 0.0
        high_odd = 0.0
        for k in range(win_len):
            val = padded_plane[n + k, m_high] * Tx[dT_idx, k]
            if k % 2 == 0:
                high_even += val
            else:
                high_odd += val

        if m == M - 1 and cancel_even:
            high_even = 0.0
        if m == M - 1 and not cancel_even:
            high_odd = 0.0

        high_phase = (2.0 * m + 1.0) * dt_val * math.pi / (2.0 * M)
        if is_odd_n:
            high_term = (high_even + high_odd) * math.sin(high_phase)
        else:
            high_term = (high_even - high_odd) * math.sin(high_phase)

        if m == 0 or m == M - 1:
            high_term *= math.sqrt(2.0)

        if is_odd_mn:
            result -= high_term
        else:
            result += high_term

    # Zero out if edge + cancel condition
    if is_edge_layer and cancel_even:
        return 0.0
    return result


@njit(cache=True, parallel=True)
def batch_get_td_vecs(pixel_indices, padded00, padded90, T0, Tx, M, n_coeffs, K, J):
    """
    Batch TD vector extraction over all pixels in parallel.

    Compiled once per dtype signature (int32/float64) regardless of the
    number of pixels, so there is no per-shape JIT trace cache accumulation.

    Parameters
    ----------
    pixel_indices : int32 array (n_pixels,)
    padded00      : float64 array (n_time + 2*n_coeffs, M+1)
    padded90      : float64 array (n_time + 2*n_coeffs, M+1)
    T0, Tx        : float64 arrays (2*J+1, 2*n_coeffs+1)
    M, n_coeffs, K, J : int

    Returns
    -------
    float32 array (n_pixels, 4*K+2)
        Concatenation of [a00(-K..K), a90(-K..K)] for each pixel.
    """
    n_pixels = len(pixel_indices)
    td_len = 4 * K + 2
    M1 = M + 1
    out = np.empty((n_pixels, td_len), dtype=np.float32)

    for p in prange(n_pixels):
        idx = pixel_indices[p]
        n = idx // M1
        m = idx % M1
        half = 2 * K + 1  # number of delay steps per phase
        for ki in range(2 * K + 1):
            dT = ki - K
            out[p, ki]        = _get_pixel_amplitude_nb(n, m, dT, padded00, T0, Tx, M, n_coeffs, J, False)
            out[p, half + ki] = _get_pixel_amplitude_nb(n, m, dT, padded90, T0, Tx, M, n_coeffs, J, True)
    return out


# ---------------------------------------------------------------------------
# Helpers: extract numpy arrays → call batch → write results back to pixels
# ---------------------------------------------------------------------------

def prepare_td_inputs(tf_map, wdm):
    """
    Extract numpy arrays from a TimeFrequencyMap and WDM instance for batch TD extraction.

    Parameters
    ----------
    tf_map : TimeFrequencyMap  (pycwb type, data is complex128 shape (M+1, n_time))
    wdm : WDMWavelet instance with td_filters already set via set_td_filter

    Returns
    -------
    dict with keys:
        padded00 : float64 ndarray  shape (n_time + 2*n_coeffs, M+1)
        padded90 : float64 ndarray  shape (n_time + 2*n_coeffs, M+1)
        T0       : float64 ndarray  shape (2*J+1, 2*n_coeffs+1)
        Tx       : float64 ndarray  shape (2*J+1, 2*n_coeffs+1)
        M        : int
        n_coeffs : int
        J        : int
    """
    data = np.asarray(tf_map.data, dtype=np.complex128)  # (M+1, n_time)
    # Transpose to (n_time, M+1) — matching time_delay.py convention
    tf00 = np.ascontiguousarray(data.real.T, dtype=np.float64)  # (n_time, M+1)
    tf90 = np.ascontiguousarray(data.imag.T, dtype=np.float64)

    td_filters = wdm.td_filters
    n_coeffs = int(td_filters.n_coeffs)
    M = int(td_filters.M)
    J = int(td_filters.max_delay)   # = M * L; for L=1, J=M

    pad = [(n_coeffs, n_coeffs), (0, 0)]
    padded00 = np.ascontiguousarray(np.pad(tf00, pad), dtype=np.float64)
    padded90 = np.ascontiguousarray(np.pad(tf90, pad), dtype=np.float64)

    T0 = np.ascontiguousarray(td_filters.T0, dtype=np.float64)
    Tx = np.ascontiguousarray(td_filters.Tx, dtype=np.float64)

    return {
        "padded00": padded00,
        "padded90": padded90,
        "T0": T0,
        "Tx": Tx,
        "M": M,
        "n_coeffs": n_coeffs,
        "J": J,
    }


def batch_extract_td_vecs(pixel_indices_np, tf_inputs, K):
    """
    Run the batch Numba TD extraction and return a numpy array.

    Numba compiles ``batch_get_td_vecs`` once per dtype signature (int32/float64),
    not per array shape, so there is no per-lag JIT trace cache accumulation.

    Parameters
    ----------
    pixel_indices_np : np.ndarray, shape (n_pixels,), dtype int32
    tf_inputs : dict from prepare_td_inputs
    K : int  — TD filter half-range (corresponds to config.TDSize)

    Returns
    -------
    np.ndarray, shape (n_pixels, 4*K+2), dtype float32
    """
    return batch_get_td_vecs(
        np.asarray(pixel_indices_np, dtype=np.int32),
        tf_inputs["padded00"],
        tf_inputs["padded90"],
        tf_inputs["T0"],
        tf_inputs["Tx"],
        tf_inputs["M"],
        tf_inputs["n_coeffs"],
        K,
        tf_inputs["J"],
    )
