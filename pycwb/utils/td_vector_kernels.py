"""
Numba kernels for batch time-delay vector extraction.

This module is intentionally a leaf utility so the data containers in
``pycwb.types`` can call the kernels without importing the cache-building
orchestration in ``pycwb.utils.td_vector_batch``.
"""

import math

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _get_pixel_amplitude_nb(n, m, dT, padded_plane, T0, Tx, M, n_coeffs, J, quad):
    """
    Numba reimplementation of core.time_delay.get_pixel_amplitude.

    Parameters
    ----------
    n, m   : int - time-bin and frequency-layer of the pixel
    dT     : int - delay index in [-J, J]
    padded_plane : float64 array (n_time + 2*n_coeffs, M+1) - zero-padded TF plane
    T0, Tx : float64 arrays (2*J+1, 2*n_coeffs+1) - TD filter tables
    M, n_coeffs, J : int - static filter dimensions
    quad   : bool - False for 00-phase result, True for 90-phase result

    Returns
    -------
    float64 scalar
    """
    is_odd_n = (n & 1) != 0
    is_odd_mn = ((m + n) & 1) != 0
    cancel_even = is_odd_n ^ quad  # XOR
    is_edge_layer = (m == 0) or (m == M)

    dT_idx = dT + J
    win_len = 2 * n_coeffs + 1
    # CWB divides dT by LWDM = L = J/M to convert from TDRate steps to rateANA units.
    # For L=1 this is a no-op; for L>1 it corrects the phase calculation.
    dt_val = float(dT) * float(M) / float(J)  # = dT / L in rateANA sample units

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

    Mirrors CWB's ``WDM::getTDamp`` including the pixel-level wdmShift:
    when ``|dT| >= J`` (= M for L=1), the delay is decomposed as
    ``dT = wdm_shift * J + sub_dT`` using C++-style truncation (rounds
    toward zero), and the amplitude is read from the time-bin shifted by
    ``wdm_shift``. For odd ``wdm_shift`` the two quadratures are swapped
    and a sign correction is applied, exactly as in CWB's ``getTDamp``.

    Parameters
    ----------
    pixel_indices : int32 array (n_pixels,)
    padded00      : float32 array (n_time + 2*n_coeffs, M+1)
    padded90      : float32 array (n_time + 2*n_coeffs, M+1)
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

            # Decompose dT into whole-pixel shift + sub-pixel remainder using
            # C++-style truncation (rounds toward zero, matching WDM::getTDamp).
            if dT >= 0:
                wdm_shift = dT // J
            else:
                wdm_shift = -((-dT) // J)
            sub_dT = dT - wdm_shift * J
            n_eff = n - wdm_shift

            if wdm_shift % 2 != 0:
                # Odd pixel shift: quadratures swap with sign from (n+m) parity,
                # identical to CWB getTDamp() odd-wdmShift branch.
                if (n + m) % 2 != 0:
                    a00 = -_get_pixel_amplitude_nb(n_eff, m, sub_dT, padded90, T0, Tx, M, n_coeffs, J, True)
                    a90 = _get_pixel_amplitude_nb(n_eff, m, sub_dT, padded00, T0, Tx, M, n_coeffs, J, False)
                else:
                    a00 = _get_pixel_amplitude_nb(n_eff, m, sub_dT, padded90, T0, Tx, M, n_coeffs, J, True)
                    a90 = -_get_pixel_amplitude_nb(n_eff, m, sub_dT, padded00, T0, Tx, M, n_coeffs, J, False)
            else:
                # Even pixel shift (including 0): standard per-quadrature paths.
                a00 = _get_pixel_amplitude_nb(n_eff, m, sub_dT, padded00, T0, Tx, M, n_coeffs, J, False)
                a90 = _get_pixel_amplitude_nb(n_eff, m, sub_dT, padded90, T0, Tx, M, n_coeffs, J, True)

            out[p, ki] = a00
            out[p, half + ki] = a90
    return out
