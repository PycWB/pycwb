"""Numba implementation of WDM time-delay max-energy."""

from __future__ import annotations

import dataclasses
import logging
import math
import os

import numpy as np
from wdm_wavelet.wdm import t2w_numba as _wdm_t2w_numba
from wdm_wavelet.wdm import w2t_numba as _wdm_w2t_numba

from pycwb.types.time_frequency_map import TimeFrequencyMap

from .time_delay_common import (
    frequency_bounds,
    sample_rate_from_tf_map,
    time_series_length,
    validate_time_delay_inputs,
)
from .time_delay_jax import _HAS_JAX, time_delay_max_energy
from .time_delay_packet import _compute_packet_energy_params

try:
    from wdm_wavelet.core.t2w import t2w_numba_core as _wdm_t2w_numba_core
    from numba.core.registry import CPUDispatcher as _NumbaCPUDispatcher

    _HAS_T2W_NUMBA_CORE = isinstance(_wdm_t2w_numba_core, _NumbaCPUDispatcher)
except Exception:
    _wdm_t2w_numba_core = None
    _HAS_T2W_NUMBA_CORE = False

try:
    import numba

    _HAS_NUMBA = True
    _NUMBA_IMPORT_ERROR = None
except Exception as exc:
    numba = None
    _HAS_NUMBA = False
    _NUMBA_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)


if _HAS_NUMBA:

    @numba.njit(cache=True, fastmath=True)
    def _wdm_packet_energy_nb(real_f, imag_f, M, T, J, jb, je, mL, mH, mean, p):
        """Numba njit energy kernel. Returns (M, T) energy array."""
        energy = np.zeros((M, T), dtype=np.float64)
        for j in range(J):
            m = j % M
            t = j // M
            # Pixels outside [jb, je): keep raw 00-phase forward amplitude (clamped at 0),
            # matching C++ wdmPacket's resize+maxEnergy "max(0, tmp)" boundary behaviour.
            if j < jb or j >= je:
                boundary_raw = real_f[j]
                if boundary_raw > 0.0:
                    energy[m, t] = boundary_raw
                continue
            if m < mL or m > mH:
                continue

            ss = 0.0
            ee = 0.0
            EE = 0.0
            for n in range(1, 9):
                idx = j + p[n]
                if idx < 0:
                    idx = 0
                elif idx >= J:
                    idx = J - 1
                qr = real_f[idx]
                qi = imag_f[idx]
                ss += qr * qi
                ee += qr * qr
                EE += qi * qi

            q0r = real_f[j]
            q0i = imag_f[j]
            ss += q0r * q0i * (mean - 8.0)
            ee += q0r * q0r * (mean - 8.0)
            EE += q0i * q0i * (mean - 8.0)

            cc = ee - EE
            ss2 = 2.0 * ss
            nn_sq = cc * cc + ss2 * ss2
            nn = math.sqrt(nn_sq) if nn_sq > 0.0 else 0.0
            sum_eeEE = ee + EE
            if sum_eeEE < nn:
                nn = sum_eeEE

            h1 = (sum_eeEE + nn) / 2.0
            h2 = (sum_eeEE - nn) / 2.0
            a1 = math.sqrt(h1) if h1 > 0.0 else 0.0
            a2 = math.sqrt(h2) if h2 > 0.0 else 0.0
            aa = a1 + a2

            em = sum_eeEE / 2.0 if mean == 1.0 else (aa * aa) / 4.0
            energy[m, t] = em

        return energy

    @numba.njit(cache=True, fastmath=True)
    def _wdm_packet_energy_tm_nb(real_f, imag_f, M, T, J, jb, je, mL, mH, mean, p):
        """Numba packet-energy kernel with time-major output (T, M)."""
        energy = np.zeros((T, M), dtype=np.float64)
        for j in range(J):
            m = j % M
            t = j // M

            if j < jb or j >= je:
                boundary_raw = real_f[j]
                if boundary_raw > 0.0:
                    energy[t, m] = boundary_raw
                continue
            if m < mL or m > mH:
                continue

            ss = 0.0
            ee = 0.0
            EE = 0.0
            for n in range(1, 9):
                idx = j + p[n]
                if idx < 0:
                    idx = 0
                elif idx >= J:
                    idx = J - 1
                qr = real_f[idx]
                qi = imag_f[idx]
                ss += qr * qi
                ee += qr * qr
                EE += qi * qi

            q0r = real_f[j]
            q0i = imag_f[j]
            ss += q0r * q0i * (mean - 8.0)
            ee += q0r * q0r * (mean - 8.0)
            EE += q0i * q0i * (mean - 8.0)

            cc = ee - EE
            ss2 = 2.0 * ss
            nn_sq = cc * cc + ss2 * ss2
            nn = math.sqrt(nn_sq) if nn_sq > 0.0 else 0.0
            sum_eeEE = ee + EE
            if sum_eeEE < nn:
                nn = sum_eeEE

            h1 = (sum_eeEE + nn) / 2.0
            h2 = (sum_eeEE - nn) / 2.0
            a1 = math.sqrt(h1) if h1 > 0.0 else 0.0
            a2 = math.sqrt(h2) if h2 > 0.0 else 0.0
            aa = a1 + a2

            em = sum_eeEE / 2.0 if mean == 1.0 else (aa * aa) / 4.0
            energy[t, m] = em

        return energy

    @numba.njit(cache=True, fastmath=True)
    def _max_inplace_2d_nb(current_max, candidate):
        """Update a 2D max array in-place."""
        n0, n1 = current_max.shape
        for i in range(n0):
            for j in range(n1):
                val = candidate[i, j]
                if val > current_max[i, j]:
                    current_max[i, j] = val

    @numba.njit(cache=True, fastmath=True)
    def _transpose_tm_to_mt_nb(current_max_tm):
        """Return a contiguous (M, T) copy from a time-major (T, M) array."""
        T, M = current_max_tm.shape
        out = np.empty((M, T), dtype=np.float64)
        for t in range(T):
            for m in range(M):
                out[m, t] = current_max_tm[t, m]
        return out

    @numba.njit(cache=True, parallel=True, fastmath=True)
    def _time_delay_max_energy_pattern_loop_nb(
        ts_data,
        filt,
        n_filter_taps,
        MM_eff,
        M_int,
        return_quadrature,
        max_delay,
        downsample,
        M_val,
        T_val,
        J,
        jb,
        je,
        mL,
        mH,
        mean,
        p,
        pattern,
    ):
        """Fully JIT-compiled time-delay max-energy loop (parallel over delays).

        Calls ``t2w_numba_core`` and ``_wdm_packet_energy_nb`` directly
        (njit-to-njit).  Each delay iteration uses an independent copy of
        ``ts_data`` so the k-loop can run in parallel via ``numba.prange``.

        Correctness note: the original sequential C++ cpf loop leaves the
        tail/head k samples of ``xx`` from the *previous* iteration unchanged.
        Here each iteration uses a fresh copy of ``ts_data`` instead. Those
        k boundary samples always fall within the inactive ``jb`` margin
        (jb >= 4*M time bins, while k << 4*M for any realistic max_delay),so
        the two formulations produce identical active-pixel results.
        """
        _, tf0 = _wdm_t2w_numba_core(
            ts_data, filt, n_filter_taps, MM_eff, M_int, return_quadrature
        )
        current_max = _wdm_packet_energy_nb(
            tf0[0].ravel(), tf0[1].ravel(), M_val, T_val, J, jb, je, mL, mH, mean, p
        )

        # Pre-compute number of valid delay steps for prange
        size = len(ts_data)
        n_iters = 0
        k = downsample
        while k <= max_delay and k < size:
            n_iters += 1
            k += downsample

        if n_iters > 0:
            # Allocate one energy array per shifted signal (2 per delay: left + right)
            all_en = np.empty((n_iters * 2, M_val, T_val), dtype=np.float64)

            for i in numba.prange(n_iters):  # parallel over delay values
                k_i = (i + 1) * downsample

                # cpf_left: copy ts_data[k_i:] into xx[0:size-k_i]; tail stays = ts_data tail
                xx_l = ts_data.copy()
                xx_l[: size - k_i] = ts_data[k_i:]
                _, tf_l = _wdm_t2w_numba_core(
                    xx_l, filt, n_filter_taps, MM_eff, M_int, return_quadrature
                )
                all_en[2 * i] = _wdm_packet_energy_nb(
                    tf_l[0].ravel(),
                    tf_l[1].ravel(),
                    M_val,
                    T_val,
                    J,
                    jb,
                    je,
                    mL,
                    mH,
                    mean,
                    p,
                )

                # cpf_right: copy ts_data[0:size-k_i] into xx[k_i:]; head stays = ts_data head
                xx_r = ts_data.copy()
                xx_r[k_i:] = ts_data[: size - k_i]
                _, tf_r = _wdm_t2w_numba_core(
                    xx_r, filt, n_filter_taps, MM_eff, M_int, return_quadrature
                )
                all_en[2 * i + 1] = _wdm_packet_energy_nb(
                    tf_r[0].ravel(),
                    tf_r[1].ravel(),
                    M_val,
                    T_val,
                    J,
                    jb,
                    je,
                    mL,
                    mH,
                    mean,
                    p,
                )

            # Sequential reduction (trivially fast — dominates nothing)
            for i in range(n_iters * 2):
                current_max = np.maximum(current_max, all_en[i])

        # C++ zeros layer 0 only
        current_max[0, :] = 0.0
        if pattern in (5, 6, 9) and current_max.shape[0] > 2:
            current_max[1, :] = 0.0

        return current_max

    @numba.njit(cache=True, parallel=True, fastmath=True)
    def _time_delay_max_energy_pattern_loop_tm_nb(
        ts_data,
        filt,
        n_filter_taps,
        MM_eff,
        M_int,
        return_quadrature,
        max_delay,
        downsample,
        M_val,
        T_val,
        J,
        jb,
        je,
        mL,
        mH,
        mean,
        p,
        pattern,
    ):
        """Delay-parallel loop using time-major packet-energy buffers."""
        _, tf0 = _wdm_t2w_numba_core(
            ts_data, filt, n_filter_taps, MM_eff, M_int, return_quadrature
        )
        current_max = _wdm_packet_energy_tm_nb(
            tf0[0].ravel(), tf0[1].ravel(), M_val, T_val, J, jb, je, mL, mH, mean, p
        )

        size = len(ts_data)
        n_iters = 0
        k = downsample
        while k <= max_delay and k < size:
            n_iters += 1
            k += downsample

        if n_iters > 0:
            all_en = np.empty((n_iters * 2, T_val, M_val), dtype=np.float64)

            for i in numba.prange(n_iters):
                k_i = (i + 1) * downsample

                xx_l = ts_data.copy()
                xx_l[: size - k_i] = ts_data[k_i:]
                _, tf_l = _wdm_t2w_numba_core(
                    xx_l, filt, n_filter_taps, MM_eff, M_int, return_quadrature
                )
                all_en[2 * i] = _wdm_packet_energy_tm_nb(
                    tf_l[0].ravel(),
                    tf_l[1].ravel(),
                    M_val,
                    T_val,
                    J,
                    jb,
                    je,
                    mL,
                    mH,
                    mean,
                    p,
                )

                xx_r = ts_data.copy()
                xx_r[k_i:] = ts_data[: size - k_i]
                _, tf_r = _wdm_t2w_numba_core(
                    xx_r, filt, n_filter_taps, MM_eff, M_int, return_quadrature
                )
                all_en[2 * i + 1] = _wdm_packet_energy_tm_nb(
                    tf_r[0].ravel(),
                    tf_r[1].ravel(),
                    M_val,
                    T_val,
                    J,
                    jb,
                    je,
                    mL,
                    mH,
                    mean,
                    p,
                )

            for i in range(n_iters * 2):
                _max_inplace_2d_nb(current_max, all_en[i])

        current_max[:, 0] = 0.0
        if pattern in (5, 6, 9) and current_max.shape[1] > 2:
            current_max[:, 1] = 0.0

        return _transpose_tm_to_mt_nb(current_max)

    def _normalize_numba_max_energy_mode(mode):
        mode = str(mode or "parallel").strip().lower()
        aliases = {
            "parallel": "parallel",
            "prange": "parallel",
            "default": "parallel",
            "time-major": "time-major",
            "time_major": "time-major",
            "tm": "time-major",
            "parallel-tm": "time-major",
            "parallel_tm": "time-major",
        }
        if mode not in aliases:
            raise ValueError(
                "PYCWB_NUMBA_MAX_ENERGY_MODE must be one of "
                "{'parallel', 'time-major'} "
                f"(got {mode!r})"
            )
        return aliases[mode]

    def _time_delay_max_energy_pattern_nb(
        ts_data,
        wavelet_M,
        wavelet_m_H,
        wavelet_filter,
        max_delay,
        downsample,
        mm_mode,
        pattern,
        edge,
        wavelet_rate,
        f_low,
        f_high,
        df,
        mode="parallel",
    ):
        """Numba-accelerated time-delay max-energy loop (pattern path).

        Returns an ``(M, T)`` numpy float64 energy array.

        When ``t2w_numba_core`` is available (requires *rocket-fft*), the
        entire delay loop runs inside a single ``@numba.njit`` function.
        Otherwise falls back to calling the Python-level ``t2w_numba``
        wrapper per iteration.
        """
        # Pre-compute t2w parameters (Python-level, done once)
        n_filter_taps = int(wavelet_m_H)
        M_int = int(wavelet_M)
        MM_eff = M_int if mm_mode <= 0 else int(mm_mode)
        return_quadrature = mm_mode < 0
        filt = np.ascontiguousarray(
            np.asarray(wavelet_filter, dtype=np.float64).ravel()[:n_filter_taps]
        )

        # Pre-compute TF map dimensions
        n_input = len(ts_data)
        remainder = n_input % MM_eff
        aligned_length = n_input if remainder == 0 else n_input + (MM_eff - remainder)
        n_time_bins = aligned_length // MM_eff
        M_val = M_int + 1  # n_freq
        T_val = n_time_bins
        J = M_val * T_val

        pattern_abs = abs(int(pattern))
        jb, je, mL, mH, mean, p = _compute_packet_energy_params(
            M_val, T_val, pattern_abs, edge, wavelet_rate, f_low, f_high, df
        )

        # ---- fully-JIT path (t2w_numba_core is njit-compiled, i.e. rocket-fft present) ----
        if _HAS_T2W_NUMBA_CORE:
            mode = _normalize_numba_max_energy_mode(mode)
            if mode == "time-major":
                return _time_delay_max_energy_pattern_loop_tm_nb(
                    ts_data,
                    filt,
                    n_filter_taps,
                    MM_eff,
                    M_int,
                    return_quadrature,
                    int(max_delay),
                    int(downsample),
                    M_val,
                    T_val,
                    J,
                    jb,
                    je,
                    mL,
                    mH,
                    mean,
                    p,
                    pattern_abs,
                )
            return _time_delay_max_energy_pattern_loop_nb(
                ts_data,
                filt,
                n_filter_taps,
                MM_eff,
                M_int,
                return_quadrature,
                int(max_delay),
                int(downsample),
                M_val,
                T_val,
                J,
                jb,
                je,
                mL,
                mH,
                mean,
                p,
                pattern_abs,
            )
        else:
            print(
                "Warning: t2w_numba_core is not available; falling back to slower Python-level loop for time-delay max energy."
            )

        # ---- fallback: Python-level t2w_numba wrapper per iteration ----
        _, _, tf0 = _wdm_t2w_numba(
            wavelet_M, wavelet_m_H, ts_data, wavelet_filter, mm_mode
        )
        re0 = np.ascontiguousarray(tf0[0])
        im0 = np.ascontiguousarray(tf0[1])

        current_max = _wdm_packet_energy_nb(
            re0.ravel(), im0.ravel(), M_val, T_val, J, jb, je, mL, mH, mean, p
        )

        k = int(downsample)
        xx = ts_data.copy()
        size = len(ts_data)
        while k <= int(max_delay) and k < size:
            xx[: size - k] = ts_data[k:]
            _, _, tf_left = _wdm_t2w_numba(
                wavelet_M, wavelet_m_H, xx, wavelet_filter, mm_mode
            )
            re_left = np.ascontiguousarray(tf_left[0])
            im_left = np.ascontiguousarray(tf_left[1])
            en_left = _wdm_packet_energy_nb(
                re_left.ravel(),
                im_left.ravel(),
                M_val,
                T_val,
                J,
                jb,
                je,
                mL,
                mH,
                mean,
                p,
            )
            current_max = np.maximum(current_max, en_left)

            xx[k:] = ts_data[: size - k]
            _, _, tf_right = _wdm_t2w_numba(
                wavelet_M, wavelet_m_H, xx, wavelet_filter, mm_mode
            )
            re_right = np.ascontiguousarray(tf_right[0])
            im_right = np.ascontiguousarray(tf_right[1])
            en_right = _wdm_packet_energy_nb(
                re_right.ravel(),
                im_right.ravel(),
                M_val,
                T_val,
                J,
                jb,
                je,
                mL,
                mH,
                mean,
                p,
            )
            current_max = np.maximum(current_max, en_right)

            k += int(downsample)

        current_max[0, :] = 0.0
        if pattern_abs in (5, 6, 9) and current_max.shape[0] > 2:
            current_max[1, :] = 0.0

        return current_max  # shape (M, T)

else:

    def _wdm_packet_energy_nb(*args, **kwargs):
        raise RuntimeError(
            f"numba is required for the numba backend but is unavailable: {_NUMBA_IMPORT_ERROR}"
        )

    def _normalize_numba_max_energy_mode(mode):
        raise RuntimeError(
            f"numba is required for the numba backend but is unavailable: {_NUMBA_IMPORT_ERROR}"
        )

    def _time_delay_max_energy_pattern_nb(*args, **kwargs):
        raise RuntimeError(
            f"numba is required for the numba backend but is unavailable: {_NUMBA_IMPORT_ERROR}"
        )


def time_delay_max_energy_numba(
    tf_map: TimeFrequencyMap, dt, downsample=1, pattern=0, hist=None, mode=None
):
    """
    Numba-accelerated version of :func:`time_delay_max_energy`.

    Uses ``@numba.njit`` for the inner energy kernel instead of JAX JIT,
    which avoids XLA compilation overhead and works without a JAX install.
    Only the *pattern* path (``pattern != 0``) is fully supported; for
    ``pattern == 0`` the function falls back to the JAX implementation when
    JAX is available, or raises ``NotImplementedError``.

    :param tf_map: detector TF map
    :type tf_map: TimeFrequencyMap
    :param dt: max time delay in seconds
    :type dt: float
    :param downsample: delay step in samples
    :type downsample: int
    :param pattern: wave-packet pattern (cWB ``pattern``)
    :type pattern: int
    :param hist: optional list-like container to collect transformed samples
    :type hist: list | None
    :param mode: numba loop strategy, ``"parallel"`` or ``"time-major"``.
        Defaults to ``PYCWB_NUMBA_MAX_ENERGY_MODE`` or ``"parallel"``.
    :type mode: str | None
    :return: ``(new_tf_map, alp)``
    :rtype: tuple[TimeFrequencyMap, float]
    """
    pattern_int = abs(int(pattern))

    if not pattern_int:
        # complex path — delegate to JAX implementation
        if _HAS_JAX:
            return time_delay_max_energy(
                tf_map, dt, downsample=downsample, pattern=0, hist=hist
            )
        raise NotImplementedError(
            "time_delay_max_energy_numba: pattern=0 (complex path) requires JAX. "
            "Use pattern != 0 for the pure numba path."
        )

    if not _HAS_NUMBA:
        raise RuntimeError(
            f"time_delay_max_energy_numba requires numba but it is unavailable: {_NUMBA_IMPORT_ERROR}"
        )

    validate_time_delay_inputs(tf_map, dt, downsample, require_wavelet_api=False)

    # --- decode TF map → time series ---
    len_ts = time_series_length(tf_map)

    data_np = np.asarray(tf_map.data)
    n_freq = int(data_np.shape[0])
    wavelet_filter = np.asarray(tf_map.wavelet.filter, dtype=np.float64)

    # Use the stored original time series if available to avoid the w2t→t2w roundtrip,
    # which introduces meaningful numerical error for the WDM packet energy computation.
    if getattr(tf_map, "ts_data", None) is not None:
        ts_data = np.asarray(tf_map.ts_data, dtype=np.float64)
    else:
        re_map = data_np.real.astype(np.float64)
        im_map = data_np.imag.astype(np.float64)
        # flat layout: stack [re.T, im.T] → same format w2t_numba expects
        flat = np.stack([re_map.T, im_map.T], axis=0).reshape(-1).astype(np.float64)
        ts_data = _wdm_w2t_numba(flat, n_freq, wavelet_filter, output_length=len_ts)
        ts_data = np.asarray(ts_data, dtype=np.float64)

    # --- compute params ---
    sample_rate_val = sample_rate_from_tf_map(tf_map, n_freq)
    max_delay_samples = int(sample_rate_val * abs(float(dt)))
    mm_mode = -1  # always use quadrature (both +/- freq components) for pattern path

    wavelet_M = int(tf_map.wavelet.M)
    wavelet_m_H = int(tf_map.wavelet.m_H)
    numba_mode = _normalize_numba_max_energy_mode(
        os.getenv("PYCWB_NUMBA_MAX_ENERGY_MODE") if mode is None else mode
    )

    f_low, f_high = frequency_bounds(tf_map, n_freq)

    # --- main numba loop ---
    current_max = _time_delay_max_energy_pattern_nb(
        ts_data,
        wavelet_M,
        wavelet_m_H,
        wavelet_filter,
        max_delay_samples,
        int(downsample),
        mm_mode,
        pattern_int,
        float(tf_map.edge or 0.0),
        int(tf_map.wavelet_rate),
        f_low,
        f_high,
        float(tf_map.df),
        mode=numba_mode,
    )

    new_tf_map = dataclasses.replace(tf_map, data=current_max)
    result = new_tf_map.Gamma2Gauss(hist=hist)

    return new_tf_map, result
