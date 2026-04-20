"""
Pure-function implementations of the WDM time-delay max-energy kernel.

All JAX JIT functions and the public ``time_delay_max_energy`` entry point
live here.  No backward-compatibility shims: callers import directly from
this module.

JAX cache note
--------------
The wavelet object is **not** passed as a static arg (object identity would
cause a cache miss on every new ``WDMWavelet`` instance).  Instead the four
scalar constructor parameters ``wdm_M``, ``wdm_K``, ``wdm_beta_order``,
``wdm_precision`` are static ints/floats, giving stable cache keys across
job segments.  The wavelet is reconstructed inside the JIT body at trace
time (zero runtime cost after the first compile).
"""

import dataclasses
import math
import logging
import numpy as np
from functools import partial

from wdm_wavelet.wdm import WDM as WDMWavelet
from wdm_wavelet.wdm import t2w_jax as _wdm_t2w_jax
from wdm_wavelet.wdm import w2t_jax as _wdm_w2t_jax
# import the numba version
from wdm_wavelet.wdm import t2w_numba as _wdm_t2w_numba
from wdm_wavelet.wdm import w2t_numba as _wdm_w2t_numba

from pycwb.types.time_frequency_map import TimeFrequencyMap

try:
    from wdm_wavelet.core.t2w import _t2w_jax_impl as _wdm_t2w_jax_impl
except Exception:
    _wdm_t2w_jax_impl = None

try:
    from wdm_wavelet.core.t2w import t2w_numba_core as _wdm_t2w_numba_core
    from numba.core.registry import CPUDispatcher as _NumbaCPUDispatcher
    # Only usable from @njit if it was JIT-compiled (requires rocket-fft)
    _HAS_T2W_NUMBA_CORE = isinstance(_wdm_t2w_numba_core, _NumbaCPUDispatcher)
except Exception:
    _wdm_t2w_numba_core = None
    _HAS_T2W_NUMBA_CORE = False

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
    _JAX_IMPORT_ERROR = None
except Exception as exc:
    jax = None
    jnp = None
    _HAS_JAX = False
    _JAX_IMPORT_ERROR = exc

try:
    import numba
    _HAS_NUMBA = True
    _NUMBA_IMPORT_ERROR = None
except Exception as exc:
    numba = None
    _HAS_NUMBA = False
    _NUMBA_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JAX helper functions (defined only when JAX is available)
# ---------------------------------------------------------------------------

if _HAS_JAX:
    def _cpf_left(xx, ts, k):
        """Mimic C++ xx.cpf(ts, ts.size()-k, k, 0):
        copies ts[k:size] to xx[0:size-k]; xx[size-k:size] is left unchanged."""
        size = ts.shape[0]
        idx = jnp.arange(size, dtype=jnp.int32)
        new_val = ts[jnp.minimum(idx + k, size - 1)]
        return jnp.where(idx < size - k, new_val, xx)

    def _cpf_right(xx, ts, k):
        """Mimic C++ xx.cpf(ts, ts.size()-k, 0, k):
        copies ts[0:size-k] to xx[k:size]; xx[0:k] is left unchanged."""
        size = ts.shape[0]
        idx = jnp.arange(size, dtype=jnp.int32)
        new_val = ts[jnp.maximum(idx - k, 0)]
        return jnp.where(idx >= k, new_val, xx)

    def _t2w_data_jax(wdm_M, wdm_m_H, wavelet_filter, ts_data, mm_mode):
        """Transform time series to WDM TF map using JAX.

        Parameters are explicit scalars/arrays rather than a WDMWavelet object
        so this function can be called safely inside a ``jax.jit``-traced body.
        """
        if _wdm_t2w_jax_impl is None:
            # Fallback — only works outside JIT (ts_data must be concrete)
            filt_np = np.asarray(wavelet_filter)
            m_l, _, tf_map = _wdm_t2w_jax(
                int(wdm_M), int(wdm_m_H), np.asarray(ts_data), filt_np, int(mm_mode)
            )
            return jnp.asarray(tf_map[0] + 1j * tf_map[1], dtype=jnp.complex128).T

        M = int(wdm_M)
        n_filter_taps = int(wdm_m_H)
        mm_eff = M if int(mm_mode) <= 0 else int(mm_mode)
        return_quadrature = bool(int(mm_mode) < 0)

        n_input = int(ts_data.shape[0])
        aligned_length = ((n_input + mm_eff - 1) // mm_eff) * mm_eff
        n_time_bins = aligned_length // mm_eff

        ext_len = aligned_length + 2 * n_filter_taps
        extended_signal = jnp.zeros((ext_len,), dtype=jnp.float64)

        left_mirror_max = min(n_filter_taps, n_input - 1)
        if left_mirror_max >= 0:
            idx = jnp.arange(left_mirror_max + 1, dtype=jnp.int32)
            extended_signal = extended_signal.at[n_filter_taps - idx].set(ts_data[idx])

        extended_signal = extended_signal.at[n_filter_taps:n_filter_taps + n_input].set(ts_data)

        n_right = ext_len - n_filter_taps - n_input
        if n_right > 0:
            idx = jnp.arange(n_right, dtype=jnp.int32)
            extended_signal = extended_signal.at[n_filter_taps + n_input + idx].set(ts_data[n_input - idx - 1])

        # wavelet_filter may be a JAX dynamic array inside JIT — slice in JAX
        filter_taps = jnp.asarray(wavelet_filter, dtype=jnp.float64)[:n_filter_taps]

        tf_map = _wdm_t2w_jax_impl(
            M=M,
            n_filter_taps=n_filter_taps,
            mm_eff=mm_eff,
            return_quadrature=return_quadrature,
            n_time_bins=n_time_bins,
            extended_signal=extended_signal,
            filter_taps=filter_taps,
        )

        return (tf_map[0] + 1j * tf_map[1]).T

    def _w2t_data_jax(data_complex, wavelet, output_length):
        n_freq = int(data_complex.shape[0])
        flat = jnp.stack([jnp.real(data_complex).T, jnp.imag(data_complex).T], axis=0).reshape(-1)
        out = _wdm_w2t_jax(
            np.asarray(flat),
            n_freq,
            np.asarray(wavelet.filter),
            output_length=int(output_length),
        )
        return jnp.asarray(out, dtype=jnp.float64)

    @partial(jax.jit, static_argnames=("pattern", "edge", "wavelet_rate", "f_low", "f_high", "df"))
    def _wdm_packet_energy_jax(coeffs, pattern, edge, wavelet_rate, f_low, f_high, df):
        pattern = abs(int(pattern))
        complex_map = jnp.asarray(coeffs)
        if complex_map.ndim != 2:
            raise ValueError("wdm_packet expects a 2D time-frequency map")

        M = int(complex_map.shape[0])
        T = int(complex_map.shape[1])
        J = M * T

        edge_v = jnp.asarray(edge, dtype=jnp.float64)
        jb = (jnp.floor(edge_v * float(wavelet_rate) / 4.0)).astype(jnp.int32) * jnp.int32(M)
        jb = jnp.maximum(jb, jnp.int32(4 * M))
        je = jnp.int32(J) - jb

        f_low_v = jnp.asarray(f_low, dtype=jnp.float64)
        f_high_v = jnp.asarray(f_high, dtype=jnp.float64)
        df_v = jnp.asarray(df, dtype=jnp.float64)
        mL = jnp.floor(f_low_v / df_v + 0.1).astype(jnp.int32)
        mH = jnp.floor(f_high_v / df_v + 0.1).astype(jnp.int32)
        mL = jnp.maximum(mL, jnp.int32(0))
        mH = jnp.minimum(mH, jnp.int32(M - 1))

        if pattern in (1, 3, 4):
            mean = 3.0
            mL += 1
            mH -= 1
        elif pattern == 2:
            mean = 3.0
        elif pattern in (5, 6):
            mean = 5.0
            mL += 2
            mH -= 2
        elif pattern in (7, 8):
            mean = 5.0
            mL += 1
            mH -= 1
        elif pattern == 9:
            mean = 9.0
            mL += 1
            mH -= 1
        else:
            mean = 1.0

        p = [0] * 9
        if pattern == 1:
            p[1], p[2] = 1, -1
        elif pattern == 2:
            p[1], p[2] = M, -M
        elif pattern == 3:
            p[1], p[2] = M + 1, -M - 1
        elif pattern == 4:
            p[1], p[2] = -M + 1, M - 1
        elif pattern == 5:
            p[1], p[2], p[3], p[4] = M + 1, -M - 1, 2 * M + 2, -2 * M - 2
        elif pattern == 6:
            p[1], p[2], p[3], p[4] = -M + 1, M - 1, -2 * M + 2, 2 * M - 2
        elif pattern == 7:
            p[1], p[2], p[3], p[4] = 1, -1, M, -M
        elif pattern == 8:
            p[1], p[2], p[3], p[4] = M + 1, -M + 1, M - 1, -M - 1
        elif pattern == 9:
            p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8] = 1, -1, M, -M, M + 1, M - 1, -M + 1, -M - 1

        real_f = jnp.ravel(jnp.real(complex_map).T)
        imag_f = jnp.ravel(jnp.imag(complex_map).T)

        jb = jnp.maximum(jb, jnp.int32(0))
        je = jnp.minimum(je, jnp.int32(J))
        j = jnp.arange(J, dtype=jnp.int32)

        m = j % M
        band = (m >= mL) & (m <= mH)
        inside = (j >= jb) & (j < je)
        active = inside & band

        ss = jnp.zeros_like(j, dtype=jnp.float64)
        ee = jnp.zeros_like(j, dtype=jnp.float64)
        EE = jnp.zeros_like(j, dtype=jnp.float64)
        for n in range(1, 9):
            idx = jnp.clip(j + p[n], 0, J - 1)
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
        ss2 = ss * 2.0
        nn = jnp.sqrt(cc * cc + ss2 * ss2)
        sum_eeEE = ee + EE
        nn = jnp.where(sum_eeEE < nn, sum_eeEE, nn)

        a1 = jnp.sqrt(jnp.clip((sum_eeEE + nn) / 2.0, min=0.0))
        a2 = jnp.sqrt(jnp.clip((sum_eeEE - nn) / 2.0, min=0.0))
        aa = a1 + a2

        em = jnp.where(mean == 1.0, sum_eeEE / 2.0, (aa * aa) / 4.0)
        # C++ wdmPacket: boundary pixels j not in [jb, je) keep the raw 00-phase Forward
        # amplitude from resize(J).  After maxEnergy's "*this = max(0, tmp)" step, negative
        # amplitudes become 0, positive ones survive.  Match that behaviour here.
        boundary_raw = jnp.maximum(0.0, q0r)
        energy_f = jnp.where(active, em, jnp.where(inside, 0.0, boundary_raw))
        return jnp.reshape(energy_f, (T, M)).T

    @partial(jax.jit, static_argnames=(
        "mm_mode", "pattern", "edge", "wavelet_rate", "f_low", "f_high", "df", "coeff_shape",
        "wdm_M", "wdm_m_H",
    ))
    def _time_delay_max_energy_pattern_jit(
        ts_data, sample_rate, t0, downsample, max_delay,
        wavelet_filter,
        mm_mode, pattern, edge, wavelet_rate, f_low, f_high, df, coeff_shape,
        wdm_M, wdm_m_H,
    ):
        # wavelet_filter is a JAX dynamic array — never call WDMWavelet() here
        base_data = _t2w_data_jax(wdm_M, wdm_m_H, wavelet_filter, ts_data, mm_mode)
        current_max = _wdm_packet_energy_jax(base_data, pattern, edge, wavelet_rate, f_low, f_high, df)

        def cond_fn(state):
            k, _, _xx = state
            return jnp.logical_and(k <= max_delay, k < ts_data.shape[0])

        def body_fn(state):
            k, cur, xx = state

            # C++ cpf call 1: xx.cpf(ts, size-k, k, 0) — left-shift ts into xx
            xx = _cpf_left(xx, ts_data, k)
            tmp_left = _t2w_data_jax(wdm_M, wdm_m_H, wavelet_filter, xx, mm_mode)
            cur = jnp.maximum(cur, _wdm_packet_energy_jax(tmp_left, pattern, edge, wavelet_rate, f_low, f_high, df))

            # C++ cpf call 2: xx.cpf(ts, size-k, 0, k) — right-shift ts into tail of xx
            xx = _cpf_right(xx, ts_data, k)
            tmp_right = _t2w_data_jax(wdm_M, wdm_m_H, wavelet_filter, xx, mm_mode)
            cur = jnp.maximum(cur, _wdm_packet_energy_jax(tmp_right, pattern, edge, wavelet_rate, f_low, f_high, df))

            return k + downsample, cur, xx

        _, current_max, _ = jax.lax.while_loop(
            cond_fn, body_fn, (jnp.int32(downsample), current_max, jnp.array(ts_data))
        )

        # C++ zeros layer 0 only: after wdmPacket's resize+reset, M=tmp.maxLayer()+1=1,
        # so getLayer(xx,M-1=0) zeroes layer 0 again — the actual last layer is NOT zeroed.
        current_max = current_max.at[0, :].set(0.0)
        if pattern in (5, 6, 9) and current_max.shape[0] > 2:
            current_max = current_max.at[1, :].set(0.0)

        return current_max

    @partial(jax.jit, static_argnames=(
        "mm_mode", "coeff_shape",
        "wdm_M", "wdm_m_H",
    ))
    def _time_delay_max_energy_complex_jit(
        ts_data, sample_rate, t0, downsample, max_delay, mm_mode, coeff_shape,
        wavelet_filter,
        wdm_M, wdm_m_H,
    ):
        # wavelet_filter is a JAX dynamic array — never call WDMWavelet() here
        base_data = _t2w_data_jax(wdm_M, wdm_m_H, wavelet_filter, ts_data, mm_mode)
        current_max_real = jnp.array(jnp.real(base_data))
        current_max_imag = jnp.array(jnp.imag(base_data))

        def cond_fn(state):
            k, _, _, _xx = state
            return jnp.logical_and(k <= max_delay, k < ts_data.shape[0])

        def body_fn(state):
            k, cur_r, cur_i, xx = state

            # C++ cpf call 1: left-shift ts into xx
            xx = _cpf_left(xx, ts_data, k)
            tmp_left = _t2w_data_jax(wdm_M, wdm_m_H, wavelet_filter, xx, mm_mode)
            cur_r = jnp.maximum(cur_r, jnp.real(tmp_left))
            cur_i = jnp.maximum(cur_i, jnp.imag(tmp_left))

            # C++ cpf call 2: right-shift ts into tail of xx
            xx = _cpf_right(xx, ts_data, k)
            tmp_right = _t2w_data_jax(wdm_M, wdm_m_H, wavelet_filter, xx, mm_mode)
            cur_r = jnp.maximum(cur_r, jnp.real(tmp_right))
            cur_i = jnp.maximum(cur_i, jnp.imag(tmp_right))

            return k + downsample, cur_r, cur_i, xx

        _, current_max_real, current_max_imag, _ = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (jnp.int32(downsample), current_max_real, current_max_imag, jnp.array(ts_data)),
        )

        out = current_max_real + 1j * current_max_imag
        out = out.at[0, :].set(0.0)
        out = out.at[out.shape[0] - 1, :].set(0.0)
        return out

    @partial(jax.jit, static_argnames=("pattern", "edge", "wavelet_rate", "f_low", "f_high", "df"))
    def _time_delay_max_energy_phase_jit(base_data, sample_rate, downsample, max_delay,
                                          pattern, edge, wavelet_rate, f_low, f_high, df):
        """
        Phase-shift based time-delay max energy (JIT compiled).

        .. warning::
            **NOT USED IN PRODUCTION.** This function produces physically different
            results from the reference implementation. WDM is NOT a simple analytic
            signal representation, so a time-domain shift is NOT equivalent to a
            per-frequency-bin phase rotation. Using this function causes ``alp``
            values to be ~2-5% lower than reference values, leading to significantly
            fewer coherence pixels (e.g. level-4: 89 → 34 pixels). Do NOT call this
            function in ``time_delay_max_energy`` until it has been validated against
            the cWB C++ reference. See ``docs/dev/Plans_to_review.md``.

        Applies frequency-domain phase rotations to the complex WDM TF map for
        each candidate time delay, avoiding the per-delay w2t → time-shift →
        t2w round-trip of ``_time_delay_max_energy_pattern_jit``.

        For WDM layer *m* the phase factor for a delay of *k* samples is::

            exp(-i · 2π · (m · df) · (k / sample_rate))

        This is exact in the analytic-signal / narrow-band approximation used
        by cWB and equivalent to the time-domain shift for well-separated WDM
        frequency bands.

        Parameters
        ----------
        base_data : jnp.ndarray, complex128, shape (n_freq, n_time)
            Complex WDM TF map (real = 00-phase, imag = 90-phase).
        sample_rate : jnp.ndarray scalar (float64)
            Sample rate of the underlying time series in Hz.
        downsample : jnp.ndarray int32
            Delay step in samples.
        max_delay : jnp.ndarray int32
            Maximum delay in samples.
        pattern, edge, wavelet_rate, f_low, f_high, df : static args
            Forwarded to ``_wdm_packet_energy_jax``.
        """
        n_freq = base_data.shape[0]
        freq_bins = jnp.arange(n_freq, dtype=jnp.float64) * df

        current_max = _wdm_packet_energy_jax(base_data, pattern, edge, wavelet_rate, f_low, f_high, df)

        def cond_fn(state):
            k, _ = state
            return k <= max_delay

        def body_fn(state):
            k, cur = state
            phase = jnp.exp(
                -1j * 2.0 * jnp.pi * freq_bins * (k.astype(jnp.float64) / sample_rate)
            )
            shifted_pos = base_data * phase[:, None]
            cur = jnp.maximum(
                cur,
                _wdm_packet_energy_jax(shifted_pos, pattern, edge, wavelet_rate, f_low, f_high, df),
            )
            shifted_neg = base_data * jnp.conj(phase)[:, None]
            cur = jnp.maximum(
                cur,
                _wdm_packet_energy_jax(shifted_neg, pattern, edge, wavelet_rate, f_low, f_high, df),
            )
            return k + downsample, cur

        _, current_max = jax.lax.while_loop(
            cond_fn, body_fn, (jnp.int32(downsample), current_max)
        )

        # C++ zeros layer 0 only: after wdmPacket's resize+reset, M=tmp.maxLayer()+1=1,
        # so getLayer(xx,M-1=0) zeroes layer 0 again — the actual last layer is NOT zeroed.
        current_max = current_max.at[0, :].set(0.0)
        if pattern in (5, 6, 9) and current_max.shape[0] > 2:
            current_max = current_max.at[1, :].set(0.0)

        return current_max

else:
    def _time_delay_max_energy_pattern_jit(*args, **kwargs):
        raise RuntimeError(f"JAX is required for time_delay_max_energy but is unavailable: {_JAX_IMPORT_ERROR}")

    def _time_delay_max_energy_complex_jit(*args, **kwargs):
        raise RuntimeError(f"JAX is required for time_delay_max_energy but is unavailable: {_JAX_IMPORT_ERROR}")

    def _time_delay_max_energy_phase_jit(*args, **kwargs):
        raise RuntimeError(f"JAX is required for time_delay_max_energy but is unavailable: {_JAX_IMPORT_ERROR}")

    def _wdm_packet_energy_jax(*args, **kwargs):
        raise RuntimeError(f"JAX is required for time_delay_max_energy but is unavailable: {_JAX_IMPORT_ERROR}")

    def _w2t_data_jax(*args, **kwargs):
        raise RuntimeError(f"JAX is required for time_delay_max_energy but is unavailable: {_JAX_IMPORT_ERROR}")


# ---------------------------------------------------------------------------
# Numba helper functions (defined only when numba is available)
# ---------------------------------------------------------------------------

def _compute_packet_energy_params(M, T, pattern, edge, wavelet_rate, f_low, f_high, df):
    """Pre-compute bounds and neighbor offset table for _wdm_packet_energy_nb."""
    J = M * T
    jb = int(edge * float(wavelet_rate) / 4.0) * M
    jb = max(jb, 4 * M)
    je = J - jb
    jb = max(jb, 0)
    je = min(je, J)

    mL = int(f_low / df + 0.1)
    mH = int(f_high / df + 0.1)
    mL = max(mL, 0)
    mH = min(mH, M - 1)

    pattern = abs(int(pattern))
    if pattern in (1, 3, 4):
        mean = 3.0
        mL += 1
        mH -= 1
    elif pattern == 2:
        mean = 3.0
    elif pattern in (5, 6):
        mean = 5.0
        mL += 2
        mH -= 2
    elif pattern in (7, 8):
        mean = 5.0
        mL += 1
        mH -= 1
    elif pattern == 9:
        mean = 9.0
        mL += 1
        mH -= 1
    else:
        mean = 1.0

    p = np.zeros(9, dtype=np.int64)
    if pattern == 1:
        p[1], p[2] = 1, -1
    elif pattern == 2:
        p[1], p[2] = M, -M
    elif pattern == 3:
        p[1], p[2] = M + 1, -M - 1
    elif pattern == 4:
        p[1], p[2] = -M + 1, M - 1
    elif pattern == 5:
        p[1], p[2], p[3], p[4] = M + 1, -M - 1, 2 * M + 2, -2 * M - 2
    elif pattern == 6:
        p[1], p[2], p[3], p[4] = -M + 1, M - 1, -2 * M + 2, 2 * M - 2
    elif pattern == 7:
        p[1], p[2], p[3], p[4] = 1, -1, M, -M
    elif pattern == 8:
        p[1], p[2], p[3], p[4] = M + 1, -M + 1, M - 1, -M - 1
    elif pattern == 9:
        p[1:9] = [1, -1, M, -M, M + 1, M - 1, -M + 1, -M - 1]

    return jb, je, mL, mH, mean, p


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

    @numba.njit(cache=True, parallel=True, fastmath=True)
    def _time_delay_max_energy_pattern_loop_nb(
        ts_data, filt, n_filter_taps, MM_eff, M_int, return_quadrature,
        max_delay, downsample,
        M_val, T_val, J, jb, je, mL, mH, mean, p,
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
        _, tf0 = _wdm_t2w_numba_core(ts_data, filt, n_filter_taps, MM_eff, M_int, return_quadrature)
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
                xx_l[:size - k_i] = ts_data[k_i:]
                _, tf_l = _wdm_t2w_numba_core(xx_l, filt, n_filter_taps, MM_eff, M_int, return_quadrature)
                all_en[2 * i] = _wdm_packet_energy_nb(
                    tf_l[0].ravel(), tf_l[1].ravel(), M_val, T_val, J, jb, je, mL, mH, mean, p
                )

                # cpf_right: copy ts_data[0:size-k_i] into xx[k_i:]; head stays = ts_data head
                xx_r = ts_data.copy()
                xx_r[k_i:] = ts_data[:size - k_i]
                _, tf_r = _wdm_t2w_numba_core(xx_r, filt, n_filter_taps, MM_eff, M_int, return_quadrature)
                all_en[2 * i + 1] = _wdm_packet_energy_nb(
                    tf_r[0].ravel(), tf_r[1].ravel(), M_val, T_val, J, jb, je, mL, mH, mean, p
                )

            # Sequential reduction (trivially fast — dominates nothing)
            for i in range(n_iters * 2):
                current_max = np.maximum(current_max, all_en[i])

        # C++ zeros layer 0 only
        current_max[0, :] = 0.0
        if pattern in (5, 6, 9) and current_max.shape[0] > 2:
            current_max[1, :] = 0.0

        return current_max

    def _time_delay_max_energy_pattern_nb(
        ts_data, wavelet_M, wavelet_m_H, wavelet_filter,
        max_delay, downsample, mm_mode,
        pattern, edge, wavelet_rate, f_low, f_high, df,
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
            return _time_delay_max_energy_pattern_loop_nb(
                ts_data, filt, n_filter_taps, MM_eff, M_int, return_quadrature,
                int(max_delay), int(downsample),
                M_val, T_val, J, jb, je, mL, mH, mean, p,
                pattern_abs,
            )
        else:
            print("Warning: t2w_numba_core is not available; falling back to slower Python-level loop for time-delay max energy.")

        # ---- fallback: Python-level t2w_numba wrapper per iteration ----
        _, _, tf0 = _wdm_t2w_numba(wavelet_M, wavelet_m_H, ts_data, wavelet_filter, mm_mode)
        re0 = np.ascontiguousarray(tf0[0])
        im0 = np.ascontiguousarray(tf0[1])

        current_max = _wdm_packet_energy_nb(
            re0.ravel(), im0.ravel(), M_val, T_val, J, jb, je, mL, mH, mean, p
        )

        k = int(downsample)
        xx = ts_data.copy()
        size = len(ts_data)
        while k <= int(max_delay) and k < size:
            xx[:size - k] = ts_data[k:]
            _, _, tf_left = _wdm_t2w_numba(wavelet_M, wavelet_m_H, xx, wavelet_filter, mm_mode)
            re_left = np.ascontiguousarray(tf_left[0])
            im_left = np.ascontiguousarray(tf_left[1])
            en_left = _wdm_packet_energy_nb(
                re_left.ravel(), im_left.ravel(), M_val, T_val, J, jb, je, mL, mH, mean, p
            )
            current_max = np.maximum(current_max, en_left)

            xx[k:] = ts_data[:size - k]
            _, _, tf_right = _wdm_t2w_numba(wavelet_M, wavelet_m_H, xx, wavelet_filter, mm_mode)
            re_right = np.ascontiguousarray(tf_right[0])
            im_right = np.ascontiguousarray(tf_right[1])
            en_right = _wdm_packet_energy_nb(
                re_right.ravel(), im_right.ravel(), M_val, T_val, J, jb, je, mL, mH, mean, p
            )
            current_max = np.maximum(current_max, en_right)

            k += int(downsample)

        current_max[0, :] = 0.0
        if pattern_abs in (5, 6, 9) and current_max.shape[0] > 2:
            current_max[1, :] = 0.0

        return current_max  # shape (M, T)

else:
    def _wdm_packet_energy_nb(*args, **kwargs):
        raise RuntimeError(f"numba is required for the numba backend but is unavailable: {_NUMBA_IMPORT_ERROR}")

    def _time_delay_max_energy_pattern_nb(*args, **kwargs):
        raise RuntimeError(f"numba is required for the numba backend but is unavailable: {_NUMBA_IMPORT_ERROR}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def time_delay_max_energy(tf_map: TimeFrequencyMap, dt, downsample=1, pattern=0, hist=None):
    """
    Compute the delayed max-energy map for a TF series.

    Pure-function port of ``TimeFrequencyMap.time_delay_max_energy`` (now
    removed from the class).  Updates ``tf_map.data`` in place.

    :param tf_map: detector TF map
    :type tf_map: TimeFrequencyMap
    :param dt: max time delay in seconds
    :type dt: float
    :param downsample: delay step in samples (cWB ``N``)
    :type downsample: int
    :param pattern: wave-packet pattern (cWB ``pattern``)
    :type pattern: int
    :param hist: optional list-like container to collect transformed samples
    :type hist: list | None
    :return: tuple of (new_tf_map with processed data, gamma-to-Gauss scaling
             parameter ``ALP``) where ``ALP == 1.0`` when ``pattern == 0``.
    :rtype: tuple[TimeFrequencyMap, float]
    """
    if not hasattr(tf_map.wavelet, "t2w") or not hasattr(tf_map.wavelet, "w2t"):
        raise ValueError("time_delay_max_energy requires a WDM wavelet with t2w/w2t APIs")

    if not _HAS_JAX:
        raise RuntimeError(f"time_delay_max_energy JAX JIT path requires JAX/JAXLIB: {_JAX_IMPORT_ERROR}")

    if downsample <= 0:
        raise ValueError("downsample must be >= 1")

    if not np.isfinite(dt):
        raise ValueError("dt must be finite")

    if tf_map.len_timeseries is not None:
        len_ts = int(tf_map.len_timeseries)
    else:
        len_ts = max(1, int(round((tf_map.stop - tf_map.start) / tf_map.dt)))

    # Use the stored original time series if available to avoid the w2t→t2w roundtrip,
    # which introduces meaningful numerical error for the WDM packet energy computation.
    if getattr(tf_map, 'ts_data', None) is not None:
        ts_data = jnp.asarray(tf_map.ts_data, dtype=jnp.float64)
    else:
        ts_data = _w2t_data_jax(jnp.asarray(tf_map.data, dtype=jnp.complex128), tf_map.wavelet, len_ts)

    sample_rate_val = float(2.0 * float(tf_map.df) * (int(np.asarray(tf_map.data).shape[0]) - 1))
    sample_rate = jnp.asarray(sample_rate_val, dtype=jnp.float64)
    t0 = jnp.asarray(float(tf_map.start), dtype=jnp.float64)

    max_delay = jnp.int32(int(sample_rate_val * abs(float(dt))))
    downsample_val = jnp.int32(int(downsample))
    pattern_int = abs(int(pattern))
    mm_mode = -1 if pattern_int else 0

    wdm_M = int(tf_map.wavelet.M)
    wdm_m_H = int(tf_map.wavelet.m_H)
    # Pre-compute filter outside JIT as a concrete JAX array.
    # WDMWavelet.filter is a numpy array — converting here is safe and avoids
    # TracerArrayConversionError that occurs when WDMWavelet() is constructed
    # inside a jax.jit-traced function (its __post_init__ calls np.asarray on
    # a JAX-produced filter array).
    wavelet_filter_jax = jnp.asarray(np.asarray(tf_map.wavelet.filter), dtype=jnp.float64)

    if pattern_int:
        f_low = 0.0 if tf_map.f_low is None else float(tf_map.f_low)
        f_high = (float(tf_map.df) * (int(np.asarray(tf_map.data).shape[0]) - 1)) if tf_map.f_high is None else float(tf_map.f_high)

        current_max = _time_delay_max_energy_pattern_jit(
            ts_data,
            sample_rate,
            t0,
            downsample_val,
            max_delay,
            wavelet_filter_jax,
            mm_mode,
            pattern_int,
            float(tf_map.edge or 0.0),
            int(tf_map.wavelet_rate),
            f_low,
            f_high,
            float(tf_map.df),
            tuple(np.asarray(tf_map.data).shape),
            wdm_M,
            wdm_m_H,
        )

        new_tf_map = dataclasses.replace(tf_map, data=np.asarray(current_max))
        result = new_tf_map.Gamma2Gauss(hist=hist)
        return new_tf_map, result

    max_complex = _time_delay_max_energy_complex_jit(
        ts_data,
        sample_rate,
        t0,
        downsample_val,
        max_delay,
        mm_mode,
        tuple(np.asarray(tf_map.data).shape),
        wavelet_filter_jax,
        wdm_M,
        wdm_m_H,
    )

    new_tf_map = dataclasses.replace(tf_map, data=np.asarray(max_complex))
    return new_tf_map, 1.0


def time_delay_max_energy_numba(tf_map: TimeFrequencyMap, dt, downsample=1, pattern=0, hist=None):
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
    :return: ``(new_tf_map, alp)``
    :rtype: tuple[TimeFrequencyMap, float]
    """
    pattern_int = abs(int(pattern))

    if not pattern_int:
        # complex path — delegate to JAX implementation
        if _HAS_JAX:
            return time_delay_max_energy(tf_map, dt, downsample=downsample, pattern=0, hist=hist)
        raise NotImplementedError(
            "time_delay_max_energy_numba: pattern=0 (complex path) requires JAX. "
            "Use pattern != 0 for the pure numba path."
        )

    if not _HAS_NUMBA:
        raise RuntimeError(
            f"time_delay_max_energy_numba requires numba but it is unavailable: {_NUMBA_IMPORT_ERROR}"
        )

    if downsample <= 0:
        raise ValueError("downsample must be >= 1")
    if not np.isfinite(dt):
        raise ValueError("dt must be finite")

    # --- decode TF map → time series ---
    if tf_map.len_timeseries is not None:
        len_ts = int(tf_map.len_timeseries)
    else:
        len_ts = max(1, int(round((tf_map.stop - tf_map.start) / tf_map.dt)))

    data_np = np.asarray(tf_map.data)
    n_freq = int(data_np.shape[0])
    wavelet_filter = np.asarray(tf_map.wavelet.filter, dtype=np.float64)

    # Use the stored original time series if available to avoid the w2t→t2w roundtrip,
    # which introduces meaningful numerical error for the WDM packet energy computation.
    if getattr(tf_map, 'ts_data', None) is not None:
        ts_data = np.asarray(tf_map.ts_data, dtype=np.float64)
    else:
        re_map = data_np.real.astype(np.float64)
        im_map = data_np.imag.astype(np.float64)
        # flat layout: stack [re.T, im.T] → same format w2t_numba expects
        flat = np.stack([re_map.T, im_map.T], axis=0).reshape(-1).astype(np.float64)
        ts_data = _wdm_w2t_numba(flat, n_freq, wavelet_filter, output_length=len_ts)
        ts_data = np.asarray(ts_data, dtype=np.float64)

    # --- compute params ---
    sample_rate_val = float(2.0 * float(tf_map.df) * (n_freq - 1))
    max_delay_samples = int(sample_rate_val * abs(float(dt)))
    mm_mode = -1  # always use quadrature (both +/- freq components) for pattern path

    wavelet_M = int(tf_map.wavelet.M)
    wavelet_m_H = int(tf_map.wavelet.m_H)

    f_low = 0.0 if tf_map.f_low is None else float(tf_map.f_low)
    f_high = (float(tf_map.df) * (n_freq - 1)) if tf_map.f_high is None else float(tf_map.f_high)

    # --- main numba loop ---
    current_max = _time_delay_max_energy_pattern_nb(
        ts_data, wavelet_M, wavelet_m_H, wavelet_filter,
        max_delay_samples, int(downsample), mm_mode,
        pattern_int,
        float(tf_map.edge or 0.0),
        int(tf_map.wavelet_rate),
        f_low, f_high,
        float(tf_map.df),
    )

    new_tf_map = dataclasses.replace(tf_map, data=current_max)
    result = new_tf_map.Gamma2Gauss(hist=hist)

    return new_tf_map, result
