"""
Pure-Python regression without ROOT dependencies.

This module provides Python-native implementations of cWB regression
using TimeFrequencyMap methods and NumPy operations.
"""

import os
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import config as jax_config
from wdm_wavelet.wdm import WDM

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False


_PERCENTILE_STRIDE = max(1, int(os.getenv("PYCWB_REGRESSION_PERCENTILE_STRIDE", "1")))


def _parse_jax_version(version_str):
    """Parse JAX version string into (major, minor, patch)."""
    parts = version_str.split(".")
    vals = []
    for part in parts[:3]:
        num = ""
        for ch in part:
            if ch.isdigit():
                num += ch
            else:
                break
        vals.append(int(num) if num else 0)
    while len(vals) < 3:
        vals.append(0)
    return tuple(vals)


_JAX_VERSION = _parse_jax_version(jax.__version__)
_USE_NEWER_JAX_BEHAVIOR = _JAX_VERSION >= (0, 7, 0)


@jax.jit
def _jax_eigh(matrix):
    """Compute eigendecomposition and return eigenpairs sorted descending."""
    evals, evecs = jnp.linalg.eigh(matrix)
    order = jnp.argsort(evals)[::-1]
    return evals[order], evecs[:, order]


@jax.jit
def _jax_apply_filters(wq, wQ, filt00, filt90):
    """Apply 00/90 regression filters to stacked sliding windows."""
    val_core = wq @ filt00 - wQ @ filt90
    valq_core = wq @ filt90 + wQ @ filt00
    return val_core, valq_core

def _jax_percentile_mean(arr, fraction, edge_samples):
    """
    JAX equivalent of cWB percentile mean used in matrix/vector statistics.

    For positive fractions, keeps the lowest-|x| fraction after optional
    edge trimming. Implemented without Python control flow so it is JIT-safe.
    """
    stride = _PERCENTILE_STRIDE
    if stride > 1:
        arr = arr[::stride]
        edge_samples = edge_samples // stride

    n = arr.shape[0]
    ff = float(np.clip(abs(fraction), 0.0, 1.0))
    nn = max(0, int(edge_samples))

    if nn == 0 or 2 * nn >= n - 2:
        core = arr
    else:
        core = arr[nn:n - nn]

    core_count = core.shape[0]
    mean_all = jnp.mean(arr)
    if core_count <= 0:
        return mean_all

    keep = int(core_count * ff)
    keep = max(1, min(keep, core_count))
    if keep >= core_count:
        return jnp.mean(core)

    abs_core = jnp.abs(core)
    if _USE_NEWER_JAX_BEHAVIOR:
        threshold = jnp.partition(abs_core, keep - 1)[keep - 1]
    else:
        threshold = jnp.sort(abs_core)[keep - 1]
    select = abs_core <= threshold
    select_count = jnp.sum(select)
    select_sum = jnp.sum(jnp.where(select, core, 0.0))
    return jnp.where(select_count > 0, select_sum / select_count, mean_all)


def _jax_rotated_products(real, imag, lag, boundary):
    """
    Compute rotated products (ww, WW) at one lag, in JAX.

    Mirrors the cWB ww/WW definitions used to build cross/autocorrelation
    statistics in the regression solver.
    """
    n = real.shape[0]
    j = jnp.arange(boundary, n - boundary)

    def _neg(_):
        rn = real[j]
        in_ = imag[j]
        jm = j - lag
        rm = real[jm]
        im = imag[jm]
        return rn, in_, rm, im

    def _pos(_):
        jn = j + lag
        rn = real[jn]
        in_ = imag[jn]
        rm = real[j]
        im = imag[j]
        return rn, in_, rm, im

    rn, in_, rm, im = jax.lax.cond(lag < 0, _neg, _pos, operand=None)
    ww = rn * rm + in_ * im
    WW = im * rn - rm * in_
    return ww, WW


def _jax_build_matrix(acf, ccf, K, K2, fltr):
    """
    Build the real block matrix from ACF/CCF vectors for one TF layer.

    Matrix layout matches the cWB single-witness LPE system.
    """
    ii = jnp.arange(-K, K + 1)
    jj = jnp.arange(-K, K + 1)
    lag_idx = ii[:, None] - jj[None, :] + K2

    aa = acf[lag_idx]
    cc = ccf[lag_idx]
    zero_mask = (ii[:, None] == 0) | (jj[None, :] == 0)
    aa = jnp.where(zero_mask, aa * fltr, aa)
    cc = jnp.where(zero_mask, cc * fltr, cc)

    top = jnp.concatenate([aa, cc], axis=1)
    bottom = jnp.concatenate([-cc, aa], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _numba_percentile_mean(arr, fraction, edge_samples, stride):
        ff = abs(fraction)
        if ff > 1.0:
            ff = 1.0

        if stride > 1:
            arr2 = arr[::stride]
            nn = edge_samples // stride
        else:
            arr2 = arr
            nn = edge_samples

        n = arr2.shape[0]
        if n == 0:
            return 0.0

        if nn < 0:
            nn = 0

        if nn == 0 or 2 * nn >= n - 2:
            core = arr2
        else:
            core = arr2[nn:n - nn]

        core_count = core.shape[0]
        mean_all = np.mean(arr2)
        if core_count <= 0:
            return mean_all

        keep = int(core_count * ff)
        if keep < 1:
            keep = 1
        if keep > core_count:
            keep = core_count
        if keep >= core_count:
            return np.mean(core)

        abs_core = np.abs(core)
        threshold = np.sort(abs_core)[keep - 1]

        select_sum = 0.0
        select_count = 0
        for i in range(core_count):
            if abs_core[i] <= threshold:
                select_sum += core[i]
                select_count += 1

        if select_count > 0:
            return select_sum / select_count
        return mean_all


    @njit(cache=True)
    def _numba_rotated_products(real, imag, lag, boundary):
        n = real.shape[0]
        start = boundary
        end = n - boundary
        size = end - start

        ww = np.empty(size, dtype=np.float64)
        WW = np.empty(size, dtype=np.float64)

        if lag < 0:
            for i in range(size):
                j = start + i
                rn = real[j]
                in_ = imag[j]
                jm = j - lag
                rm = real[jm]
                im = imag[jm]
                ww[i] = rn * rm + in_ * im
                WW[i] = im * rn - rm * in_
        else:
            for i in range(size):
                j = start + i
                jn = j + lag
                rn = real[jn]
                in_ = imag[jn]
                rm = real[j]
                im = imag[j]
                ww[i] = rn * rm + in_ * im
                WW[i] = im * rn - rm * in_

        return ww, WW


    @njit(cache=True)
    def _numba_build_matrix(acf, ccf, K, K2, fltr):
        size = 2 * (2 * K + 1)
        matrix = np.zeros((size, size), dtype=np.float64)
        half = size // 2

        for ii in range(-K, K + 1):
            for jj in range(-K, K + 1):
                idx = ii - jj + K2
                aa = acf[idx]
                cc = ccf[idx]
                if ii == 0 or jj == 0:
                    aa = aa * fltr
                    cc = cc * fltr

                r = ii + K
                c = jj + K
                matrix[r, c] = aa
                matrix[r, c + half] = cc
                matrix[r + half, c] = -cc
                matrix[r + half, c + half] = aa

        return matrix


    @njit(cache=True)
    def _numba_process_one_layer(real, imag, K, K2, K4, half, fm, edge_samples, fltr,
                                 eigen_threshold, eigen_num, regulator_code,
                                 apply_threshold, rate_tf, edge_seconds, stride):
        n_time = real.shape[0]

        power = real * real + imag * imag
        norm0_sq = _numba_percentile_mean(power, fm, edge_samples, stride)
        norm0 = np.sqrt(norm0_sq)
        valid_norm = np.isfinite(norm0) and (norm0 > 0.0)
        safe_norm = norm0 if valid_norm else 1.0
        base = safe_norm * safe_norm

        v_cross = np.zeros((K4,), dtype=np.float64)
        for lag in range(-K, K + 1):
            ww, WW = _numba_rotated_products(real, imag, lag, K)
            idx = K + lag
            v0 = _numba_percentile_mean(ww, fm, edge_samples, stride) / base
            v1 = _numba_percentile_mean(WW, fm, edge_samples, stride) / base
            scale = fltr if lag == 0 else 1.0
            v_cross[idx] = v0 * scale
            v_cross[idx + half] = v1 * scale

        lag_count = 2 * K2 + 1
        acf = np.zeros((lag_count,), dtype=np.float64)
        ccf = np.zeros((lag_count,), dtype=np.float64)
        for lag in range(-K2, K2 + 1):
            ww, WW = _numba_rotated_products(real, imag, lag, K2)
            idx = lag + K2
            acf[idx] = _numba_percentile_mean(ww, fm, edge_samples, stride) / base
            ccf[idx] = _numba_percentile_mean(WW, fm, edge_samples, stride) / base

        matrix = _numba_build_matrix(acf, ccf, K, K2, fltr)
        evals, evecs = np.linalg.eigh(matrix)

        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

        th = (-eigen_threshold * evals[0]) if (eigen_threshold < 0.0) else (eigen_threshold + 1.0e-12)
        nlast = 0
        for i in range(K4):
            if evals[i] >= th:
                nlast = i
        if nlast < 1:
            nlast = 1

        ne = K4 if eigen_num <= 0 else (eigen_num - 1)
        if ne > K4 - 1:
            ne = K4 - 1
        if ne < 1:
            ne = 1
        if nlast > ne:
            nlast = ne

        last_s = (1.0 / evals[nlast]) if evals[nlast] > 0.0 else 0.0
        last_m = (1.0 / evals[0]) if evals[0] > 0.0 else 0.0
        if regulator_code == 1:
            last = last_s
        elif regulator_code == 2:
            last = last_m
        else:
            last = 0.0

        lam = np.empty((K4,), dtype=np.float64)
        for i in range(K4):
            inv = (1.0 / evals[i]) if evals[i] > 0.0 else 0.0
            lam[i] = inv if i <= nlast else last

        vv = np.dot(evecs.T, v_cross) * lam
        aa = np.dot(evecs, vv)
        filt00 = aa[:2 * K + 1]
        filt90 = aa[half:half + 2 * K + 1]

        qq = real / safe_norm
        QQ = imag / safe_norm
        nn = np.zeros((n_time,), dtype=np.float64)
        NN = np.zeros((n_time,), dtype=np.float64)

        for center in range(K, n_time - K):
            val = 0.0
            VAL = 0.0
            for k in range(-K, K + 1):
                coeff_idx = k + K
                x = center + k
                val += qq[x] * filt00[coeff_idx] - QQ[x] * filt90[coeff_idx]
                VAL += qq[x] * filt90[coeff_idx] + QQ[x] * filt00[coeff_idx]
            nn[center] = val
            NN[center] = VAL

        kk = int(rate_tf * edge_seconds)
        if kk < K:
            kk = K
        kk += 1
        if kk < 0:
            kk = 0
        if kk > n_time:
            kk = n_time
        s0 = kk
        s1 = n_time - kk
        if s1 < s0:
            s1 = s0
        valid_range = s1 > s0

        count = s1 - s0
        if count < 1:
            count = 1

        nn_mean = 0.0
        NN_mean = 0.0
        for i in range(s0, s1):
            nn_mean += nn[i]
            NN_mean += NN[i]
        nn_mean /= count
        NN_mean /= count

        nn_var = 0.0
        NN_var = 0.0
        for i in range(s0, s1):
            d0 = nn[i] - nn_mean
            d1 = NN[i] - NN_mean
            nn_var += d0 * d0
            NN_var += d1 * d1
        nn_var /= count
        NN_var /= count
        layer_power = nn_var + NN_var

        included = valid_norm and valid_range and (layer_power >= apply_threshold * apply_threshold)
        noise = np.zeros((n_time,), dtype=np.complex128)
        if included:
            for i in range(n_time):
                noise[i] = (nn[i] + 1j * NN[i]) * norm0

        return noise, included


    @njit(cache=True, parallel=True)
    def _numba_process_layers(real_layers, imag_layers, K, K2, K4, half, fm, edge_samples, fltr,
                              eigen_threshold, eigen_num, regulator_code,
                              apply_threshold, rate_tf, edge_seconds, stride):
        n_layers = real_layers.shape[0]
        n_time = real_layers.shape[1]
        noise_layers = np.zeros((n_layers, n_time), dtype=np.complex128)
        include_mask = np.zeros((n_layers,), dtype=np.bool_)

        for i in prange(n_layers):
            noise, included = _numba_process_one_layer(
                real_layers[i], imag_layers[i],
                K, K2, K4, half,
                fm, edge_samples, fltr,
                eigen_threshold, eigen_num, regulator_code,
                apply_threshold, rate_tf, edge_seconds,
                stride,
            )
            noise_layers[i, :] = noise
            include_mask[i] = included

        return noise_layers, include_mask

@partial(jax.jit, static_argnames=("K", "K2", "K4", "half", "fm", "edge_samples", "fltr"))
def _jax_layer_build_stats(real, imag, K, K2, K4, half, fm, edge_samples, fltr):
    """Build normalized vector/matrix statistics for one layer."""
    power = real * real + imag * imag
    norm0_sq = _jax_percentile_mean(power, fm, edge_samples)
    norm0 = jnp.sqrt(norm0_sq)
    valid_norm = jnp.isfinite(norm0) & (norm0 > 0)
    safe_norm = jnp.where(valid_norm, norm0, 1.0)

    v_cross = _jax_layer_build_v_cross(real, imag, safe_norm, K, K4, half, fm, edge_samples, fltr)
    acf, ccf = _jax_layer_build_acf_ccf(real, imag, safe_norm, K2, fm, edge_samples)
    return norm0, valid_norm, safe_norm, v_cross, acf, ccf


@partial(jax.jit, static_argnames=("K", "K4", "half", "fm", "edge_samples", "fltr"))
def _jax_layer_build_v_cross(real, imag, safe_norm, K, K4, half, fm, edge_samples, fltr):
    """Build cross vector V over lags [-K, K] for one layer."""

    def _build_v(i, v_cross):
        lag = i - K
        ww, WW = _jax_rotated_products(real, imag, lag, K)
        idx = K + lag
        base = safe_norm * safe_norm
        v0 = _jax_percentile_mean(ww, fm, edge_samples) / base
        v1 = _jax_percentile_mean(WW, fm, edge_samples) / base
        scale = jnp.where(lag == 0, fltr, 1.0)
        v_cross = v_cross.at[idx].set(v0 * scale)
        v_cross = v_cross.at[idx + half].set(v1 * scale)
        return v_cross

    return jax.lax.fori_loop(0, 2 * K + 1, _build_v, jnp.zeros((K4,), dtype=jnp.float64))


@partial(jax.jit, static_argnames=("K2", "fm", "edge_samples"))
def _jax_layer_build_acf_ccf(real, imag, safe_norm, K2, fm, edge_samples):
    """Build autocorrelation/cross-correlation vectors over lags [-2K, 2K]."""
    lag_count = 2 * K2 + 1

    def _build_acf(i, state):
        acf, ccf = state
        lag = i - K2
        ww, WW = _jax_rotated_products(real, imag, lag, K2)
        idx = lag + K2
        base = safe_norm * safe_norm
        acf = acf.at[idx].set(_jax_percentile_mean(ww, fm, edge_samples) / base)
        ccf = ccf.at[idx].set(_jax_percentile_mean(WW, fm, edge_samples) / base)
        return acf, ccf

    return jax.lax.fori_loop(
        0,
        lag_count,
        _build_acf,
        (jnp.zeros((lag_count,), dtype=jnp.float64), jnp.zeros((lag_count,), dtype=jnp.float64)),
    )



@partial(jax.jit, static_argnames=("K", "K2", "K4", "half", "fltr", "eigen_threshold", "eigen_num", "regulator_code"))
def _jax_layer_solve_filters(v_cross, acf, ccf, K, K2, K4, half, fltr,
                             eigen_threshold, eigen_num, regulator_code):
    """Solve regularized LPE system and return filter taps."""
    matrix = _jax_build_matrix(acf, ccf, K, K2, fltr)
    evals, evecs = _jax_eigh(matrix)

    th = jnp.where(eigen_threshold < 0, -eigen_threshold * evals[0], eigen_threshold + 1.0e-12)
    nlast = jnp.sum(evals >= th) - 1
    nlast = jnp.maximum(nlast, 1)

    ne = jnp.where(eigen_num <= 0, K4, eigen_num - 1)
    ne = jnp.minimum(ne, K4 - 1)
    ne = jnp.maximum(ne, 1)
    nlast = jnp.minimum(nlast, ne)

    last_s = jnp.where(evals[nlast] > 0, 1.0 / evals[nlast], 0.0)
    last_m = jnp.where(evals[0] > 0, 1.0 / evals[0], 0.0)
    last = jnp.where(regulator_code == 1, last_s, jnp.where(regulator_code == 2, last_m, 0.0))

    idxs = jnp.arange(K4)
    inv_evals = jnp.where(evals > 0, 1.0 / evals, 0.0)
    lam = jnp.where(idxs <= nlast, inv_evals, last)

    vv = (evecs.T @ v_cross) * lam
    aa = evecs @ vv
    filt00 = aa[:2 * K + 1]
    filt90 = aa[half:half + 2 * K + 1]
    return filt00, filt90


@partial(jax.jit, static_argnames=("K",))
def _jax_layer_apply_filters(real, imag, safe_norm, filt00, filt90, K):
    """Apply solved filters over one normalized TF layer."""
    n_time = real.shape[0]
    qq = real / safe_norm
    QQ = imag / safe_norm
    centers = jnp.arange(K, n_time - K)

    def _window_at(center):
        start = center - K
        q_slice = jax.lax.dynamic_slice(qq, (start,), (2 * K + 1,))
        Q_slice = jax.lax.dynamic_slice(QQ, (start,), (2 * K + 1,))
        return q_slice, Q_slice

    wq, wQ = jax.vmap(_window_at)(centers)
    val_core, VAL_core = _jax_apply_filters(wq, wQ, filt00, filt90)

    nn = jnp.zeros((n_time,), dtype=jnp.float64).at[K:n_time - K].set(val_core)
    NN = jnp.zeros((n_time,), dtype=jnp.float64).at[K:n_time - K].set(VAL_core)
    return nn, NN


@partial(jax.jit, static_argnames=("K", "apply_threshold", "rate_tf", "edge_seconds"))
def _jax_layer_gate(nn, NN, norm0, valid_norm, apply_threshold, rate_tf, edge_seconds, K):
    """Apply cWB-like RMS threshold gate and build complex noise layer."""
    n_time = nn.shape[0]
    kk = jnp.int32(rate_tf * edge_seconds)
    kk = jnp.maximum(kk, K)
    kk = kk + 1
    s0 = jnp.minimum(jnp.maximum(kk, 0), n_time)
    s1 = jnp.maximum(s0, n_time - kk)
    valid_range = s1 > s0

    tidx = jnp.arange(n_time)
    mask = (tidx >= s0) & (tidx < s1)
    count = jnp.maximum(jnp.sum(mask), 1)

    nn_mean = jnp.sum(jnp.where(mask, nn, 0.0)) / count
    NN_mean = jnp.sum(jnp.where(mask, NN, 0.0)) / count
    nn_var = jnp.sum(jnp.where(mask, (nn - nn_mean) ** 2, 0.0)) / count
    NN_var = jnp.sum(jnp.where(mask, (NN - NN_mean) ** 2, 0.0)) / count
    layer_power = nn_var + NN_var

    included = valid_norm & valid_range & (layer_power >= apply_threshold * apply_threshold)
    noise = (nn + 1j * NN) * norm0
    noise = jnp.where(included, noise, jnp.zeros_like(noise))
    return noise, included


@partial(
    jax.jit,
    static_argnames=(
        "K", "K2", "K4", "half",
        "fm", "edge_samples", "fltr",
        "eigen_threshold", "eigen_num", "regulator_code",
        "apply_threshold", "rate_tf", "edge_seconds",
    ),
)
def _jax_process_one_layer(real, imag, K, K2, K4, half, fm, edge_samples, fltr,
                           eigen_threshold, eigen_num, regulator_code,
                           apply_threshold, rate_tf, edge_seconds):
    """
    Process one TF layer end-to-end in JAX.

    Steps: build statistics -> construct matrix -> solve regularized system ->
    apply filter -> threshold by non-edge RMS -> return predicted noise layer.
    """
    norm0, valid_norm, safe_norm, v_cross, acf, ccf = _jax_layer_build_stats(
        real, imag, K, K2, K4, half, fm, edge_samples, fltr
    )
    filt00, filt90 = _jax_layer_solve_filters(
        v_cross, acf, ccf, K, K2, K4, half, fltr,
        eigen_threshold, eigen_num, regulator_code,
    )
    nn, NN = _jax_layer_apply_filters(real, imag, safe_norm, filt00, filt90, K)
    return _jax_layer_gate(nn, NN, norm0, valid_norm, apply_threshold, rate_tf, edge_seconds, K)


@partial(
    jax.jit,
    static_argnames=(
        "K", "K2", "K4", "half",
        "fm", "edge_samples", "fltr",
        "eigen_threshold", "eigen_num", "regulator_code",
        "apply_threshold", "rate_tf", "edge_seconds",
    ),
)
def _jax_process_layers(real_layers, imag_layers, K, K2, K4, half, fm, edge_samples, fltr,
                        eigen_threshold, eigen_num, regulator_code,
                        apply_threshold, rate_tf, edge_seconds):
    """Vectorized JAX execution of `_jax_process_one_layer` across layers."""
    return jax.vmap(
        _jax_process_one_layer,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None, None, None, None),
        out_axes=(0, 0),
    )(
        real_layers,
        imag_layers,
        K,
        K2,
        K4,
        half,
        fm,
        edge_samples,
        fltr,
        eigen_threshold,
        eigen_num,
        regulator_code,
        apply_threshold,
        rate_tf,
        edge_seconds,
    )


def regression_python(config, h):
    """
        Clean data with regression method (JAX-accelerated implementation).

    This follows the cWB LPE regression path used in `regression.py`:
        target TF map + self-witness ("target"), then setFilter/setMatrix/solve/apply.

        Notes
        -----
        - JAX is required and used for the heavy per-layer computation path.
        - Legacy NumPy helper functions are kept below for compatibility with
            `data_conditioning_python.py` imports.
    """
    import pycbc.types

    backend = str(getattr(config, "REGRESSION_ENGINE", os.getenv("PYCWB_REGRESSION_ENGINE", "numba"))).lower()

    # Match cWB defaults from schema/regression.cc
    filter_length = int(getattr(config, "REGRESSION_FILTER_LENGTH", 8))
    apply_threshold = float(getattr(config, "REGRESSION_APPLY_THR", 0.8))
    matrix_fraction = float(getattr(config, "REGRESSION_MATRIX_FRACTION", 0.95))
    eigen_threshold = float(getattr(config, "REGRESSION_SOLVE_EIGEN_THR", 0.0))
    eigen_num = int(getattr(config, "REGRESSION_SOLVE_EIGEN_NUM", 10))
    regulator = str(getattr(config, "REGRESSION_SOLVE_REGULATOR", "h")).lower()
    if regulator not in ("h", "s", "m"):
        regulator = "h"

    if not isinstance(h, pycbc.types.TimeSeries):
        h_ts = pycbc.types.TimeSeries(
            h.value if hasattr(h, 'value') else h.data,
            delta_t=h.dt if hasattr(h, 'dt') else h.sample_rate**-1,
            epoch=h.t0 if hasattr(h, 't0') else h.start_time,
        )
    else:
        h_ts = h

    if filter_length <= 0:
        return h_ts

    layers = int(config.rateANA / 8)
    beta_order = getattr(config, 'WDM_beta_order', 6)
    precision = getattr(config, 'WDM_precision', 10)
    f_high = float(config.fHigh)
    sample_rate = float(h_ts.sample_rate)
    edge_seconds = float(getattr(config, 'segEdge', 0.0))

    wdm = WDM(M=layers, K=layers, beta_order=beta_order, precision=precision)

    signal_data = np.array(h_ts.data, dtype=np.float64)
    t0 = float(h_ts.start_time)

    tf_map = wdm.t2w(signal_data, sample_rate=sample_rate, t0=t0, MM=-1)

    coeff = np.asarray(tf_map.data, dtype=np.complex128)

    if coeff.ndim != 2:
        return h_ts

    n_freq, n_time = coeff.shape

    if n_freq < 3 or n_time <= 2 * filter_length + 2:
        return h_ts

    df = float(getattr(tf_map, 'df', sample_rate / max(1.0, 2.0 * (n_freq - 1))))
    dt_tf = float(getattr(tf_map, 'dt', 1.0))
    rate_tf = 1.0 / dt_tf if dt_tf > 0 else 1.0

    # In cWB wrapper, constructor uses flow=1 and fhigh=config.fHigh for target.
    # setFilter then loops layer indices 1..maxLayer-1.
    flow_target = 1.0
    layer_freq = np.arange(n_freq, dtype=np.float64) * df
    selected_layers = [
        i for i in range(1, n_freq - 1)
        if flow_target <= layer_freq[i] <= f_high
    ]
    if not selected_layers:
        return h_ts

    K = filter_length
    K2 = 2 * K
    K4 = 2 * (2 * K + 1)
    half = K4 // 2
    fm = abs(matrix_fraction)
    edge_samples = int(max(0.0, edge_seconds) * rate_tf)

    # LPE path in cWB: witness has same channel name as target -> FLTR=0.
    fltr = 0.0

    noise_coeff = np.zeros_like(coeff, dtype=np.complex128)
    regulator_code = 1 if regulator == 's' else (2 if regulator == 'm' else 0)

    # Pack selected layers and run batched JAX processing in one call.
    selected_layers_arr = np.asarray(selected_layers, dtype=np.int32)
    real_layers_np = np.asarray(coeff[selected_layers_arr].real, dtype=np.float64)
    imag_layers_np = np.asarray(coeff[selected_layers_arr].imag, dtype=np.float64)

    use_numba = backend == "numba" and _NUMBA_AVAILABLE
    if use_numba:
        noise_layers, include_mask = _numba_process_layers(
            real_layers_np,
            imag_layers_np,
            K,
            K2,
            K4,
            half,
            fm,
            int(edge_samples),
            fltr,
            float(eigen_threshold),
            int(eigen_num),
            int(regulator_code),
            float(apply_threshold),
            float(rate_tf),
            float(edge_seconds),
            int(_PERCENTILE_STRIDE),
        )
    else:
        real_layers = jnp.asarray(real_layers_np)
        imag_layers = jnp.asarray(imag_layers_np)
        noise_layers_jax, include_mask_jax = _jax_process_layers(
            real_layers,
            imag_layers,
            K,
            K2,
            K4,
            half,
            fm,
            int(edge_samples),
            fltr,
            float(eigen_threshold),
            int(eigen_num),
            int(regulator_code),
            float(apply_threshold),
            float(rate_tf),
            float(edge_seconds),
        )
        noise_layers_jax = jax.block_until_ready(noise_layers_jax)
        include_mask_jax = jax.block_until_ready(include_mask_jax)
        noise_layers = np.asarray(noise_layers_jax)
        include_mask = np.asarray(include_mask_jax, dtype=bool)

    included_layers = int(np.sum(include_mask))
    noise_coeff[selected_layers_arr] = noise_layers

    if included_layers == 0:
        return h_ts

    # Reconstruct target and predicted noise in time domain, then clean.
    coeff_orig = coeff.copy()
    tf_map.data = coeff_orig

    target_ts = np.array(wdm.w2t(tf_map), dtype=np.float64)

    tf_map.data = noise_coeff
    noise_ts = np.array(wdm.w2t(tf_map), dtype=np.float64)

    noiseQ_ts = np.array(wdm.w2tQ(tf_map), dtype=np.float64)
    # cWB usage: combine two phase reconstructions (w2t=normal, w2tQ=quadrature)
    # The 0.5 factor averages the two channels as in WSeries Inverse() + Inverse(-2)
    cleaned_data = target_ts - 0.5 * (noise_ts + noiseQ_ts)

    cleaned_pycbc = pycbc.types.TimeSeries(
        cleaned_data,
        delta_t=h_ts.delta_t,
        epoch=h_ts.start_time
    )
    return cleaned_pycbc

