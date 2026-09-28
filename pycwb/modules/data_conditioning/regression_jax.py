"""JAX regression kernels, batched across selected WDM frequency layers.

Imported by regression.apply_regression when the JAX backend is selected or
Numba is unavailable. This module owns the JIT kernels rather than re-exporting
them through the public dispatcher.
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

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


def _jax_percentile_mean(arr, fraction, edge_samples, percentile_stride=1):
    """
    JAX equivalent of cWB percentile mean used in matrix/vector statistics.

    For positive fractions, keeps the lowest-|x| fraction after optional
    edge trimming. Implemented without Python control flow so it is JIT-safe.
    """
    stride = percentile_stride
    if stride > 1:
        arr = arr[::stride]
        edge_samples = edge_samples // stride

    n = arr.shape[0]
    ff = float(np.clip(abs(fraction), 0.0, 1.0))
    nn = max(0, int(edge_samples))

    if nn == 0 or 2 * nn >= n - 2:
        core = arr
    else:
        core = arr[nn : n - nn]

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


@partial(jax.jit, static_argnames=("fraction",))
def _cap_witness_jax(real, imag, fraction=1.0):
    """cWB 6.4.6.9 regression::_apply_ amplitude cap, including edge bins."""
    if fraction >= 1.0:
        return real, imag
    energy = real * real + imag * imag
    kth = int(fraction * real.size - 1)
    if not 0 <= kth < real.size:
        raise ValueError("Regression cap fraction has no valid order statistic")
    threshold = 5.0 * jnp.partition(energy, kth)[kth]
    scale = jnp.where(energy > threshold, jnp.sqrt(threshold / jnp.where(energy > 0, energy, 1.0)), 1.0)
    return real * scale, imag * scale


@partial(jax.jit, static_argnames=("K", "K2", "K4", "half", "fm", "edge_samples", "fltr", "percentile_stride"))
def _jax_layer_build_stats(real, imag, K, K2, K4, half, fm, edge_samples, fltr, percentile_stride=1):
    """Build normalized vector/matrix statistics for one layer."""
    power = real * real + imag * imag
    norm0_sq = _jax_percentile_mean(power, fm, edge_samples, percentile_stride)
    norm0 = jnp.sqrt(norm0_sq)
    valid_norm = jnp.isfinite(norm0) & (norm0 > 0)
    safe_norm = jnp.where(valid_norm, norm0, 1.0)

    v_cross = _jax_layer_build_v_cross(real, imag, safe_norm, K, K4, half, fm, edge_samples, fltr, percentile_stride)
    acf, ccf = _jax_layer_build_acf_ccf(real, imag, safe_norm, K2, fm, edge_samples, percentile_stride)
    return norm0, valid_norm, safe_norm, v_cross, acf, ccf


@partial(jax.jit, static_argnames=("K", "K4", "half", "fm", "edge_samples", "fltr", "percentile_stride"))
def _jax_layer_build_v_cross(real, imag, safe_norm, K, K4, half, fm, edge_samples, fltr, percentile_stride=1):
    """Build cross vector V over lags [-K, K] for one layer."""
    # ROOT fills products for j in [K, n-K] then trims edge_samples from the full array,
    # giving effective trim (edge_samples - K) on the product sub-array.
    edge_v = max(0, edge_samples - K)

    def _build_v(i, v_cross):
        lag = i - K
        ww, WW = _jax_rotated_products(real, imag, lag, K)
        idx = K + lag
        base = safe_norm * safe_norm
        v0 = _jax_percentile_mean(ww, fm, edge_v, percentile_stride) / base
        v1 = _jax_percentile_mean(WW, fm, edge_v, percentile_stride) / base
        scale = jnp.where(lag == 0, fltr, 1.0)
        v_cross = v_cross.at[idx].set(v0 * scale)
        v_cross = v_cross.at[idx + half].set(v1 * scale)
        return v_cross

    return jax.lax.fori_loop(0, 2 * K + 1, _build_v, jnp.zeros((K4,), dtype=jnp.float64))


@partial(jax.jit, static_argnames=("K2", "fm", "edge_samples", "percentile_stride"))
def _jax_layer_build_acf_ccf(real, imag, safe_norm, K2, fm, edge_samples, percentile_stride=1):
    """Build autocorrelation/cross-correlation vectors over lags [-2K, 2K]."""
    # ROOT fills products for j in [K2, n-K2] then trims edge_samples from the full array,
    # giving effective trim (edge_samples - K2) on the product sub-array.
    edge_m = max(0, edge_samples - K2)
    lag_count = 2 * K2 + 1

    def _build_acf(i, state):
        acf, ccf = state
        lag = i - K2
        ww, WW = _jax_rotated_products(real, imag, lag, K2)
        idx = lag + K2
        base = safe_norm * safe_norm
        # ROOT matrix WW = x_m*xQ_n - xQ_m*x_n (opposite sign from cross-vector/rotated_products WW)
        # negate to match ROOT's matrix formula
        acf = acf.at[idx].set(_jax_percentile_mean(ww, fm, edge_m, percentile_stride) / base)
        ccf = ccf.at[idx].set(-_jax_percentile_mean(WW, fm, edge_m, percentile_stride) / base)
        return acf, ccf

    return jax.lax.fori_loop(
        0,
        lag_count,
        _build_acf,
        (jnp.zeros((lag_count,), dtype=jnp.float64), jnp.zeros((lag_count,), dtype=jnp.float64)),
    )


@partial(jax.jit, static_argnames=("K", "K2", "K4", "half", "fltr", "eigen_threshold", "eigen_num", "regulator_code"))
def _jax_layer_solve_filters(v_cross, acf, ccf, K, K2, K4, half, fltr, eigen_threshold, eigen_num, regulator_code):
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
    filt00 = aa[: 2 * K + 1]
    filt90 = aa[half : half + 2 * K + 1]
    return filt00, filt90


@partial(jax.jit, static_argnames=("K", "apply_fraction"))
def _jax_layer_apply_filters(real, imag, safe_norm, filt00, filt90, K, apply_fraction=1.0):
    """Apply solved filters over one normalized TF layer."""
    n_time = real.shape[0]
    qq = real / safe_norm
    QQ = imag / safe_norm
    qq, QQ = _cap_witness_jax(qq, QQ, apply_fraction)
    centers = jnp.arange(K, n_time - K)

    def _window_at(center):
        start = center - K
        q_slice = jax.lax.dynamic_slice(qq, (start,), (2 * K + 1,))
        Q_slice = jax.lax.dynamic_slice(QQ, (start,), (2 * K + 1,))
        return q_slice, Q_slice

    wq, wQ = jax.vmap(_window_at)(centers)
    val_core, VAL_core = _jax_apply_filters(wq, wQ, filt00, filt90)

    nn = jnp.zeros((n_time,), dtype=jnp.float64).at[K : n_time - K].set(val_core)
    NN = jnp.zeros((n_time,), dtype=jnp.float64).at[K : n_time - K].set(VAL_core)
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
        "K",
        "K2",
        "K4",
        "half",
        "fm",
        "edge_samples",
        "fltr",
        "eigen_threshold",
        "eigen_num",
        "regulator_code",
        "apply_threshold",
        "rate_tf",
        "edge_seconds",
        "apply_fraction",
        "percentile_stride",
    ),
)
def _jax_process_one_layer(
    real,
    imag,
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
    apply_fraction=1.0,
    percentile_stride=1,
):
    """
    Process one TF layer end-to-end in JAX.

    Steps: build statistics -> construct matrix -> solve regularized system ->
    apply filter -> threshold by non-edge RMS -> return predicted noise layer.
    """
    norm0, valid_norm, safe_norm, v_cross, acf, ccf = _jax_layer_build_stats(
        real, imag, K, K2, K4, half, fm, edge_samples, fltr, percentile_stride
    )
    filt00, filt90 = _jax_layer_solve_filters(
        v_cross,
        acf,
        ccf,
        K,
        K2,
        K4,
        half,
        fltr,
        eigen_threshold,
        eigen_num,
        regulator_code,
    )
    nn, NN = _jax_layer_apply_filters(real, imag, safe_norm, filt00, filt90, K, apply_fraction)
    return _jax_layer_gate(nn, NN, norm0, valid_norm, apply_threshold, rate_tf, edge_seconds, K)


@partial(
    jax.jit,
    static_argnames=(
        "K",
        "K2",
        "K4",
        "half",
        "fm",
        "edge_samples",
        "fltr",
        "eigen_threshold",
        "eigen_num",
        "regulator_code",
        "apply_threshold",
        "rate_tf",
        "edge_seconds",
        "apply_fraction",
        "percentile_stride",
    ),
)
def _jax_process_layers(
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
    apply_fraction=1.0,
    percentile_stride=1,
):
    """Vectorized JAX execution of `_jax_process_one_layer` across layers."""
    return jax.vmap(
        _jax_process_one_layer,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
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
        apply_fraction,
        percentile_stride,
    )
