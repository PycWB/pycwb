"""
Pure-Python regression without ROOT dependencies.

This module provides Python-native implementations of cWB regression
using TimeFrequencyMap methods and NumPy operations.
"""

import logging
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import config as jax_config
from wdm_wavelet.wdm import WDM

jax_config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)


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
    n = arr.shape[0]
    ff = jnp.clip(jnp.abs(fraction), 0.0, 1.0)
    nn = jnp.maximum(0, edge_samples)
    idx = jnp.arange(n)
    use_full = (nn == 0) | (2 * nn >= n - 2)
    core_mask = (idx >= nn) & (idx < (n - nn))
    core_mask = jnp.where(use_full, jnp.ones_like(core_mask, dtype=bool), core_mask)
    core_count = jnp.sum(core_mask)
    mean_all = jnp.mean(arr)

    def _compute(_):
        keep = jnp.maximum(1, (core_count * ff).astype(jnp.int32))
        keep = jnp.minimum(keep, core_count)
        abs_vals = jnp.where(core_mask, jnp.abs(arr), jnp.inf)
        threshold = jnp.sort(abs_vals)[keep - 1]
        select = core_mask & (abs_vals <= threshold)
        select_count = jnp.sum(select)
        select_sum = jnp.sum(jnp.where(select, arr, 0.0))
        return jnp.where(select_count > 0, select_sum / select_count, mean_all)

    return jax.lax.cond(core_count > 0, _compute, lambda _: mean_all, operand=None)


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


@partial(jax.jit, static_argnames=("K", "K2", "K4", "half"))
def _jax_process_one_layer(real, imag, K, K2, K4, half, fm, edge_samples, fltr,
                           eigen_threshold, eigen_num, regulator_code,
                           apply_threshold, rate_tf, edge_seconds):
    """
    Process one TF layer end-to-end in JAX.

    Steps: build statistics -> construct matrix -> solve regularized system ->
    apply filter -> threshold by non-edge RMS -> return predicted noise layer.
    """
    n_time = real.shape[0]
    power = real * real + imag * imag
    norm0_sq = _jax_percentile_mean(power, fm, edge_samples)
    norm0 = jnp.sqrt(norm0_sq)
    valid_norm = jnp.isfinite(norm0) & (norm0 > 0)
    safe_norm = jnp.where(valid_norm, norm0, 1.0)

    # Build cWB cross vector V over lags [-K, K].
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

    # Build cWB autocorrelation/cross-correlation vectors over [-2K, 2K].
    def _build_acf(i, state):
        acf, ccf = state
        lag = i - K2
        ww, WW = _jax_rotated_products(real, imag, lag, K2)
        idx = lag + K2
        base = safe_norm * safe_norm
        acf = acf.at[idx].set(_jax_percentile_mean(ww, fm, edge_samples) / base)
        ccf = ccf.at[idx].set(_jax_percentile_mean(WW, fm, edge_samples) / base)
        return acf, ccf

    v_cross = jax.lax.fori_loop(0, 2 * K + 1, _build_v, jnp.zeros((K4,), dtype=jnp.float64))
    acf, ccf = jax.lax.fori_loop(
        0,
        4 * K + 1,
        _build_acf,
        (jnp.zeros((4 * K + 1,), dtype=jnp.float64), jnp.zeros((4 * K + 1,), dtype=jnp.float64)),
    )

    # Solve regularized linear system in eigen basis.
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

    # Apply filters using explicit sliding windows in JAX.
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

    # RMS-based gate over non-edge region, matching cWB threshold behavior.
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


@partial(jax.jit, static_argnames=("K", "K2", "K4", "half"))
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

    # Match cWB defaults from schema/regression.cc
    filter_length = int(getattr(config, 'REGRESSION_FILTER_LENGTH', 8))
    apply_threshold = float(getattr(config, 'REGRESSION_APPLY_THR', 0.8))
    matrix_fraction = float(getattr(config, 'REGRESSION_MATRIX_FRACTION', 0.95))
    eigen_threshold = float(getattr(config, 'REGRESSION_SOLVE_EIGEN_THR', 0.0))
    eigen_num = int(getattr(config, 'REGRESSION_SOLVE_EIGEN_NUM', 10))
    regulator = str(getattr(config, 'REGRESSION_SOLVE_REGULATOR', 'h')).lower()
    if regulator not in ('h', 's', 'm'):
        regulator = 'h'

    if not isinstance(h, pycbc.types.TimeSeries):
        h_ts = pycbc.types.TimeSeries(
            h.value if hasattr(h, 'value') else h.data,
            delta_t=h.dt if hasattr(h, 'dt') else h.sample_rate**-1,
            epoch=h.t0 if hasattr(h, 't0') else h.start_time,
        )
    else:
        h_ts = h

    if filter_length <= 0:
        logger.info("Regression: pass-through mode (REGRESSION_FILTER_LENGTH <= 0)")
        return h_ts

    layers = int(config.rateANA / 8)
    beta_order = getattr(config, 'WDM_beta_order', 6)
    precision = getattr(config, 'WDM_precision', 10)
    f_high = float(config.fHigh)
    sample_rate = float(h_ts.sample_rate)
    edge_seconds = float(getattr(config, 'segEdge', 0.0))

    logger.info(f"Regression: cWB-LPE mode (K={filter_length}, thr={apply_threshold})")

    wdm = WDM(M=layers, K=layers, beta_order=beta_order, precision=precision)
    signal_data = np.array(h_ts.data, dtype=np.float64)
    t0 = float(h_ts.start_time)
    tf_map = wdm.t2w(signal_data, sample_rate=sample_rate, t0=t0, MM=-1)

    coeff = np.asarray(tf_map.data, dtype=np.complex128)
    if coeff.ndim != 2:
        logger.warning("  TF data not 2D - returning pass-through")
        return h_ts

    n_freq, n_time = coeff.shape
    logger.info(f"  TF map shape: {coeff.shape}")

    if n_freq < 3 or n_time <= 2 * filter_length + 2:
        logger.warning("  TF map too small for regression - returning pass-through")
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
        logger.warning("  No TF layers selected by frequency mask - returning pass-through")
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
    real_layers = jnp.asarray(np.asarray(coeff[selected_layers_arr].real, dtype=np.float64))
    imag_layers = jnp.asarray(np.asarray(coeff[selected_layers_arr].imag, dtype=np.float64))

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

    # Bring results back to NumPy for downstream WDM/PyCBC integration.
    noise_layers = np.asarray(noise_layers_jax)
    include_mask = np.asarray(include_mask_jax, dtype=bool)
    included_layers = int(np.sum(include_mask))
    noise_coeff[selected_layers_arr] = noise_layers

    if included_layers == 0:
        logger.info("  Regression complete: no layers passed threshold")
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

    logger.info(f"  Regression complete: {len(cleaned_pycbc)} output samples")
    return cleaned_pycbc

