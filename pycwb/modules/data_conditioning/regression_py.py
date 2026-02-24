"""
Pure-Python regression without ROOT dependencies.

This module provides Python-native implementations of cWB regression
using TimeFrequencyMap methods and NumPy operations.
"""

import logging

import numpy as np
from wdm_wavelet.wdm import WDM

logger = logging.getLogger(__name__)


def regression_python(config, h):
    """
    Clean data with regression method (pure-Python implementation).

    This follows the cWB LPE regression path used in `regression.py`:
    target TF map + self-witness ("target"), then setFilter/setMatrix/solve/apply.
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
    included_layers = 0

    for layer in selected_layers:
        real = np.asarray(coeff[layer].real, dtype=np.float64)
        imag = np.asarray(coeff[layer].imag, dtype=np.float64)

        power = real * real + imag * imag
        norm0 = np.sqrt(_cwb_percentile_mean(power, fm, edge_samples))
        norm1 = norm0  # self-witness for LPE
        if not np.isfinite(norm0) or norm0 <= 0:
            continue

        # Build cross vector V and symmetric matrix M (single-witness case).
        v_cross = np.zeros(K4, dtype=np.float64)
        acf = np.zeros(4 * K + 1, dtype=np.float64)
        ccf = np.zeros(4 * K + 1, dtype=np.float64)

        for lag in range(-K, K + 1):
            ww, WW = _cwb_rotated_products(real, imag, lag, K)
            idx = K + lag
            v_cross[idx] = _cwb_percentile_mean(ww, fm, edge_samples) / (norm0 * norm1)
            v_cross[idx + half] = _cwb_percentile_mean(WW, fm, edge_samples) / (norm0 * norm1)
            if lag == 0:
                v_cross[idx] *= fltr
                v_cross[idx + half] *= fltr

        for lag in range(-K2, K2 + 1):
            ww, WW = _cwb_rotated_products(real, imag, lag, K2)
            idx = lag + K2
            acf[idx] = _cwb_percentile_mean(ww, fm, edge_samples) / (norm1 * norm1)
            ccf[idx] = _cwb_percentile_mean(WW, fm, edge_samples) / (norm1 * norm1)

        matrix = np.zeros((K4, K4), dtype=np.float64)
        for ii in range(-K, K + 1):
            row_r = ii + K
            row_i = row_r + half
            for jj in range(-K, K + 1):
                col_r = jj + K
                col_i = col_r + half
                lag_idx = ii - jj + K2
                aa = acf[lag_idx]
                cc = ccf[lag_idx]
                if ii == 0 or jj == 0:
                    aa *= fltr
                    cc *= fltr

                matrix[row_r, col_r] = aa
                matrix[col_r, row_r] = aa
                matrix[row_r, col_i] = cc
                matrix[col_i, row_r] = cc
                matrix[row_i, col_r] = -cc
                matrix[col_r, row_i] = -cc
                matrix[row_i, col_i] = aa
                matrix[col_i, row_i] = aa

        # cWB solve(): eigen decomposition + regulator in eigen basis.
        try:
            evals, evecs = np.linalg.eigh(matrix)
        except np.linalg.LinAlgError:
            continue

        # ROOT implementation expects eigenvalues sorted descending.
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

        th = (-eigen_threshold * evals[0]) if eigen_threshold < 0 else (eigen_threshold + 1.0e-12)
        nlast = int(np.sum(evals >= th) - 1)
        if nlast < 1:
            nlast = 1

        ne = K4 if eigen_num <= 0 else (eigen_num - 1)
        if ne >= K4:
            ne = K4 - 1
        if ne < 1:
            ne = 1
        if nlast > ne:
            nlast = ne

        if regulator == 's':
            last = 1.0 / evals[nlast] if evals[nlast] > 0 else 0.0
        elif regulator == 'm':
            last = 1.0 / evals[0] if evals[0] > 0 else 0.0
        else:  # hard
            last = 0.0

        lam = np.full(K4, last, dtype=np.float64)
        positive = evals[:nlast + 1] > 0
        lam[:nlast + 1] = 0.0
        lam[:nlast + 1][positive] = 1.0 / evals[:nlast + 1][positive]

        vv = (evecs.T @ v_cross) * lam
        aa = evecs @ vv
        filt00 = aa[:2 * K + 1]
        filt90 = aa[half:half + 2 * K + 1]

        # cWB _apply_: produce predicted 00/90 witness noise for this layer.
        qq = real / norm1
        QQ = imag / norm1
        wq = np.lib.stride_tricks.sliding_window_view(qq, 2 * K + 1)
        wQ = np.lib.stride_tricks.sliding_window_view(QQ, 2 * K + 1)
        val_core = wq @ filt00 - wQ @ filt90
        VAL_core = wq @ filt90 + wQ @ filt00

        nn = np.zeros(n_time, dtype=np.float64)
        NN = np.zeros(n_time, dtype=np.float64)
        nn[K:n_time - K] = val_core
        NN[K:n_time - K] = VAL_core

        # cWB apply threshold: per-layer RMS gate over non-edge region.
        kk = int(rate_tf * edge_seconds)
        if kk < K:
            kk = K
        kk += 1
        s0 = min(max(kk, 0), n_time)
        s1 = max(s0, n_time - kk)
        if s1 <= s0:
            continue

        layer_power = np.std(nn[s0:s1])**2 + np.std(NN[s0:s1])**2
        if layer_power < apply_threshold * apply_threshold:
            continue

        included_layers += 1
        noise_coeff[layer] = (nn * norm0) + 1j * (NN * norm0)

    if included_layers == 0:
        logger.info("  Regression complete: no layers passed threshold")
        return h_ts

    # Reconstruct target and predicted noise in time domain, then clean.
    coeff_orig = coeff.copy()
    tf_map.data = coeff_orig
    target_ts = np.array(wdm.w2t(tf_map), dtype=np.float64)
    tf_map.data = noise_coeff
    noise_ts = np.array(wdm.w2t(tf_map), dtype=np.float64)
    # cWB combines two inverse passes (Inverse() and Inverse(-2)) and averages.
    # In wdm_wavelet we only have one inverse path, so apply a 0.5 factor to match
    # the cWB effective subtraction scale more closely.
    cleaned_data = target_ts - 0.5 * noise_ts

    cleaned_pycbc = pycbc.types.TimeSeries(
        cleaned_data,
        delta_t=h_ts.delta_t,
        epoch=h_ts.start_time
    )

    logger.info(f"  Regression complete: {len(cleaned_pycbc)} output samples")
    return cleaned_pycbc


def _cwb_percentile_mean(data, fraction, edge_samples):
    """
    Approximate wavearray::mean(double f) used in cWB matrix building.

    For positive f: keep the lowest |x| fraction after removing edge samples.
    """
    arr = np.asarray(data, dtype=np.float64)
    n = arr.size
    if n == 0:
        return 0.0

    ff = abs(float(fraction))
    if ff > 1.0:
        ff = 1.0

    nn = int(max(0, edge_samples))
    if nn == 0 or 2 * nn >= n - 2:
        return float(np.mean(arr))

    core = arr[nn:n - nn]
    if core.size == 0:
        return float(np.mean(arr))

    if ff == 0.0:
        return float(np.median(core))

    if fraction > 0:
        keep = int(core.size * ff)
        if keep < 1:
            keep = 1
        if keep >= core.size:
            return float(np.mean(core))
        idx = np.argpartition(np.abs(core), keep - 1)[:keep]
        return float(np.mean(core[idx]))

    # Two-sided percentile mean for negative fraction (unused in current regression path).
    keep = int(core.size * ff)
    if keep < 1:
        keep = 1
    if keep >= core.size:
        return float(np.mean(core))
    sorted_core = np.sort(core)
    left = (core.size - keep) // 2
    right = left + keep
    return float(np.mean(sorted_core[left:right]))


def _cwb_rotated_products(real, imag, lag, boundary):
    """
    Compute cWB ww/WW products for one layer at a given lag.
    """
    n = real.size
    if n <= 2 * boundary + 1:
        return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)

    j = np.arange(boundary, n - boundary, dtype=np.int64)
    if lag < 0:
        rn = real[j]
        in_ = imag[j]
        jm = j - lag
        rm = real[jm]
        im = imag[jm]
    else:
        jn = j + lag
        rn = real[jn]
        in_ = imag[jn]
        rm = real[j]
        im = imag[j]

    ww = rn * rm + in_ * im
    WW = im * rn - rm * in_
    return ww, WW


def _build_correlation_matrix(signal, K):
    """
    Build Toeplitz autocorrelation matrix R(i,j) = E[x(n) * x(n+j-i)].

    This creates a symmetric positive semi-definite autocorrelation matrix
    suitable for Wiener filter design via eigenvalue decomposition.

    Parameters
    ----------
    signal : np.ndarray
        1D signal array.
    K : int
        Maximum lag (filter half-length).

    Returns
    -------
    np.ndarray
        (2*K+1, 2*K+1) symmetric autocorrelation matrix (Toeplitz structure).
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    # Compute autocorrelation at lags 0 to 2K
    # r[k] = E[x(n) * x(n-k)]
    autocorr = np.zeros(2 * K + 1)

    for lag in range(2 * K + 1):
        if lag == 0:
            autocorr[lag] = np.mean(signal**2)
        else:
            # Correlation at positive lag
            if lag < n:
                autocorr[lag] = np.mean(signal[lag:] * signal[:-lag])

    # Build Toeplitz matrix from autocorrelation values
    # R[i,j] depends on |i - j|, creating a Hermitian Toeplitz matrix
    R = np.zeros((2 * K + 1, 2 * K + 1))
    for i in range(2 * K + 1):
        for j in range(2 * K + 1):
            lag = np.abs(i - j)
            R[i, j] = autocorr[lag]

    # Ensure positive semi-definiteness by adding small regularization
    eigvals = np.linalg.eigvalsh(R)
    if eigvals[0] < 0:
        R += np.eye(2 * K + 1) * (np.abs(eigvals[0]) + 1e-10)

    return R


def _build_crosscorr_vector(signal, K):
    """
    Build cross-correlation vector p(i) = E[x(n) * x(n-i)] for i in [-K, K].

    Parameters
    ----------
    signal : np.ndarray
        1D signal array.
    K : int
        Maximum lag.

    Returns
    -------
    np.ndarray
        (2*K+1,) cross-correlation vector.
    """
    p = np.zeros(2 * K + 1)

    p[K] = np.mean(signal**2)  # Lag 0

    for lag in range(1, K + 1):
        p[K + lag] = np.mean(signal[lag:] * signal[:-lag])
        p[K - lag] = np.mean(signal[:-lag] * signal[lag:])

    return p
