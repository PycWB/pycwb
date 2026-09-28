"""Configure WDM regression and dispatch to the selected numerical backend.

Both engines preserve the self-witness LPE path, release amplitude cap,
percentile-stride setting, and quadrature reconstruction conventions.
"""

import shlex
from copy import copy
from dataclasses import replace

import numpy as np
from wdm_wavelet.wdm import WDM
from pycwb.constants.execution_profile import execution_profile, wdm_options


def _regression_apply_fraction(config, fraction):
    """Select release amplitude capping, honoring the explicit OLD search option."""
    if not execution_profile(config).regression_cap:
        return 1.0
    options = shlex.split(str(getattr(config, "Search", "")))
    for i, option in enumerate(options):
        if option == "--regression" and i + 1 < len(options) and options[i + 1] == "OLD":
            return 1.0
    if not 0 < fraction <= 1.0:
        raise ValueError("Regression cap fraction must be in (0, 1]")
    return fraction


def apply_regression(config, h):
    """
    Clean data with native regression using the selected Numba or JAX backend.

    This follows the cWB LPE regression path used in `regression.py`:
        target TF map + self-witness ("target"), then setFilter/setMatrix/solve/apply.

    Notes
    -----
    - Numba is the default per-layer backend; JAX is separately selectable.
    - Numerical kernels live in regression_numba.py and regression_jax.py.
    """
    from pycwb.types.time_series import TimeSeries

    backend = execution_profile(config).regression_engine

    # Match cWB defaults from schema/regression.cc
    filter_length = int(getattr(config, "REGRESSION_FILTER_LENGTH", 8))
    apply_threshold = float(getattr(config, "REGRESSION_APPLY_THR", 0.8))
    matrix_fraction = float(getattr(config, "REGRESSION_MATRIX_FRACTION", 0.95))
    eigen_threshold = float(getattr(config, "REGRESSION_SOLVE_EIGEN_THR", 0.0))
    eigen_num = int(getattr(config, "REGRESSION_SOLVE_EIGEN_NUM", 10))
    regulator = str(getattr(config, "REGRESSION_SOLVE_REGULATOR", "h")).lower()
    if regulator not in ("h", "s", "m"):
        regulator = "h"

    if not isinstance(h, TimeSeries):
        h_ts = TimeSeries.from_input(h)
    else:
        h_ts = h

    if filter_length <= 0:
        return h_ts

    layers = int(config.rateANA / 8)
    beta_order = getattr(config, "WDM_beta_order", 6)
    precision = getattr(config, "WDM_precision", 10)
    f_high = float(config.fHigh)
    sample_rate = float(h_ts.sample_rate)
    edge_seconds = float(getattr(config, "segEdge", 0.0))

    wdm = WDM(M=layers, K=layers, beta_order=beta_order, precision=precision, **wdm_options(config))

    signal_data = np.array(h_ts.data, dtype=np.float64)
    t0 = float(h_ts.start_time)

    tf_map = wdm.t2w(signal_data, sample_rate=sample_rate, t0=t0, MM=-1)

    coeff = np.asarray(tf_map.data, dtype=np.complex128)

    if coeff.ndim != 2:
        return h_ts

    n_freq, n_time = coeff.shape

    if n_freq < 3 or n_time <= 2 * filter_length + 2:
        return h_ts

    df = float(getattr(tf_map, "df", sample_rate / max(1.0, 2.0 * (n_freq - 1))))
    dt_tf = float(getattr(tf_map, "dt", 1.0))
    rate_tf = 1.0 / dt_tf if dt_tf > 0 else 1.0

    # In cWB wrapper, constructor uses flow=1 and fhigh=config.fHigh for target.
    # setFilter then loops layer indices 1..maxLayer-1.
    flow_target = 1.0
    layer_freq = np.arange(n_freq, dtype=np.float64) * df
    selected_layers = [i for i in range(1, n_freq - 1) if flow_target <= layer_freq[i] <= f_high]
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
    regulator_code = 1 if regulator == "s" else (2 if regulator == "m" else 0)

    # Pack selected layers and run batched JAX processing in one call.
    selected_layers_arr = np.asarray(selected_layers, dtype=np.int32)
    real_layers_np = np.asarray(coeff[selected_layers_arr].real, dtype=np.float64)
    imag_layers_np = np.asarray(coeff[selected_layers_arr].imag, dtype=np.float64)

    apply_fraction = _regression_apply_fraction(config, fm)
    use_numba = False
    if backend == "numba":
        from . import regression_numba

        use_numba = regression_numba._NUMBA_AVAILABLE
    if use_numba:
        noise_layers, include_mask = regression_numba._numba_process_layers(
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
            execution_profile(config).regression_percentile_stride,
            apply_fraction,
        )
    else:
        import jax
        import jax.numpy as jnp
        from .regression_jax import _jax_process_layers

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
            apply_fraction,
            percentile_stride=execution_profile(config).regression_percentile_stride,
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

    cleaned_pycwb = TimeSeries(
        data=cleaned_data,
        dt=h_ts.delta_t,
        t0=h_ts.start_time,
    )
    return cleaned_pycwb


def apply_regression_jax(config, strain):
    """Apply regression with JAX without mutating the caller's execution profile."""
    selected = copy(config)
    selected.execution_profile = replace(execution_profile(selected), regression_engine="jax")
    return apply_regression(selected, strain)


__all__ = ["apply_regression", "apply_regression_jax"]
