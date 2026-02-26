"""
Batch time-frequency map generation for coherence using JAX vmap.

Replaces the serial per-detector loop:
    tf_maps = [
        TimeFrequencyMap.from_timeseries(ts=strain, wavelet=wdm_wavelet, ...)
        for strain in strains
    ]

with a single vmap'd call over all detectors:
    batch_t2w_detectors(strains, wdm_wavelet, config)

All detectors share the same WDM parameters and the same segment length, so
vmap over the leading detector dimension compiles once and runs in parallel.
"""

import logging
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


def _build_extended_signal(ts_data, n_filter_taps, mm_eff):
    """Mirror-pad signal to match the convention in TimeFrequencyMap._t2w_data_jax."""
    n_input = ts_data.shape[0]
    aligned_length = ((n_input + mm_eff - 1) // mm_eff) * mm_eff
    ext_len = aligned_length + 2 * n_filter_taps

    extended = jnp.zeros((ext_len,), dtype=jnp.float64)

    left_mirror_max = min(n_filter_taps, n_input - 1)
    if left_mirror_max >= 0:
        idx = jnp.arange(left_mirror_max + 1, dtype=jnp.int32)
        extended = extended.at[n_filter_taps - idx].set(ts_data[idx])

    extended = extended.at[n_filter_taps: n_filter_taps + n_input].set(ts_data)

    n_right = ext_len - n_filter_taps - n_input
    if n_right > 0:
        idx = jnp.arange(n_right, dtype=jnp.int32)
        extended = extended.at[n_filter_taps + n_input + idx].set(ts_data[n_input - idx - 1])

    return extended, aligned_length


def batch_t2w_detectors(strains, wdm_wavelet):
    """
    Compute TimeFrequencyMap data for all detectors in one batched JAX call.

    Parameters
    ----------
    strains : list of pycwb TimeSeries
        Whitened strain time series for each detector.  All must have the same
        sample_rate and the same number of samples.
    wdm_wavelet : WDMWavelet  (wdm_wavelet.WDM instance)
        The WDM wavelet used for this resolution level.  Must have JAX backend.

    Returns
    -------
    list of np.ndarray
        One complex128 array per detector, each shape (M+1, n_time).
        These are the raw TF map data arrays ready to populate pycwb TimeFrequencyMap.
    tuple (dt, df)
        Time and frequency resolution metadata.
    """
    try:
        from wdm_wavelet.core.t2w import _t2w_jax_impl
    except ImportError:
        _t2w_jax_impl = None

    M = int(wdm_wavelet.M)
    m_H = int(wdm_wavelet.m_H)
    filter_taps = jnp.asarray(np.asarray(wdm_wavelet.filter)[:m_H], dtype=jnp.float64)
    mm_eff = M   # MM=-1 → stride = M (full quadrature)
    return_quadrature = True

    sample_rate = float(strains[0].sample_rate)
    dt = M / sample_rate
    df = sample_rate / (2.0 * M)

    # Compute output shape from first strain
    n_input = len(strains[0].data)
    aligned_length = ((n_input + mm_eff - 1) // mm_eff) * mm_eff
    n_time_bins = aligned_length // mm_eff

    if _t2w_jax_impl is None:
        # Fallback: call high-level t2w_jax per detector
        from wdm_wavelet.core.t2w import t2w_jax

        def _single_t2w(signal_np):
            sig_jax = jnp.asarray(signal_np, dtype=jnp.float64)
            _, _, tf = t2w_jax(M, m_H, sig_jax, filter_taps, -1)
            return tf  # (2, n_time, M+1)

        # Stack input signals and vmap
        signals_np = np.stack([np.asarray(s.data, dtype=np.float64) for s in strains])
        signals_jax = jnp.asarray(signals_np)  # (n_det, n_input)

        batched = jax.jit(jax.vmap(_single_t2w))(signals_jax)  # (n_det, 2, n_time, M+1)
        batched = jax.block_until_ready(batched)

        result = []
        for i in range(len(strains)):
            # Convert to pycwb TimeFrequencyMap data format: complex128, shape (M+1, n_time)
            data = (np.asarray(batched[i, 0]) + 1j * np.asarray(batched[i, 1])).T
            result.append(data.astype(np.complex128))
        return result, (dt, df)

    # Fast path using the low-level fused kernel
    # Bind all static args via closure; only extended_signal is dynamic.
    def _single_t2w_impl(extended_signal):
        return _t2w_jax_impl(
            M=M,
            n_filter_taps=m_H,
            mm_eff=mm_eff,
            return_quadrature=return_quadrature,
            n_time_bins=n_time_bins,
            extended_signal=extended_signal,
            filter_taps=filter_taps,
        )

    # Pre-compute extended signals for all detectors (mirror-padded, on CPU)
    n_filter_taps = m_H
    ext_len = aligned_length + 2 * n_filter_taps
    all_extended = np.zeros((len(strains), ext_len), dtype=np.float64)
    for i, strain in enumerate(strains):
        ext, _ = _build_extended_signal(
            jnp.asarray(np.asarray(strain.data, dtype=np.float64)),
            n_filter_taps,
            mm_eff,
        )
        all_extended[i] = np.asarray(ext)

    extended_jax = jnp.asarray(all_extended)  # (n_det, ext_len)

    batched = jax.jit(jax.vmap(_single_t2w_impl))(extended_jax)   # (n_det, 2, n_time, M+1)
    batched = jax.block_until_ready(batched)

    result = []
    for i in range(len(strains)):
        data = (np.asarray(batched[i, 0]) + 1j * np.asarray(batched[i, 1])).T
        result.append(data.astype(np.complex128))

    return result, (dt, df)
