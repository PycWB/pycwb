"""
Batch time-delay vector extraction utility.

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
- Call ``TimeFrequencyMap.prepare_td_inputs(td_filters)`` to extract padded
  numpy arrays from the TF map before invoking the batch function.
"""

import logging

import numpy as np
from pycwb.utils.td_vector_kernels import batch_get_td_vecs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TD-inputs cache builder
# ---------------------------------------------------------------------------

def _build_td_inputs_single_level(level, config, strains_ts, upTDF):
    """Build WDM context and extract TD inputs for one resolution level.

    This is the per-level body extracted from :func:`build_td_inputs_cache`
    so that the online workflow can call it in parallel via
    ``ThreadPoolExecutor``.

    Parameters
    ----------
    level : int
        Resolution level from ``config.WDM_level``.
    config : Config
        Configuration object (needs ``WDM_beta_order``, ``WDM_precision``,
        ``TDSize``, ``nIFO``).
    strains_ts : list
        Pre-converted ``pycwb.types.time_series.TimeSeries`` per IFO.
    upTDF : int
        Up-sampling factor for TD filters.

    Returns
    -------
    tuple[int, list]
        ``(wdm_layers, per_ifo_td_inputs)`` where *wdm_layers* is the
        canonical layer key and *per_ifo_td_inputs* is one
        :class:`TDBatchInputs` per IFO.
    """
    from wdm_wavelet.wdm import WDM as WDMWavelet
    from pycwb.types.time_frequency_map import TimeFrequencyMap

    layers_at_level = 2 ** level if level > 0 else 0
    wdm_layers = max(1, int(layers_at_level))
    wdm = WDMWavelet(
        M=wdm_layers,
        K=wdm_layers,
        beta_order=config.WDM_beta_order,
        precision=config.WDM_precision,
    )
    wdm.set_td_filter(int(config.TDSize), upTDF)

    detector_tf_maps = []
    for n in range(config.nIFO):
        strain_ts = strains_ts[n]
        ts_data = np.asarray(strain_ts.data, dtype=np.float64)
        sample_rate = float(strain_ts.sample_rate)
        t0 = float(strain_ts.t0)
        wdm_tf = wdm.t2w(ts_data, sample_rate=sample_rate, t0=t0, MM=-1)
        detector_tf_maps.append(
            TimeFrequencyMap(
                data=wdm_tf.data,
                is_whitened=True,
                dt=wdm_tf.dt,
                df=wdm_tf.df,
                start=wdm_tf.start_time,
                stop=wdm_tf.end_time,
                f_low=wdm_tf.start_freq,
                f_high=wdm_tf.end_freq,
                edge=None,
                wavelet=wdm,
                len_timeseries=wdm_tf.len_timeseries,
            )
        )

    per_ifo = [
        detector_tf_maps[n].prepare_td_inputs(wdm.td_filters)
        for n in range(config.nIFO)
    ]
    return wdm_layers, per_ifo


def build_td_inputs_cache(config, strains):
    """
    Build a TD-inputs cache for all WDM resolution levels.

    For each resolution level in ``config.WDM_level``, a WDM wavelet is
    constructed, TD filters are computed, TF maps are produced for every
    detector, and :meth:`TimeFrequencyMap.prepare_td_inputs` extracts the
    padded planes needed by :func:`batch_extract_td_vecs`.

    The returned dict is keyed by both ``wdm_layers`` and ``wdm_layers + 1``
    (the cWB pixel layer-tag convention), so callers can look up by either.

    Parameters
    ----------
    config : Config
        Must have ``WDM_level``, ``nIFO``, ``TDSize``, ``upTDF``,
        ``WDM_beta_order``, ``WDM_precision``.
    strains : list
        Whitened strain time series (one per IFO).

    Returns
    -------
    dict[int, list[TDBatchInputs]]
        Layer key → list of per-IFO :class:`TDBatchInputs`.
    """
    from pycwb.types.time_series import TimeSeries

    strains_ts = [TimeSeries.from_input(strain) for strain in strains]
    upTDF = int(getattr(config, 'upTDF', 1))

    # Build per-level results sequentially (online workflow parallelises
    # this by calling _build_td_inputs_single_level directly).
    td_inputs_cache = {}
    for level in config.WDM_level:
        wdm_layers, per_ifo = _build_td_inputs_single_level(
            level, config, strains_ts, upTDF,
        )
        td_inputs_cache[int(wdm_layers)] = per_ifo
        td_inputs_cache[int(wdm_layers) + 1] = per_ifo

    return td_inputs_cache
