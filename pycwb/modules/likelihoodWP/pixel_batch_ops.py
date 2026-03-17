"""
Vectorized pixel data extraction and batch TD amplitude computation for likelihood.

Provides drop-in replacements for the Python loops in:
  - ``load_data_from_pixels``  (always called — highest impact)
  - ``_ensure_td_amp``         (fallback path when td_amp is missing)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vectorized pixel data extraction
# ---------------------------------------------------------------------------

def load_data_from_pixels_vectorized(pixels, nifo):
    """
    Vectorized replacement for the per-pixel Python loop in ``load_data_from_pixels``.

    Extracts noise RMS and time-delay amplitudes from a list of Pixel objects
    into pre-allocated numpy arrays using bulk array operations instead of
    element-wise Python loops.

    Parameters
    ----------
    pixels : list[Pixel]
    nifo   : int  — number of interferometers

    Returns
    -------
    rms       : np.ndarray, shape (nifo, n_pix)         float32
    td00      : np.ndarray, shape (nifo, n_pix, tsize2)  float32
    td90      : np.ndarray, shape (nifo, n_pix, tsize2)  float32
    td_energy : np.ndarray, shape (nifo, n_pix, tsize2)  float32
    """
    n_pix = len(pixels)
    tsize = int(np.asarray(pixels[0].td_amp[0]).shape[0])
    tsize2 = tsize // 2

    # ---- Extract noise_rms: shape (nifo, n_pix) ----
    inv_rms_arr = np.empty((nifo, n_pix), dtype=np.float64)
    for i in range(nifo):
        inv_rms_arr[i] = [1.0 / pix.data[i].noise_rms for pix in pixels]

    # Per-pixel network RMS: 1 / sqrt(sum(inv_rms^2))
    rms_pix = 1.0 / np.sqrt(np.sum(inv_rms_arr ** 2, axis=0))  # (n_pix,)

    # Normalised inverse rms: shape (nifo, n_pix)
    rms = (inv_rms_arr * rms_pix[np.newaxis, :]).astype(np.float32)

    # ---- Extract td_amp: shape (nifo, n_pix, tsize) ----
    # pixel.td_amp[i] can be a numpy array or a list of floats
    td_amp_arr = np.empty((nifo, n_pix, tsize), dtype=np.float32)
    for i in range(nifo):
        for pid, pix in enumerate(pixels):
            td_amp_arr[i, pid] = pix.td_amp[i]

    # Split into 00 and 90 quadrature halves
    td00 = td_amp_arr[:, :, :tsize2]
    td90 = td_amp_arr[:, :, tsize2:tsize]
    td_energy = td00 ** 2 + td90 ** 2

    return rms, td00, td90, td_energy


# ---------------------------------------------------------------------------
# Batch TD amplitude computation (fallback path in _ensure_td_amp)
# ---------------------------------------------------------------------------

def batch_ensure_td_amp(cluster, nIFO, strains, config, td_inputs_cache=None):
    """
    Batch replacement for the per-pixel ``wdm.get_td_vec()`` loop inside
    ``_ensure_td_amp``.

    Groups pixels by WDM layer, builds TF maps once per layer, then calls
    Numba-parallelised batch TD extraction (from super_cluster.td_vector_batch)
    for all pixels in a layer simultaneously.

    Numba compiles the kernel once per dtype signature regardless of array
    shape, so there is no per-lag JIT trace cache accumulation.

    Parameters
    ----------
    cluster : Cluster
        The cluster whose pixel.td_amp fields need to be populated.
    nIFO    : int
    strains : list of TimeSeries
    config  : Config
    td_inputs_cache : dict | None
        Optional pre-built TD-input cache
        (``{layer_key: [per_ifo_TDBatchInputs, ...]}``) .  When provided,
        the expensive ``set_td_filter`` + ``t2w`` + ``prepare_td_inputs``
        steps are skipped for any layer found in the cache.

    Returns
    -------
    bool  — True if td_amp was (re-)computed, False if already present.
    """
    from pycwb.types.time_series import TimeSeries
    from pycwb.types.time_frequency_map import TimeFrequencyMap
    from wdm_wavelet.wdm import WDM as WDMWavelet

    # Import batch helpers from super_cluster (exact same math, one impl)

    if len(cluster.pixels) == 0:
        return False

    # Check if td_amp is already fully populated
    has_td = all(
        getattr(pix, "td_amp", None) is not None and len(pix.td_amp) >= nIFO
        for pix in cluster.pixels
    )
    if has_td:
        return False

    # Normalise strains
    normalized = []
    for strain in strains:
        if isinstance(strain, TimeSeries):
            normalized.append(strain)
        elif hasattr(strain, "data") and isinstance(getattr(strain, "data"), TimeSeries):
            normalized.append(getattr(strain, "data"))
        else:
            normalized.append(TimeSeries.from_input(strain))

    def _normalize_layers(layer_tag):
        layer_tag = int(layer_tag)
        if layer_tag <= 1:
            return 1
        cand = layer_tag - 1
        return cand if cand % 2 == 0 else layer_tag

    # Build WDM contexts once per unique layer, reusing the pre-built cache
    # from supercluster_wrapper when available (Priority 2 optimisation).
    unique_layers = sorted({int(_normalize_layers(pix.layers)) for pix in cluster.pixels})
    wdm_contexts = {}
    for lc in unique_layers:
        # Check if supercluster already built td_inputs for this layer
        cached_td_inputs = None
        if td_inputs_cache:
            cached_td_inputs = td_inputs_cache.get(lc) or td_inputs_cache.get(lc + 1)
        if cached_td_inputs is not None:
            logger.debug("batch_ensure_td_amp: using cached td_inputs for layer %d", lc)
            wdm_contexts[lc] = {"td_inputs": cached_td_inputs}
            wdm_contexts[lc + 1] = wdm_contexts[lc]
            continue

        wdm = WDMWavelet(
            M=lc, K=lc,
            beta_order=config.WDM_beta_order,
            precision=config.WDM_precision,
        )
        wdm.set_td_filter(int(config.TDSize), 1)
        tf_maps = []
        for n in range(nIFO):
            strain = normalized[n]
            wdm_tf = wdm.t2w(
                np.asarray(strain.data, dtype=np.float64),
                sample_rate=float(strain.sample_rate),
                t0=float(strain.t0),
                MM=-1,
            )
            tf_maps.append(TimeFrequencyMap(
                data=wdm_tf.data,
                is_whitened=True,
                dt=wdm_tf.dt,
                df=wdm_tf.df,
                start=float(strain.t0),
                stop=float(strain.t0) + len(strain.data) / float(strain.sample_rate),
                f_low=getattr(config, "fLow", None),
                f_high=getattr(config, "fHigh", None),
                edge=getattr(config, "segEdge", None),
                wavelet=wdm,
                len_timeseries=len(strain.data),
            ))
        td_inputs_per_ifo = [tf_maps[n].prepare_td_inputs(wdm.td_filters) for n in range(nIFO)]
        wdm_contexts[lc] = {"td_inputs": td_inputs_per_ifo}
        wdm_contexts[lc + 1] = wdm_contexts[lc]

    K = int(config.TDSize)
    td_vec_default = np.zeros(4 * K + 2, dtype=np.float32)

    # Group pixels by layer
    from collections import defaultdict
    pixels_by_layer = defaultdict(list)
    for pix in cluster.pixels:
        pixels_by_layer[int(pix.layers)].append(pix)

    # Numba batch_get_td_vecs compiles once per dtype signature (not per shape),
    # so simply call batch_extract_td_vecs per (layer, ifo) — no async dispatch needed.
    for layer_key, layer_pixels in pixels_by_layer.items():
        ctx = wdm_contexts.get(layer_key) or wdm_contexts.get(layer_key - 1)
        if ctx is None:
            for pix in layer_pixels:
                pix.td_amp = [td_vec_default.copy() for _ in range(nIFO)]
            continue

        td_inputs_per_ifo = ctx["td_inputs"]
        n_layer = len(layer_pixels)

        pixel_indices = np.array(
            [[int(pix.data[n].index) for n in range(nIFO)] for pix in layer_pixels],
            dtype=np.int32,
        )  # (n_layer, nIFO)

        td_results = np.zeros((n_layer, nIFO, 4 * K + 2), dtype=np.float32)
        for ifo_idx in range(nIFO):
            td_results[:, ifo_idx, :] = td_inputs_per_ifo[ifo_idx].extract_td_vecs(
                pixel_indices[:, ifo_idx], K
            )

        for pid, pix in enumerate(layer_pixels):
            pix.td_amp = [td_results[pid, ifo_idx, :] for ifo_idx in range(nIFO)]

    return True
