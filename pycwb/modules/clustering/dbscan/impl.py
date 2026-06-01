"""
DBSCAN replacement backend operating on raw pixel candidates.

Algorithm
---------
For each WDM resolution in the candidate list:

1. Convert the raw candidate dict to a flat
   :class:`~pycwb.types.pixel_arrays.PixelArrays` using
   :func:`~pycwb.modules.clustering.pixel_utils.build_pixel_arrays_from_candidates`.
2. Build a 2-D (or higher) feature matrix where time and frequency bin
   indices are normalised by configurable scale factors.
3. Run :class:`sklearn.cluster.DBSCAN`.
4. Collect per-resolution
   :class:`~pycwb.types.network_cluster.FragmentCluster` objects.

All resolutions are then merged, TD-amplitudes attached, and the result
finalised for likelihood computation.

Configuration
-------------
Parameters are read from ``config.clustering.dbscan`` when present.

Default parameter values
~~~~~~~~~~~~~~~~~~~~~~~~
======================== ======= ================================================
Parameter                Default  Description
======================== ======= ================================================
eps_time_bins            2.0     Normalisation divisor for time bin coordinate
eps_freq_bins            3.0     Normalisation divisor for frequency bin coordinate
eps                      1.2     DBSCAN neighbourhood radius in feature space
min_samples              1       Min neighbours for a point to be a core point
log_energy_weight        0.0     Weight of log-energy feature (0 = disabled)
energy_bal_weight        0.0     Weight of energy-balance feature (0 = disabled)
noise_as_singletons      True    Keep noise pixels as single-pixel clusters
min_pixels               1       Min pixels per output cluster
======================== ======= ================================================

Entry point
-----------
cluster_candidates(pixel_candidates_by_resolution, config, lag_idx,
                   setup, xtalk, td_inputs_cache, **kwargs)
"""

from __future__ import annotations

import logging

from pycwb.modules.clustering.pixel_utils import (
    build_pixel_arrays_from_candidates,
    build_fragment_cluster_from_candidates,
)
from pycwb.modules.clustering.common import (
    build_feature_matrix,
    labels_to_clusters,
)

# Re-use _get_params from the existing DBSCAN backend
from pycwb.modules.clustering.dbscan.method import _get_params

try:
    from sklearn.cluster import DBSCAN as _DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


def _cluster_one_candidates(candidates: dict, params: dict):
    """Apply DBSCAN clustering to one resolution's candidate dict."""
    pooled = build_pixel_arrays_from_candidates(candidates)
    n_pix = len(pooled)

    if n_pix == 0:
        return build_fragment_cluster_from_candidates(candidates, [])

    X = build_feature_matrix(
        pooled,
        time_scale        = float(params["eps_time_bins"]),
        freq_scale        = float(params["eps_freq_bins"]),
        log_energy_weight = float(params["log_energy_weight"]),
        energy_bal_weight = float(params["energy_bal_weight"]),
    )

    model = _DBSCAN(
        eps         = float(params["eps"]),
        min_samples = int(params["min_samples"]),
        algorithm   = "ball_tree",
        n_jobs      = 1,
    )
    labels = model.fit_predict(X)

    new_clusters = labels_to_clusters(
        pooled, labels,
        noise_as_singletons = bool(params["noise_as_singletons"]),
        min_pixels          = int(params["min_pixels"]),
    )
    return build_fragment_cluster_from_candidates(candidates, new_clusters)


def cluster_candidates(
    pixel_candidates_by_resolution: list[dict],
    config=None,
    lag_idx: int | None = None,
    setup: dict | None = None,
    xtalk=None,
    td_inputs_cache: dict | None = None,
    **kwargs,
):
    """Re-cluster raw pixel candidates using DBSCAN.

    Parameters
    ----------
    pixel_candidates_by_resolution : list[dict]
        Raw candidate dicts from
        :func:`~pycwb.modules.cwb_coherence.coherence.select_pixels_single_lag`.
    config
        Config object; optional ``clustering.dbscan`` sub-block is read.
    lag_idx : int or None
        Lag index for logging.
    setup, xtalk, td_inputs_cache
        Forwarded to :mod:`pycwb.modules.clustering.pipeline` helpers.
    **kwargs
        Per-call parameter overrides (e.g. ``eps=0.8``).

    Returns
    -------
    FragmentCluster or None

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    """
    if not HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for the DBSCAN clustering backend. "
            "Install it with: pip install scikit-learn"
        )

    from pycwb.modules.clustering.pipeline import (
        merge_fragment_clusters,
        attach_td_amplitudes,
        finalize_clusters_for_likelihood,
    )

    params  = _get_params(config, **kwargs)
    lag_str = f"lag {lag_idx}" if lag_idx is not None else "lag ?"

    if not pixel_candidates_by_resolution:
        logger.debug("[dbscan_impl] %s: no resolutions → None", lag_str)
        return None

    fragment_clusters = []
    for res_idx, candidates in enumerate(pixel_candidates_by_resolution):
        n_cand = int(len(candidates.get("frequency", [])))
        new_fc = _cluster_one_candidates(candidates, params)
        logger.debug(
            "[dbscan_impl] %s res=%d candidates=%d clusters=%d (eps=%.3f min_samples=%d)",
            lag_str, res_idx, n_cand, len(new_fc.clusters),
            params["eps"], params["min_samples"],
        )
        fragment_clusters.append(new_fc)

    merged = merge_fragment_clusters(fragment_clusters)
    if merged is None:
        return None

    attach_td_amplitudes(merged, config, setup, td_inputs_cache)
    return finalize_clusters_for_likelihood(merged, config, setup, xtalk, lag_idx)
