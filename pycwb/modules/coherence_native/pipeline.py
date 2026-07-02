"""Public coherence pipeline entry points."""

from __future__ import annotations

import logging
import time

import numpy as np

from pycwb.config import Config
from pycwb.types.job import WaveSegment
from pycwb.types.network_cluster import FragmentCluster
from pycwb.types.time_series import TimeSeries

from .clustering import cluster_pixels
from .selection import select_network_pixels
from .setup import setup_coherence
from .veto_threshold import build_veto_mask

logger = logging.getLogger(__name__)


def coherence(
    config: Config,
    strains: list[TimeSeries],
    return_rejected: bool = False,
    job_seg: WaveSegment | None = None,
) -> list[list[FragmentCluster]]:
    """
    Select the significant pixels for all resolution levels and all lags.

    This is the interactive convenience wrapper.  Internally it calls
    :func:`setup_coherence` once (expensive: WDM decomposition + TF maps)
    and then :func:`coherence_single_lag` for every lag (cheap: pixel
    selection + clustering only).

    Parameters
    ----------
    config : Config
        Configuration object.
    strains : list[TimeSeries]
        List of whitened strain time series.
    return_rejected : bool
        If True, keep rejected clusters in the output.
    job_seg : WaveSegment, optional
        Job segment supplying lag count and per-lag time shifts.
        When *None* a single zero-lag pass is performed.

    Returns
    -------
    list[list[FragmentCluster]]
        ``result[res][lag]`` — one FragmentCluster per resolution per lag.
    """
    logger.info("Starting coherence")

    if job_seg is None:
        # Minimal single-lag fallback for interactive / testing use.
        n_ifo = len(strains)
        import types as _types

        job_seg = _types.SimpleNamespace(
            n_lag=1,
            lag_shifts=[np.zeros(n_ifo)],
        )

    setups = setup_coherence(config, strains, job_seg=job_seg)
    n_lag = job_seg.n_lag
    n_res = len(setups)

    # Run per-lag coherence using the pre-built setup
    per_lag = [
        coherence_single_lag(setups, lag, return_rejected) for lag in range(n_lag)
    ]

    # Transpose from [lag][res] → [res][lag] (legacy output format)
    result = [[per_lag[lag][res] for lag in range(n_lag)] for res in range(n_res)]

    return result


def coherence_single_lag(
    coherence_setups: list[dict],
    lag_idx: int,
    return_rejected: bool = False,
    veto_windows: list[tuple[float, float]] | None = None,
) -> list[FragmentCluster]:
    """
    Compute coherence for one lag index, using pre-built per-resolution setups from :func:`setup_coherence`.

    This is the per-lag cheap counterpart: only pixel selection and clustering
    are performed here; all expensive TF/WDM work is already done.

    Parameters
    ----------
    coherence_setups : list[dict]
        Returned by :func:`setup_coherence`.
    lag_idx : int
        Zero-based lag index.
    return_rejected : bool
        If True keep rejected clusters in the output.
    veto_windows : list[tuple[float, float]] or None
        GPS intervals ``(start, end)`` to keep.  When not ``None``, a binary
        mask is built via :func:`build_veto_mask` and passed to pixel
        selection so that only pixels inside these windows survive.

    Returns
    -------
    list[FragmentCluster]
        One FragmentCluster per resolution for this lag.
    """
    fragment_clusters = []
    for setup in coherence_setups:
        # Unpack lag-independent setup data for this resolution
        tf_maps = setup["tf_maps"]
        Eo = setup["Eo"]
        job_seg = setup["job_seg"]
        pattern = setup["pattern"]

        # Validate lag index is within range
        if lag_idx >= job_seg.n_lag:
            raise IndexError(f"lag_idx={lag_idx} is out of range n_lag={job_seg.n_lag}")

        # Select significant pixels above threshold for this lag
        # (applies time shifts and optional veto masks)
        veto = None
        if veto_windows is not None:
            veto = build_veto_mask(tf_maps[0], veto_windows, edge=setup["segEdge"])
        t0_select = time.perf_counter()
        candidates = select_network_pixels(
            tf_maps=tf_maps,
            lag_index=lag_idx,
            energy_threshold=Eo,
            lag_shifts=job_seg.lag_shifts[lag_idx],
            veto=veto,
            edge=setup["segEdge"],
            selection_cache=setup.get("selection_cache"),
        )
        t_select = time.perf_counter() - t0_select
        n_candidates = (
            int(len(candidates["frequency"])) if isinstance(candidates, dict) else -1
        )

        # Cluster selected pixels and apply statistical selection criteria
        # (min/max cluster sizes depend on wave pattern)
        t0_cluster = time.perf_counter()
        if pattern != 0:
            # Multi-pixel clusters for network patterns (kt=2 time bins, kf=3 freq bins)
            c = cluster_pixels(candidates, kt=2, kf=3)
            c.select("subrho", setup["select_subrho"])
            c.select("subnet", setup["select_subnet"])
        else:
            # Single-pixel clusters for non-network patterns
            c = cluster_pixels(candidates, kt=1, kf=1)
        t_cluster = time.perf_counter() - t0_cluster

        # Remove clusters rejected by statistical selection unless explicitly requested
        if not return_rejected:
            c.remove_rejected()

        logger.info(
            "lag=%3d level=%d | events=%d pixels=%d candidates=%d | select=%.3fs cluster=%.3fs",
            lag_idx,
            setup["level"],
            c.event_count(),
            c.pixel_count(),
            n_candidates,
            t_select,
            t_cluster,
        )
        fragment_clusters.append(c)

    return fragment_clusters
