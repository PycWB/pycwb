"""Utilities for converting raw pixel-candidate dicts to pycWB types.

A *pixel-candidate dict* is the output of
:func:`~pycwb.modules.cwb_coherence.coherence.select_pixels_single_lag` for
one WDM resolution level.  It is a plain Python dict with the following keys
(a superset of :func:`~pycwb.modules.cwb_coherence.coherence.select_network_pixels`):

``time``             ``(n_pix,) int64``   — raw time bin index
``frequency``        ``(n_pix,) int64``   — raw frequency bin index
``energy``           ``(n_pix,) float64`` — total pixel energy
``pix_det_energy``   ``(n_pix, n_ifo) float64`` — per-IFO pixel energy
``pix_det_index``    ``(n_pix, n_ifo) int64``   — per-IFO flat sample index
``rate``             ``float``  — WDM sample rate
``layers``           ``int``    — number of frequency layers
``start``            ``float``  — segment start GPS
``stop``             ``float``  — segment stop  GPS
``f_low``            ``float``  — minimum frequency (Hz)
``f_high``           ``float``  — maximum frequency (Hz)
``pattern``          ``int``    — wavelet pattern flag
``select_subrho``    ``float``  — sub-ρ selection threshold
``select_subnet``    ``float``  — subnet selection threshold
``level``            ``int``    — WDM resolution level
``segEdge``          ``float``  — segment edge margin (s)

Public API
----------
build_pixel_arrays_from_candidates(candidates) -> PixelArrays
    Convert raw candidate arrays to a PixelArrays for clustering.

build_fragment_cluster_from_candidates(candidates, clusters) -> FragmentCluster
    Wrap a list of Cluster objects in a FragmentCluster with metadata from
    the candidate dict.

pool_mra_pixel_candidates(pixel_candidates_by_resolution) -> MRAPixelPool
    Pool raw selected pixels from all resolutions into one PixelArrays plus
    sidecar coordinates for multi-resolution clustering methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycwb.types.network_cluster import Cluster, ClusterMeta, FragmentCluster
from pycwb.types.pixel_arrays import PixelArrays, empty_pixel_arrays


@dataclass
class MRAPixelPool:
    """Pooled selected pixels and sidecar coordinates for MRA clustering.

    ``pixel_arrays`` preserves the likelihood-facing pycWB storage contract:
    encoded ``time`` values, detector order, mixed ``layers`` / ``rate``
    values, and lag-adjusted ``pixel_index`` arrays.  The sidecar arrays keep
    the raw selected-pixel coordinates that are better suited for building
    multi-resolution graph features.
    """

    pixel_arrays: PixelArrays
    resolution_index: np.ndarray
    level: np.ndarray
    raw_time: np.ndarray
    raw_frequency: np.ndarray
    time_start: np.ndarray
    time_stop: np.ndarray
    frequency_low: np.ndarray
    frequency_high: np.ndarray
    detector_energy: np.ndarray
    source_index: np.ndarray


def build_pixel_arrays_from_candidates(candidates: dict) -> PixelArrays:
    """Convert a raw pixel-candidate dict to a :class:`~pycwb.types.pixel_arrays.PixelArrays`.

    The ``time`` field of the returned :class:`PixelArrays` is the *encoded*
    linear pixel index ``t_bin * layers + f_bin`` — the same convention used
    by :func:`~pycwb.modules.cwb_coherence.coherence.cluster_pixels` — so
    that backends operating on ``PixelArrays`` objects work identically
    regardless of whether they received data from the old
    ``coherence_single_lag`` path or the new ``select_pixels_single_lag`` path.

    Parameters
    ----------
    candidates : dict
        Raw candidate dict from
        :func:`~pycwb.modules.cwb_coherence.coherence.select_pixels_single_lag`.

    Returns
    -------
    PixelArrays
        Pixel-level data ready for backend clustering.  An empty
        ``PixelArrays`` is returned when the candidate dict contains no
        pixels.
    """
    time_bin = np.asarray(candidates.get("time", []), dtype=np.int64)
    freq_bin = np.asarray(candidates.get("frequency", []), dtype=np.int64)

    pix_det_energy_raw = candidates.get("pix_det_energy", None)
    pix_det_index_raw  = candidates.get("pix_det_index",  None)
    energy_raw         = candidates.get("energy", [])

    layers = int(candidates.get("layers", 1))
    rate   = float(candidates.get("rate", 0.0))
    n_pix  = int(len(time_bin))

    # Determine n_ifo from pix_det_energy shape
    if pix_det_energy_raw is not None:
        pix_det_energy = np.asarray(pix_det_energy_raw, dtype=np.float64)
        n_ifo = int(pix_det_energy.shape[1]) if pix_det_energy.ndim == 2 and pix_det_energy.shape[0] > 0 else 0
    else:
        pix_det_energy = np.empty((0, 0), dtype=np.float64)
        n_ifo = 0

    if n_pix == 0:
        return empty_pixel_arrays(n_ifo)

    # Encoded linear pixel index: same as cluster_pixels convention
    encoded_time = (time_bin * layers + freq_bin).astype(np.int32)

    # Per-IFO arrays: transpose to (n_ifo, n_pix)
    if pix_det_energy.shape[0] > 0 and n_ifo > 0:
        asnr = np.sqrt(np.maximum(pix_det_energy.T, 0.0)).astype(np.float32)  # (n_ifo, n_pix)
    else:
        asnr = np.zeros((n_ifo, n_pix), dtype=np.float32)

    if pix_det_index_raw is not None:
        pixel_index = np.asarray(pix_det_index_raw, dtype=np.int64).T.astype(np.int32)  # (n_ifo, n_pix)
    else:
        pixel_index = np.zeros((n_ifo, n_pix), dtype=np.int32)

    energy_arr  = np.asarray(energy_raw, dtype=np.float64).astype(np.float32)

    return PixelArrays.from_arrays(
        time        = encoded_time,
        frequency   = freq_bin.astype(np.int32),
        layers      = np.full(n_pix, layers, dtype=np.int32),
        rate        = np.full(n_pix, rate,   dtype=np.float32),
        noise_rms   = np.ones((n_ifo, n_pix), dtype=np.float32),
        pixel_index = pixel_index,
        n_ifo       = n_ifo,
        likelihood  = energy_arr,
        asnr        = asnr,
    )


def build_fragment_cluster_from_candidates(
    candidates: dict,
    clusters: list[Cluster],
) -> FragmentCluster:
    """Wrap *clusters* in a :class:`~pycwb.types.network_cluster.FragmentCluster`.

    Scalar metadata (``rate``, ``start``, ``stop``, ``f_low``, ``f_high``)
    are taken from *candidates*.  Fields that have no counterpart in the
    candidate dict (``bpp``, ``shift``, ``run``, ``pair``,
    ``subnet_threshold``) receive safe default values.

    Parameters
    ----------
    candidates : dict
        Raw candidate dict for one WDM resolution.
    clusters : list[Cluster]
        Cluster objects to store in the returned fragment cluster.

    Returns
    -------
    FragmentCluster
    """
    n_pix = sum(len(c.pixel_arrays) for c in clusters)
    fc = FragmentCluster(
        rate             = float(candidates.get("rate",   0.0)),
        start            = float(candidates.get("start",  0.0)),
        stop             = float(candidates.get("stop",   0.0)),
        bpp              = 1.0,
        shift            = 0.0,
        f_low            = float(candidates.get("f_low",  0.0)),
        f_high           = float(candidates.get("f_high", 0.0)),
        n_pix            = n_pix,
        run              = 0,
        pair             = False,
        subnet_threshold = 0.0,
    )
    fc.clusters = list(clusters)
    return fc


def pool_mra_pixel_candidates(pixel_candidates_by_resolution: list[dict]) -> MRAPixelPool:
    """Pool selected pixels from all resolutions for MRA clustering.

    Existing non-MRA backends intentionally cluster one resolution at a time.
    This helper is additive: it keeps that behavior untouched while giving
    ``mra_*`` methods one mixed-resolution pixel table plus raw/physical
    sidecar coordinates for cross-resolution edge construction.

    Parameters
    ----------
    pixel_candidates_by_resolution : list[dict]
        Raw candidate dicts from
        :func:`~pycwb.modules.cwb_coherence.coherence.select_pixels_single_lag`.

    Returns
    -------
    MRAPixelPool
        One mixed-resolution ``PixelArrays`` object and sidecar arrays.  Empty
        inputs return an empty pool with the best available ``n_ifo`` inferred
        from candidate metadata.
    """
    parts: list[PixelArrays] = []
    resolution_tags: list[np.ndarray] = []
    level_tags: list[np.ndarray] = []
    raw_times: list[np.ndarray] = []
    raw_frequencies: list[np.ndarray] = []
    time_starts: list[np.ndarray] = []
    time_stops: list[np.ndarray] = []
    frequency_lows: list[np.ndarray] = []
    frequency_highs: list[np.ndarray] = []
    detector_energies: list[np.ndarray] = []
    source_indices: list[np.ndarray] = []

    inferred_n_ifo = 0

    for res_idx, candidates in enumerate(pixel_candidates_by_resolution):
        pa = build_pixel_arrays_from_candidates(candidates)
        n_pix = len(pa)
        if pa._n_ifo > 0:
            inferred_n_ifo = pa._n_ifo

        pix_det_energy = np.asarray(
            candidates.get("pix_det_energy", np.empty((0, pa._n_ifo), dtype=np.float64)),
            dtype=np.float64,
        )
        if pix_det_energy.ndim == 2 and pix_det_energy.shape[1] > inferred_n_ifo:
            inferred_n_ifo = int(pix_det_energy.shape[1])

        if n_pix == 0:
            continue

        raw_time = np.asarray(candidates.get("time", []), dtype=np.int32)
        raw_frequency = np.asarray(candidates.get("frequency", []), dtype=np.int32)
        rate = float(candidates.get("rate", 0.0))
        level = int(candidates.get("level", res_idx))

        if rate > 0.0:
            dt = 1.0 / rate
            df = rate / 2.0
        else:
            dt = 1.0
            df = 1.0

        parts.append(pa)
        resolution_tags.append(np.full(n_pix, res_idx, dtype=np.int32))
        level_tags.append(np.full(n_pix, level, dtype=np.int32))
        raw_times.append(raw_time)
        raw_frequencies.append(raw_frequency)
        time_starts.append(raw_time.astype(np.float64) * dt)
        time_stops.append((raw_time.astype(np.float64) + 1.0) * dt)
        frequency_lows.append(raw_frequency.astype(np.float64) * df)
        frequency_highs.append((raw_frequency.astype(np.float64) + 1.0) * df)
        detector_energies.append(pix_det_energy.astype(np.float64, copy=False))
        source_indices.append(np.arange(n_pix, dtype=np.int32))

    if not parts:
        empty = empty_pixel_arrays(inferred_n_ifo)
        return MRAPixelPool(
            pixel_arrays=empty,
            resolution_index=np.zeros(0, dtype=np.int32),
            level=np.zeros(0, dtype=np.int32),
            raw_time=np.zeros(0, dtype=np.int32),
            raw_frequency=np.zeros(0, dtype=np.int32),
            time_start=np.zeros(0, dtype=np.float64),
            time_stop=np.zeros(0, dtype=np.float64),
            frequency_low=np.zeros(0, dtype=np.float64),
            frequency_high=np.zeros(0, dtype=np.float64),
            detector_energy=np.zeros((0, inferred_n_ifo), dtype=np.float64),
            source_index=np.zeros(0, dtype=np.int32),
        )

    return MRAPixelPool(
        pixel_arrays=PixelArrays.concat(parts),
        resolution_index=np.concatenate(resolution_tags),
        level=np.concatenate(level_tags),
        raw_time=np.concatenate(raw_times),
        raw_frequency=np.concatenate(raw_frequencies),
        time_start=np.concatenate(time_starts),
        time_stop=np.concatenate(time_stops),
        frequency_low=np.concatenate(frequency_lows),
        frequency_high=np.concatenate(frequency_highs),
        detector_energy=np.concatenate(detector_energies, axis=0),
        source_index=np.concatenate(source_indices),
    )
