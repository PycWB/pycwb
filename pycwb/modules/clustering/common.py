"""Shared helpers for all clustering backends.

These utilities are intentionally small and dependency-free (NumPy only)
so every backend can import them without pulling in heavy optional
libraries.
"""

from __future__ import annotations

import numpy as np

from pycwb.types.network_cluster import Cluster, ClusterMeta, FragmentCluster
from pycwb.types.pixel_arrays import PixelArrays


# ────────────────────────────────────────────────────────────────────────────
# Pixel pooling
# ────────────────────────────────────────────────────────────────────────────

def pool_accepted_pixels(fragment_cluster: FragmentCluster):
    """Concatenate all accepted cluster pixel arrays into one flat pool.

    Parameters
    ----------
    fragment_cluster : FragmentCluster
        One resolution's fragment cluster object.

    Returns
    -------
    pooled : PixelArrays
        Concatenated pixels from all accepted (``cluster_status <= 0``)
        clusters.  Empty ``PixelArrays`` if no clusters survive.
    origin : np.ndarray, shape (n_pix,) int32
        Cluster index (into ``fragment_cluster.clusters``) that each pooled
        pixel came from.
    """
    accepted = [
        (ci, c) for ci, c in enumerate(fragment_cluster.clusters)
        if c.cluster_status <= 0 and len(c.pixel_arrays) > 0
    ]
    if not accepted:
        n_ifo = _guess_n_ifo(fragment_cluster)
        from pycwb.types.pixel_arrays import empty_pixel_arrays
        return empty_pixel_arrays(n_ifo), np.zeros(0, dtype=np.int32)

    parts = []
    labels = []
    for ci, c in accepted:
        pa = c.pixel_arrays
        parts.append(pa)
        labels.append(np.full(len(pa), ci, dtype=np.int32))

    pooled = PixelArrays.concat(parts)
    origin = np.concatenate(labels)
    return pooled, origin


def _guess_n_ifo(fragment_cluster: FragmentCluster) -> int:
    """Return n_ifo from the first cluster that has pixels, or 0."""
    for c in fragment_cluster.clusters:
        if c.pixel_arrays is not None and c.pixel_arrays._n_ifo > 0:
            return c.pixel_arrays._n_ifo
    return 0


# ────────────────────────────────────────────────────────────────────────────
# ClusterMeta estimation
# ────────────────────────────────────────────────────────────────────────────

def meta_from_pixel_arrays(pa: PixelArrays) -> ClusterMeta:
    """Compute approximate :class:`ClusterMeta` from a :class:`PixelArrays`.

    Only the fields that can be computed directly from pixel-level data are
    filled; the rest remain at their default zero value.

    Parameters
    ----------
    pa : PixelArrays
        Pixel data for a single cluster component.

    Returns
    -------
    ClusterMeta
    """
    if len(pa) == 0:
        return ClusterMeta()

    energy = float(np.sum(pa.likelihood))
    like_net = energy

    # Time and frequency centroids weighted by likelihood.
    w = pa.likelihood.astype(np.float64)
    w_sum = w.sum() + 1e-30
    c_time = float((pa.time * w).sum() / w_sum)
    c_freq = float((pa.frequency * w).sum() / w_sum)

    return ClusterMeta(
        energy=energy,
        like_net=like_net,
        c_time=c_time,
        c_freq=c_freq,
    )


# ────────────────────────────────────────────────────────────────────────────
# Cluster construction from a pixel subset
# ────────────────────────────────────────────────────────────────────────────

def build_cluster_from_mask(
    pooled_pa: PixelArrays,
    mask: np.ndarray,
    cluster_status: int = 0,
) -> Cluster:
    """Return a :class:`Cluster` containing the pixels selected by *mask*.

    Parameters
    ----------
    pooled_pa : PixelArrays
        Full pool of pixels from which to select.
    mask : np.ndarray, bool or int
        Boolean or integer index into *pooled_pa*.
    cluster_status : int
        Initial cluster status (0 = accepted / not-yet-processed).

    Returns
    -------
    Cluster
    """
    sub_pa = pooled_pa[mask]
    meta = meta_from_pixel_arrays(sub_pa)
    return Cluster(
        pixel_arrays=sub_pa,
        cluster_meta=meta,
        cluster_status=cluster_status,
    )


# ────────────────────────────────────────────────────────────────────────────
# FragmentCluster reconstruction
# ────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    pa: PixelArrays,
    *,
    time_scale: float = 2.0,
    freq_scale: float = 3.0,
    log_energy_weight: float = 0.0,
    energy_bal_weight: float = 0.0,
) -> np.ndarray:
    """Build a 2-D feature matrix suitable for density-based clustering.

    The time and frequency bin indices are normalised so that the default
    WDM TF neighbourhood (±2 bins in time, ±3 bins in frequency) maps to
    roughly ±1 in feature space.  Optional log-energy and energy-balance
    columns can be appended with configurable weights.

    Parameters
    ----------
    pa : PixelArrays
        Pixel data for one flat pixel pool.
    time_scale : float
        Normalisation divisor for time bins (default 2.0).
    freq_scale : float
        Normalisation divisor for frequency bins (default 3.0).
    log_energy_weight : float
        Weight applied to the ``log1p(likelihood)`` feature column.
        Set to 0.0 (default) to exclude it.
    energy_bal_weight : float
        Weight applied to the energy-balance-ratio column.
        Set to 0.0 (default) to exclude it.

    Returns
    -------
    X : np.ndarray, shape (n_pix, n_features)
        Feature matrix; each row is one pixel.
    """
    cols = [
        pa.time.astype(np.float64) / time_scale,
        pa.frequency.astype(np.float64) / freq_scale,
    ]
    if log_energy_weight != 0.0:
        log_e = np.log1p(np.maximum(pa.likelihood.astype(np.float64), 0.0))
        # normalise to [0, ~1] by dividing by median (robust to extremes)
        med = np.median(log_e) + 1e-30
        cols.append(log_energy_weight * log_e / med)
    if energy_bal_weight != 0.0:
        asnr = pa.asnr.astype(np.float64)
        e_per_ifo = np.square(asnr)
        e_total = e_per_ifo.sum(axis=0) + 1e-30
        ratio = e_per_ifo[0] / e_total
        cols.append(energy_bal_weight * ratio)
    return np.column_stack(cols) if len(cols) > 1 else cols[0].reshape(-1, 1)


def labels_to_clusters(
    pooled: PixelArrays,
    labels: np.ndarray,
    noise_as_singletons: bool = True,
    min_pixels: int = 1,
    rejected_clusters: list | None = None,
) -> list:
    """Convert cluster-label array → list of :class:`Cluster` objects.

    Parameters
    ----------
    pooled : PixelArrays
        Flat pool of pixels whose index matches *labels*.
    labels : np.ndarray, shape (n_pix,) int
        Component / cluster label per pixel. ``-1`` means noise (DBSCAN
        / OPTICS convention).
    noise_as_singletons : bool
        If True, each noise pixel (label == -1) is kept as its own
        single-pixel cluster.  If False, noise pixels are discarded.
    min_pixels : int
        Drop clusters with fewer than this many pixels.
    rejected_clusters : list[Cluster] | None
        Previously-rejected clusters to append verbatim.

    Returns
    -------
    list[Cluster]
    """
    new_clusters: list = []

    # Named clusters (label >= 0)
    unique_labels = np.unique(labels[labels >= 0])
    for lbl in unique_labels:
        mask = labels == lbl
        if mask.sum() < min_pixels:
            continue
        new_clusters.append(build_cluster_from_mask(pooled, mask))

    # Noise pixels
    if noise_as_singletons:
        noise_mask = labels == -1
        noise_indices = np.where(noise_mask)[0]
        for idx in noise_indices:
            new_clusters.append(build_cluster_from_mask(pooled, np.array([idx])))

    if rejected_clusters:
        new_clusters.extend(rejected_clusters)

    return new_clusters


def rebuild_fragment_cluster(
    original: FragmentCluster,
    new_clusters: list[Cluster],
) -> FragmentCluster:
    """Create a new :class:`FragmentCluster` with *new_clusters*, preserving
    all scalar metadata (rate, start, stop, f_low, f_high, …) from *original*.

    Parameters
    ----------
    original : FragmentCluster
        Source fragment cluster whose metadata fields are copied.
    new_clusters : list[Cluster]
        Replacement cluster list.

    Returns
    -------
    FragmentCluster
    """
    fc = FragmentCluster(
        rate=original.rate,
        start=original.start,
        stop=original.stop,
        bpp=original.bpp,
        shift=original.shift,
        f_low=original.f_low,
        f_high=original.f_high,
        n_pix=original.n_pix,
        run=original.run,
        pair=original.pair,
        subnet_threshold=original.subnet_threshold,
    )
    fc.clusters = new_clusters
    return fc
