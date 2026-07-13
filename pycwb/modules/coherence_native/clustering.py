"""Connected-component clustering for selected coherence pixels."""

from __future__ import annotations

import logging

import numpy as np

from pycwb.types.network_cluster import Cluster, ClusterMeta, FragmentCluster
from pycwb.types.pixel_arrays import PixelArrays

from .kernels import _label_components_grid, _subnet_subrho_batch_numba
from .veto_threshold import _igamma_inv_upper

logger = logging.getLogger(__name__)


def cluster_pixels(pixel_candidates: dict, kt: int = 1, kf: int = 1) -> FragmentCluster:
    """
    Cluster selected pixels using connected-component analysis.

    Parameters
    ----------
    pixel_candidates : dict
        Candidate payload from :func:`select_network_pixels`.
    kt : int
        Time-connectivity radius in bins (adjacency tolerance |Δtime| ≤ kt).
    kf : int
        Frequency-connectivity radius in bins (adjacency tolerance |Δfreq| ≤ kf).

    Returns
    -------
    FragmentCluster
        Clustered pixels with selected and rejected flags based on size.
    """
    if pixel_candidates is None:
        raise ValueError("cluster_pixels requires pixel_candidates")
    mask = np.asarray(pixel_candidates["mask"], dtype=bool)

    kt = int(max(1, kt))
    kf = int(max(1, kf))

    f_idx_arr = np.asarray(pixel_candidates.get("frequency", []), dtype=np.int64)
    t_idx_arr = np.asarray(pixel_candidates.get("time", []), dtype=np.int64)
    pix_det_energy = np.asarray(
        pixel_candidates.get("pix_det_energy", np.empty((0, 0), dtype=np.float64)),
        dtype=np.float64,
    )
    pix_det_index = np.asarray(
        pixel_candidates.get("pix_det_index", np.empty((0, 0), dtype=np.int64)),
        dtype=np.int64,
    )
    energy_arr = np.asarray(pixel_candidates.get("energy", []), dtype=np.float64)
    layers = int(pixel_candidates.get("layers", 1))
    rate = float(pixel_candidates.get("rate", 0.0))
    dt = 1.0 / rate if rate > 0.0 else 1.0
    n_ifo = (
        int(pix_det_energy.shape[1])
        if pix_det_energy.ndim == 2 and pix_det_energy.shape[1] > 0
        else 0
    )
    n_pix = len(f_idx_arr)

    if n_pix == 0:
        clusters = []
    else:
        # --- Connected-components labeling with rectangular (kf, kt) connectivity ---
        # Two pixels are directly connected if |Δfreq| ≤ kf AND |Δtime| ≤ kt.
        raw_labels = _label_components_grid(
            f_idx_arr, t_idx_arr, mask.shape[0], mask.shape[1], kf, kt
        )

        n_groups = int(raw_labels.max()) if raw_labels.size else 0
        group_list = [[] for _ in range(n_groups)]
        for pix_idx, lbl in enumerate(raw_labels):
            if lbl > 0:
                group_list[int(lbl) - 1].append(pix_idx)

        # Batch subnet/subrho: one Numba call across all clusters instead of ~n_clusters calls.
        pix_asnr = (
            np.sqrt(pix_det_energy)
            if n_ifo > 0
            else np.empty((n_pix, 0), dtype=np.float64)
        )
        if n_groups > 0 and n_ifo > 1:
            n_sub_c = 2.0 * _igamma_inv_upper(float(n_ifo - 1), 0.314)
            all_pix_idx = np.array(
                [pid for g in group_list for pid in g], dtype=np.int64
            )
            asnr_all = pix_asnr[all_pix_idx]  # (n_flat, n_ifo)
            noise_rms_all = np.ones_like(asnr_all)  # noise_rms=1.0 at this stage
            sizes = np.array([len(g) for g in group_list], dtype=np.int64)
            offsets_arr = np.zeros(n_groups + 1, dtype=np.int64)
            offsets_arr[1:] = np.cumsum(sizes)
            subnet_arr, subrho_arr = _subnet_subrho_batch_numba(
                asnr_all, noise_rms_all, offsets_arr, n_sub_c
            )
        else:
            subnet_arr = np.zeros(n_groups, dtype=np.float64)
            subrho_arr = np.zeros(n_groups, dtype=np.float64)

        # Construct Cluster objects with PixelArrays directly; legacy Pixel
        # objects can still be reconstructed lazily through Cluster.pixels.
        clusters = []
        for c_idx, group_indices in enumerate(group_list):
            idx_arr = np.array(group_indices, dtype=np.int64)
            f_group = f_idx_arr[idx_arr]
            t_group = t_idx_arr[idx_arr]
            n_group = len(idx_arr)
            pixel_arrays = PixelArrays.from_arrays(
                time=t_group * layers + f_group,
                frequency=f_group,
                layers=np.full(n_group, layers, dtype=np.int32),
                rate=np.full(n_group, rate, dtype=np.float32),
                core=np.ones(n_group, dtype=bool),
                likelihood=energy_arr[idx_arr],
                null=np.zeros(n_group, dtype=np.float32),
                noise_rms=np.ones((n_ifo, n_group), dtype=np.float32),
                pixel_index=pix_det_index[idx_arr].T
                if n_ifo > 0
                else np.zeros((0, n_group), dtype=np.int32),
                asnr=pix_asnr[idx_arr].T
                if n_ifo > 0
                else np.zeros((0, n_group), dtype=np.float32),
                n_ifo=n_ifo,
            )
            energy = float(energy_arr[idx_arr].sum())
            c_time = float(np.mean(t_group.astype(float) * dt - dt / 2))
            c_freq = float(np.mean(f_group.astype(float) * (rate / 2)))
            cluster_meta = ClusterMeta(
                energy=energy,
                like_net=energy,
                sub_net=float(subnet_arr[c_idx]),
                net_rho=float(subrho_arr[c_idx]),
                c_time=c_time,
                c_freq=c_freq,
            )
            clusters.append(
                Cluster(pixel_arrays=pixel_arrays, cluster_meta=cluster_meta)
            )

    n_pix_final = int(sum(len(c.pixel_arrays) for c in clusters))
    logger.info("cluster_pixels: n_clusters=%d n_pix=%d", len(clusters), n_pix_final)
    return FragmentCluster(
        rate=float(pixel_candidates.get("rate", 0.0)),
        start=float(pixel_candidates.get("start", 0.0)),
        stop=float(pixel_candidates.get("stop", 0.0)),
        bpp=1.0,
        shift=0.0,
        f_low=float(pixel_candidates.get("f_low", 0.0)),
        f_high=float(pixel_candidates.get("f_high", 0.0)),
        n_pix=n_pix_final,
        run=0,
        pair=False,
        subnet_threshold=0.0,
        clusters=clusters,
    )
