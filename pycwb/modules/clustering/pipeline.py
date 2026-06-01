"""
Shared finalization helpers for Phase 3 clustering backends.

These functions extract the TD-amplitude attachment and post-clustering
finalization logic from
:func:`~pycwb.modules.super_cluster.super_cluster.supercluster_single_lag`
so that every clustering backend can share a single native-compatible
finalization path instead of each duplicating the complex supercluster code.

Public API
----------
merge_fragment_clusters(fragment_clusters) -> FragmentCluster | None
    Merge per-resolution FragmentClusters into one, keeping only accepted clusters.

attach_td_amplitudes(fragment_cluster, config, setup, td_inputs_cache)
    Populate time-delay amplitude fields on all cluster PixelArrays in-place.
    No-op when *td_inputs_cache* or *setup* is None.

finalize_clusters_for_likelihood(fragment_cluster, config, setup, xtalk, lag_idx)
    Run supercluster linking, subnet cut, defragmentation, and core marking.
    Returns the final likelihood-ready FragmentCluster, or None if all
    clusters are rejected.  Returns *fragment_cluster* unchanged when *setup*,
    *config*, or *xtalk* is None (test/offline mode).

finalize_mra_clusters_for_likelihood(fragment_cluster, config, setup, xtalk, lag_idx, ...)
    MRA-specific finalization that attaches TD amplitudes and applies subnet
    cuts/core marking without calling native supercluster as the primary merger.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Resolution merging
# ─────────────────────────────────────────────────────────────────────────────

def merge_fragment_clusters(fragment_clusters: list) -> object | None:
    """Merge per-resolution FragmentClusters into one combined cluster.

    Only clusters with ``cluster_status < 1`` are kept.

    Parameters
    ----------
    fragment_clusters : list[FragmentCluster]
        Per-resolution fragment clusters (one per WDM resolution level).

    Returns
    -------
    FragmentCluster or None
        Merged cluster whose ``clusters`` list contains all accepted
        clusters from every resolution.  Returns *None* if the input is
        empty or no accepted clusters remain after filtering.
    """
    if not fragment_clusters:
        return None

    merged = fragment_clusters[0]
    merged.clusters = [c for c in merged.clusters if c.cluster_status < 1]
    for fc in fragment_clusters[1:]:
        merged.clusters += [c for c in fc.clusters if c.cluster_status < 1]

    if not merged.clusters:
        return None

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# TD amplitude attachment
# ─────────────────────────────────────────────────────────────────────────────

def attach_td_amplitudes(
    fragment_cluster,
    config,
    setup: dict | None,
    td_inputs_cache: dict | None,
) -> None:
    """Attach time-delay amplitude vectors to all cluster PixelArrays in-place.

    Mirrors the TD-attachment block in
    :func:`~pycwb.modules.super_cluster.super_cluster.supercluster_single_lag`.
    When *td_inputs_cache* or *setup* is ``None`` this function is a no-op
    so that backends can be called in test mode without a full pipeline setup.

    Parameters
    ----------
    fragment_cluster : FragmentCluster
        All accepted clusters from all resolutions (already merged).
    config
        Configuration object; provides ``nIFO``.
    setup : dict or None
        Supercluster setup from
        :func:`~pycwb.modules.super_cluster.super_cluster.setup_supercluster`.
        Must contain ``"K_td"``.
    td_inputs_cache : dict or None
        Pre-built TD input cache keyed by WDM layer index → list of
        per-IFO :class:`~pycwb.utils.td_vector_batch.TDBatchInputs`.
    """
    if td_inputs_cache is None or setup is None or config is None:
        logger.debug("attach_td_amplitudes: skipped (setup, config, or td_inputs_cache is None)")
        return

    from pycwb.modules.super_cluster.utils import expected_td_vec_len

    n_ifo = config.nIFO
    K = int(setup["K_td"])

    all_clusters = fragment_cluster.clusters
    if not all_clusters:
        return

    cluster_sizes = [len(c.pixel_arrays) for c in all_clusters]
    n_pixels = sum(cluster_sizes)
    if n_pixels == 0:
        return

    pixel_layers = np.concatenate(
        [c.pixel_arrays.layers for c in all_clusters]
    ).astype(np.int32)

    # pixel_index: (n_ifo, n_pix) per cluster → concatenate along pixel axis
    pixel_indices_all = np.concatenate(
        [c.pixel_arrays.pixel_index for c in all_clusters], axis=1
    )  # (n_ifo, n_pixels)
    pixel_indices = pixel_indices_all.T.astype(np.int32)  # (n_pixels, n_ifo)

    unique_layers = np.unique(pixel_layers)
    pixels_by_layer = {
        int(layer): np.where(pixel_layers == layer)[0] for layer in unique_layers
    }

    td_vec_len = expected_td_vec_len(K)
    all_td_amps = np.zeros((n_pixels, n_ifo, td_vec_len), dtype=np.float32)

    for layer_key, pixel_idxs in pixels_by_layer.items():
        per_ifo_inputs = td_inputs_cache.get(layer_key)
        if per_ifo_inputs is None:
            per_ifo_inputs = (
                td_inputs_cache.get(layer_key - 1)
                or td_inputs_cache.get(layer_key + 1)
            )
        if per_ifo_inputs is None:
            logger.warning(
                "Missing TD input cache for layer %d, skipping %d pixels",
                layer_key, len(pixel_idxs),
            )
            continue
        layer_pixel_indices = pixel_indices[pixel_idxs]
        for ifo_idx in range(n_ifo):
            indices_np = np.asarray(layer_pixel_indices[:, ifo_idx], dtype=np.int32)
            batch_result = per_ifo_inputs[ifo_idx].extract_td_vecs(indices_np, K)
            all_td_amps[pixel_idxs, ifo_idx, :] = batch_result

    # Distribute dense TD-amp matrix back to per-cluster PixelArrays
    offset = 0
    for ci, cluster in enumerate(all_clusters):
        n = cluster_sizes[ci]
        cluster_td = all_td_amps[offset: offset + n]
        cluster.pixel_arrays.set_td_amp_from_dense(cluster_td)
        offset += n


# ─────────────────────────────────────────────────────────────────────────────
# Post-clustering finalization
# ─────────────────────────────────────────────────────────────────────────────

def finalize_clusters_for_likelihood(
    fragment_cluster,
    config,
    setup: dict | None,
    xtalk,
    lag_idx: int,
) -> object | None:
    """Run supercluster linking, subnet cut, defragmentation, and core marking.

    Mirrors the finalization block in
    :func:`~pycwb.modules.super_cluster.super_cluster.supercluster_single_lag`.
    When *setup*, *config*, or *xtalk* is ``None`` the function returns
    *fragment_cluster* unchanged (test/offline mode — no cuts applied).

    Parameters
    ----------
    fragment_cluster : FragmentCluster
        Merged cluster with TD amplitudes already attached.
    config
        Configuration object (thresholds, flags, etc.).
    setup : dict or None
        Supercluster setup dict from
        :func:`~pycwb.modules.super_cluster.super_cluster.setup_supercluster`.
    xtalk : XTalk or None
        Cross-talk catalog used by the subnet cut.
    lag_idx : int
        Lag index (used for logging only).

    Returns
    -------
    FragmentCluster or None
        Likelihood-ready cluster after all cuts and core marking, or
        *None* if all clusters are rejected at the subnet-cut stage.
        When *setup* / *config* / *xtalk* is *None*, the input
        *fragment_cluster* is returned as-is without any cuts.
    """
    if setup is None or config is None or xtalk is None:
        logger.debug(
            "finalize_clusters_for_likelihood: skipped (setup, config, or xtalk is None); "
            "returning fragment_cluster as-is (lag=%s)", lag_idx
        )
        return fragment_cluster

    from pycwb.modules.super_cluster.super_cluster import supercluster, defragment
    from pycwb.modules.super_cluster.utils import apply_subnet_cut
    from pycwb.types.detector import calculate_e2or_from_acore

    n_ifo = config.nIFO
    super_acor = config.Acore
    super_e2or = calculate_e2or_from_acore(super_acor, n_ifo)
    subnet_acor = config.subacor if config.subacor > 0 else config.Acore
    subnet_e2or = calculate_e2or_from_acore(subnet_acor, n_ifo)
    pattern = int(getattr(config, "pattern", 0))

    clusters = fragment_cluster.clusters
    logger.info("-> Processing lag=%d with %d clusters", lag_idx, len(clusters))
    logger.info("   --------------------------------------------------")

    superclusters = supercluster(clusters, "L", config.TFgap, super_e2or, n_ifo)
    total_pixels = sum(len(c.pixel_arrays) for c in superclusters)
    accepted_superclusters = [sc for sc in superclusters if sc.cluster_status <= 0]
    logger.info(
        "   super clusters|pixels      : %6d|%d", len(superclusters), total_pixels
    )
    logger.info("   accepted superclusters     : %6d", len(accepted_superclusters))

    if not accepted_superclusters:
        logger.warning(
            "No accepted superclusters after supercluster stage (lag=%d)", lag_idx
        )
        return None

    if pattern != 0:
        accepted_superclusters = defragment(
            accepted_superclusters, config.Tgap, config.Fgap, n_ifo
        )
        logger.info(
            "   defrag clusters|pixels     : %6d|%d",
            len(accepted_superclusters),
            sum(len(c.pixel_arrays) for c in accepted_superclusters),
        )

    subrho = config.subrho if config.subrho > 0 else config.netRHO
    accepted_superclusters = apply_subnet_cut(
        accepted_superclusters,
        config.LOUD,
        setup["ml"],
        setup["FP"],
        setup["FX"],
        subnet_acor,
        subnet_e2or,
        n_ifo,
        setup["n_sky"],
        config.subnet,
        config.subcut,
        config.subnorm,
        subrho,
        xtalk,
    )

    if pattern == 0:
        accepted_superclusters = defragment(
            accepted_superclusters, config.Tgap, config.Fgap, n_ifo
        )

    total_pixels = sum(len(c.pixel_arrays) for c in accepted_superclusters)
    logger.info(
        "   post-cut clusters|pixels   : %6d|%d", len(accepted_superclusters), total_pixels
    )

    fragment_cluster.clusters = [c for c in accepted_superclusters if c.cluster_status <= 0]
    total_pixels = sum(len(c.pixel_arrays) for c in fragment_cluster.clusters)
    logger.info(
        "   final clusters|pixels      : %6d|%d",
        len(fragment_cluster.clusters),
        total_pixels,
    )

    # Mark all surviving pixels as core
    for c in fragment_cluster.clusters:
        c.pixel_arrays.core[:] = True

    return fragment_cluster


def finalize_mra_clusters_for_likelihood(
    fragment_cluster,
    config,
    setup: dict | None,
    xtalk,
    lag_idx: int,
    td_inputs_cache: dict | None = None,
    final_defrag: bool = False,
) -> object | None:
    """Finalize MRA-primary clusters without native supercluster grouping.

    ``mra_*`` backends already perform primary clustering across WDM
    resolutions, so re-running native ``supercluster()`` would turn the method
    back into the old post-hoc TF-gap merger.  This helper keeps the later
    likelihood-facing steps shared: TD-amplitude attachment, subnet cut,
    optional defragmentation, and core marking.

    When *setup*, *config*, or *xtalk* is ``None`` the subnet/final cuts are
    skipped and the input cluster is returned as-is, matching the unit-test
    behavior of :func:`finalize_clusters_for_likelihood`.
    """
    if fragment_cluster is None:
        return None

    attach_td_amplitudes(fragment_cluster, config, setup, td_inputs_cache)

    if setup is None or config is None or xtalk is None:
        logger.debug(
            "finalize_mra_clusters_for_likelihood: skipped cuts "
            "(setup, config, or xtalk is None); returning as-is (lag=%s)",
            lag_idx,
        )
        return fragment_cluster

    from pycwb.modules.super_cluster.super_cluster import defragment
    from pycwb.modules.super_cluster.utils import apply_subnet_cut
    from pycwb.types.detector import calculate_e2or_from_acore

    n_ifo = config.nIFO
    subnet_acor = config.subacor if config.subacor > 0 else config.Acore
    subnet_e2or = calculate_e2or_from_acore(subnet_acor, n_ifo)

    accepted_clusters = [c for c in fragment_cluster.clusters if c.cluster_status <= 0]
    if not accepted_clusters:
        logger.warning("No accepted MRA clusters before subnet cut (lag=%d)", lag_idx)
        return None

    logger.info("-> Processing MRA lag=%d with %d clusters", lag_idx, len(accepted_clusters))
    logger.info("   --------------------------------------------------")

    subrho = config.subrho if config.subrho > 0 else config.netRHO
    accepted_clusters = apply_subnet_cut(
        accepted_clusters,
        config.LOUD,
        setup["ml"],
        setup["FP"],
        setup["FX"],
        subnet_acor,
        subnet_e2or,
        n_ifo,
        setup["n_sky"],
        config.subnet,
        config.subcut,
        config.subnorm,
        subrho,
        xtalk,
    )

    if final_defrag and accepted_clusters:
        accepted_clusters = defragment(
            accepted_clusters, config.Tgap, config.Fgap, n_ifo
        )

    fragment_cluster.clusters = [c for c in accepted_clusters if c.cluster_status <= 0]
    if not fragment_cluster.clusters:
        logger.warning("No accepted MRA clusters after subnet cut (lag=%d)", lag_idx)
        return None

    for cluster in fragment_cluster.clusters:
        cluster.pixel_arrays.core[:] = True

    total_pixels = sum(len(c.pixel_arrays) for c in fragment_cluster.clusters)
    logger.info(
        "   final MRA clusters|pixels  : %6d|%d",
        len(fragment_cluster.clusters),
        total_pixels,
    )
    return fragment_cluster
