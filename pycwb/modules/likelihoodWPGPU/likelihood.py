"""
Likelihood entry points — JAX GPU implementation.

Provides ``setup_likelihood``, ``likelihood``, and ``likelihood_wrapper`` with
the same external interface as the CPU module so the GPU version is a drop-in
replacement.

The sky scan and DPF kernels run on JAX (CPU or GPU); the per-cluster post-
processing (xtalk norms, waveform reconstruction, chirp mass) reuses the CPU
implementations since they run once per cluster and are not performance-critical.
"""

from __future__ import annotations

import logging
import time
from math import sqrt

import numpy as np
import jax

from pycwb.types.network_cluster import Cluster, FragmentCluster
from pycwb.types.time_series import TimeSeries
from pycwb.types.time_frequency_map import TimeFrequencyMap
from pycwb.modules.xtalk.type import XTalk

# Re-use setup_likelihood from CPU module (it only builds numpy arrays)
from pycwb.modules.likelihoodWP.likelihood import (
    setup_likelihood,
    load_data_from_ifo,
    load_data_from_pixels,
    threshold_cut,
    fill_detection_statistic,
    get_chirp_mass,
    get_error_region,
)

from .types import SkyStatistics, SkyMapStatistics
from .dpf import compute_dpf, calculate_dpf_regulator
from .sky_scan import find_optimal_sky_localization
from .sky_stat import (
    compute_pixel_energy,
    project_gw_packet,
    orthogonalise_polarisations,
    compute_coherent_statistics,
)
from .utils import (
    compute_packet_rotation,
    compute_noise_correction,
    set_packet_amplitudes,
    compute_null_packet,
    project_polarisation,
    xtalk_energy_sum,
)
from .xtalk_ops import packet_norm_numpy, gw_norm_numpy

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pycwb.config.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# calculate_sky_statistics — detailed evaluation at l_max
# ---------------------------------------------------------------------------

def calculate_sky_statistics_gpu(
    sky_idx: int,
    n_ifo: int,
    n_pix: int,
    FP: np.ndarray,
    FX: np.ndarray,
    rms: np.ndarray,
    td00: np.ndarray,
    td90: np.ndarray,
    ml: np.ndarray,
    REG: np.ndarray,
    network_energy_threshold: float,
    cluster_xtalk: np.ndarray,
    cluster_xtalk_lookup_table: np.ndarray,
    DEBUG: bool = False,
    xgb_rho_mode: bool = False,
) -> SkyStatistics:
    """Calculate detailed sky statistics at the optimal sky direction.

    This mirrors ``calculate_sky_statistics`` from the CPU module but uses JAX
    kernels for the DPF / signal-packet / orthogonalisation / coherent-stats
    steps, then falls back to numpy for xtalk-dependent post-processing.
    """
    import jax.numpy as jnp

    offset = int(td00.shape[0] / 2)

    # --- Apply time delay for this sky direction ---
    v00 = np.empty((n_ifo, n_pix), dtype=np.float32)
    v90 = np.empty((n_ifo, n_pix), dtype=np.float32)
    td_energy = np.zeros((n_ifo, n_pix), dtype=np.float32)
    for i in range(n_ifo):
        v00[i] = td00[ml[i, sky_idx] + offset, i]
        v90[i] = td90[ml[i, sky_idx] + offset, i]

    for i in range(n_ifo):
        for j in range(n_pix):
            td_energy[i, j] = v00[i, j] ** 2 + v90[i, j] ** 2

    # --- JAX kernels: DPF + GW projection + orthogonalisation + coherent stats ---
    v00_j = jnp.asarray(v00, dtype=jnp.float32)
    v90_j = jnp.asarray(v90, dtype=jnp.float32)
    rms_j = jnp.asarray(rms, dtype=jnp.float32)  # (n_pix, n_ifo)
    FP_j = jnp.asarray(FP[sky_idx], dtype=jnp.float32)
    FX_j = jnp.asarray(FX[sky_idx], dtype=jnp.float32)
    REG_j = jnp.asarray(REG, dtype=jnp.float32)

    pixel_info = compute_pixel_energy(v00_j, v90_j, jnp.float32(network_energy_threshold))
    Eo = float(pixel_info["Eo"])
    total_energy_j = pixel_info["total_energy"]
    mask_j = pixel_info["mask"]

    dpf = compute_dpf(FP_j, FX_j, rms_j)

    gw = project_gw_packet(
        v00_j, v90_j,
        dpf["f"], dpf["F"], dpf["fp"], dpf["fx"],
        dpf["network_index"], total_energy_j, mask_j, REG_j,
    )

    ort = orthogonalise_polarisations(gw["signal_00"], gw["signal_90"], gw["mask"])

    stats = compute_coherent_statistics(
        v00_j, v90_j, gw["signal_00"], gw["signal_90"],
        ort["psi_sin"], ort["psi_cos"], gw["mask"],
    )

    # --- Convert JAX arrays back to numpy for xtalk post-processing ---
    mask = np.asarray(gw["mask"])
    total_energy = np.asarray(total_energy_j)
    f = np.asarray(dpf["f"])
    F = np.asarray(dpf["F"])
    fp = np.asarray(dpf["fp"])
    fx = np.asarray(dpf["fx"])
    ps_np = np.asarray(gw["signal_00"])
    pS_np = np.asarray(gw["signal_90"])
    coherent_energy = np.asarray(stats["ec"])
    gn = np.asarray(stats["gn"])
    rn = np.asarray(stats["rn"])
    ee = np.asarray(ort["energy_plus"])
    EE = np.asarray(ort["energy_cross"])
    Lo = float(ort["total_energy"])

    # --- Packet rotation (data + signal) ---
    Ep_data, pd, pD, pD_E, pD_si, pD_co, pD_a, pD_A = _numpy_packet_rotation(v00, v90, mask)
    Lp_sig, ps_rot, pS_rot, pS_E, pS_si, pS_co, pS_a, pS_A = _numpy_packet_rotation(ps_np, pS_np, mask)

    # --- Xtalk-corrected norms ---
    detector_snr, pD_E_out, rn_out, pD_norm = packet_norm_numpy(
        pd, pD, cluster_xtalk, cluster_xtalk_lookup_table, mask, pD_E)
    D_snr = np.sum(detector_snr)
    S_snr, signal_snr, pS_E_out, pS_norm = gw_norm_numpy(pD_norm, pD_E_out, pS_E, coherent_energy)

    if DEBUG:
        print(S_snr, signal_snr)
        print("Eo =", Eo, ", Lo =", Lo, ", Ep =", D_snr, ", Lp =", S_snr)

    # --- Gaussian noise correction ---
    Gn, Ec, Dc, Rc, Eh, Es, NC, NS = compute_noise_correction(
        pS_norm, pD_norm, total_energy, mask, coherent_energy, gn, rn)

    if DEBUG:
        print("Gn =", Gn, ", Ec =", Ec, ", Dc =", Dc, ", Rc =", Rc,
              ", Eh =", Eh, ", Es =", Es, ", NC =", NC, ", NS =", NS)

    # --- Set packet amplitudes ---
    N, pd, pD = set_packet_amplitudes(pd, pD, pD_norm, pD_si, pD_co, pD_a, pD_A, mask)
    N = N - 1
    _, ps_rot, pS_rot = set_packet_amplitudes(ps_rot, pS_rot, pS_norm, pS_si, pS_co, pS_a, pS_A, mask)
    pn, pN = compute_null_packet(pd, pD, ps_rot, pS_rot)

    # Raw xtalk sums
    _, pD_E_out2, rn_out2, _ = packet_norm_numpy(
        pd, pD, cluster_xtalk, cluster_xtalk_lookup_table, mask, pD_E_out)
    Em = xtalk_energy_sum(pd, pD, cluster_xtalk, cluster_xtalk_lookup_table, mask)
    Np = xtalk_energy_sum(pn, pN, cluster_xtalk, cluster_xtalk_lookup_table, mask)
    Lm = Em - Np - Gn
    norm = (Eo - Eh) / Em if Em > 0 else 1.e9
    if norm < 1:
        norm = 1
    Ec /= norm
    Dc /= norm
    ch = (Np + Gn) / (N * n_ifo)

    if DEBUG:
        print("Np =", Np, ", Em =", Em, ", Lm =", Lm, ", norm =", norm,
              ", Ec =", Ec, ", Dc =", Dc, ", ch =", ch)

    # --- Detection statistic rho ---
    xrho = 0.0
    if not xgb_rho_mode:
        cc = ch if ch > 1 else 1
        rho = np.sqrt(Ec * Rc / 2.0) if Ec > 0 else 0
    else:
        penalty = ch
        ecor = Ec
        rho = np.sqrt(ecor / (1 + penalty * (max(float(1), penalty) - 1)))
        cc = ch if ch > 1 else 1
        xrho = np.sqrt(Ec * Rc / 2.0) if Ec > 0 else 0

    # --- Polarisation projection ---
    v00_out, v90_out, p00_POL, p90_POL = project_polarisation(v00, v90, mask, fp, fx, f, F)
    v00_out, v90_out, r00_POL, r90_POL = project_polarisation(v00_out, v90_out, mask, fp, fx, f, F)

    return SkyStatistics(
        Gn=np.float32(Gn),
        Ec=np.float32(Ec),
        Dc=np.float32(Dc),
        Rc=np.float32(Rc),
        Eh=np.float32(Eh),
        Es=np.float32(Es),
        Np=np.float32(Np),
        Em=np.float32(Em),
        Lm=np.float32(Lm),
        norm=np.float32(norm),
        cc=np.float32(cc),
        rho=np.float32(rho),
        xrho=np.float32(xrho),
        Lo=np.float32(Lo),
        Eo=np.float32(Eo),
        energy_array_plus=ee,
        energy_array_cross=EE,
        pixel_mask=mask,
        v00=v00_out,
        v90=v90_out,
        gaussian_noise_correction=gn,
        coherent_energy=coherent_energy,
        N_pix_effective=N,
        noise_amplitude_00=pn,
        noise_amplitude_90=pN,
        pd=pd,
        pD=pD,
        ps=ps_rot,
        pS=pS_rot,
        p00_POL=p00_POL,
        p90_POL=p90_POL,
        r00_POL=r00_POL,
        r90_POL=r90_POL,
        S_snr=signal_snr,
        f=f,
        F=F,
    )


def _numpy_packet_rotation(v00, v90, mask):
    """Wrapper to call avx_packet_ps from the CPU module (numba-compiled)."""
    from pycwb.modules.likelihoodWP.utils import avx_packet_ps
    return avx_packet_ps(np.asarray(v00), np.asarray(v90), np.asarray(mask))


# ---------------------------------------------------------------------------
# likelihood — main per-cluster entry point
# ---------------------------------------------------------------------------

def likelihood(
    nIFO: int,
    cluster: Cluster,
    config: Config,
    MRAcatalog: str | None = None,
    strains: list[TimeSeries] | None = None,
    cluster_id: int | None = None,
    nRMS: list[TimeFrequencyMap] | None = None,
    setup: dict | None = None,
    xtalk: XTalk | None = None,
    supercluster_setup: dict | None = None,
) -> tuple[Cluster | None, SkyMapStatistics | None]:
    """Calculate the likelihood for a single cluster using JAX GPU kernels.

    Same interface as ``pycwb.modules.likelihoodWP.likelihood.likelihood``.
    The sky scan runs on JAX; post-processing reuses the CPU module.
    """
    if xtalk is None:
        if MRAcatalog is None:
            raise ValueError("likelihood(): xtalk or MRAcatalog must be provided.")
        xtalk = XTalk.load(MRAcatalog, dump=True)
    if setup is None:
        ml, FP, FX = None, None, None
        if supercluster_setup is not None:
            ml = supercluster_setup.get("ml_likelihood", supercluster_setup.get("ml"))
            FP = supercluster_setup.get("FP_likelihood", supercluster_setup.get("FP"))
            FX = supercluster_setup.get("FX_likelihood", supercluster_setup.get("FX"))
        if strains is None and ml is None:
            raise ValueError("likelihood(): setup, strains, or supercluster_setup must be provided.")
        setup = setup_likelihood(
            config, strains, nIFO, ml=ml, FP=FP, FX=FX,
            ml_big=supercluster_setup.get("ml_big_cluster") if supercluster_setup else None,
            FP_big=supercluster_setup.get("FP_big_cluster") if supercluster_setup else None,
            FX_big=supercluster_setup.get("FX_big_cluster") if supercluster_setup else None,
            big_cluster_healpix_order=supercluster_setup.get("big_cluster_healpix_order") if supercluster_setup else None,
        )
    if config is None:
        raise ValueError("likelihood(): config is required.")

    timer_start = time.perf_counter()
    stage_timings: dict[str, float] = {}
    logger.info("-------------------------------------------------------")
    logger.info("-> [GPU] Processing cluster-id=%d|pixels=%d",
                int(cluster_id) if cluster_id is not None else -1, len(cluster.pixel_arrays))
    logger.info("   ----------------------------------------------------")

    if nRMS is not None and len(nRMS) == nIFO:
        cluster.pixel_arrays.populate_noise_rms(nRMS)

    network_energy_threshold = setup["network_energy_threshold"]
    xgb_rho_mode             = setup["xgb_rho_mode"]
    gamma_regulator          = setup["gamma_regulator"]
    delta_regulator          = setup["delta_regulator"]
    net_rho_threshold        = setup["net_rho_threshold"]
    netEC_threshold          = setup["netEC_threshold"]
    netCC                    = setup["netCC"]
    ml                       = setup["ml"]
    FP                       = setup["FP_t"]
    FX                       = setup["FX_t"]
    n_sky                    = setup["n_sky"]

    REG = np.array([delta_regulator * np.sqrt(2), 0., 0.], dtype=np.float32)
    n_pix = len(cluster.pixel_arrays)

    # --- Big-cluster sky thinning ---
    _precision = int(abs(getattr(config, 'precision', 0) or 0))
    _csize = _precision % 65536
    _nres = int(getattr(config, 'nRES', 1) or 1)
    _bBB = (_csize > 0 and n_pix > _nres * _csize
            and setup.get("ml_big_cluster") is not None)
    if _bBB:
        ml = setup["ml_big_cluster"]
        FP = setup["FP_big_cluster_t"]
        FX = setup["FX_big_cluster_t"]
        n_sky = setup["n_sky_big_cluster"]
        logger.info("Cluster-id=%s is big (%d px): using coarse sky grid (%d dirs)",
                    cluster_id, n_pix, n_sky)

    # --- Prepare per-cluster inputs ---
    _t0 = time.perf_counter()
    cluster_xtalk_lookup, cluster_xtalk = xtalk.get_xtalk_pixels(cluster.pixel_arrays, True)
    rms, td00, td90, td_energy = load_data_from_pixels(None, nIFO, pixel_arrays=cluster.pixel_arrays)
    td00 = np.transpose(td00.astype(np.float32), (2, 0, 1))
    td90 = np.transpose(td90.astype(np.float32), (2, 0, 1))
    rms_t = rms.T.astype(np.float32)  # (n_pix, n_ifo) — GPU-optimal layout
    stage_timings["data_prep"] = time.perf_counter() - _t0

    # --- DPF regulator (JAX) ---
    _t0 = time.perf_counter()
    REG[1] = calculate_dpf_regulator(FP, FX, rms_t, gamma_regulator, network_energy_threshold)
    stage_timings["dpf_regulator"] = time.perf_counter() - _t0

    # --- Sky scan (JAX vmap) ---
    _t0 = time.perf_counter()
    skymap_statistics = find_optimal_sky_localization(
        nIFO, n_pix, n_sky, FP, FX, rms_t, td00, td90, ml, REG, netCC,
        delta_regulator, network_energy_threshold,
    )
    skymap_statistics = SkyMapStatistics.from_tuple(skymap_statistics)
    stage_timings["sky_scan"] = time.perf_counter() - _t0

    # --- Normalised sky probability (softmax) ---
    _t0 = time.perf_counter()
    _sky_stat_f64 = skymap_statistics.nSkyStat.astype(np.float64)
    _sky_stat_shifted = _sky_stat_f64 - _sky_stat_f64.max()
    _exp_stat = np.exp(_sky_stat_shifted)
    skymap_statistics.nProbability = (_exp_stat / _exp_stat.sum()).astype(np.float32)
    stage_timings["sky_probability"] = time.perf_counter() - _t0

    # --- l_max → sky coordinates ---
    _t0 = time.perf_counter()
    _ra_arr = setup["ra_arr"]
    _dec_arr = setup["dec_arr"]
    _l_max = int(skymap_statistics.l_max)
    _theta_rad = float(np.pi / 2.0 - _dec_arr[_l_max])
    _phi_rad = float(_ra_arr[_l_max])
    _theta_deg = float(np.degrees(_theta_rad)) % 180.0
    _phi_deg = float(np.degrees(_phi_rad)) % 360.0
    stage_timings["sky_coords"] = time.perf_counter() - _t0

    # --- Detailed statistics at l_max (JAX + numpy) ---
    _t0 = time.perf_counter()
    sky_statistics = calculate_sky_statistics_gpu(
        skymap_statistics.l_max, nIFO, n_pix, FP, FX, rms_t, td00, td90, ml, REG,
        network_energy_threshold, cluster_xtalk, cluster_xtalk_lookup,
        xgb_rho_mode=xgb_rho_mode,
    )
    stage_timings["sky_statistics_at_lmax"] = time.perf_counter() - _t0

    # --- Threshold cuts ---
    _t0 = time.perf_counter()
    selected_core_pixels = int(np.count_nonzero(np.asarray(sky_statistics.pixel_mask) > 0))
    logger.info("Selected core pixels: %d / %d", selected_core_pixels, n_pix)

    rejected = threshold_cut(
        sky_statistics, network_energy_threshold, netEC_threshold,
        net_rho_threshold=net_rho_threshold, xgb_rho_mode=xgb_rho_mode,
    )
    stage_timings["threshold_cut"] = time.perf_counter() - _t0
    if rejected:
        logger.debug("Cluster rejected: %s", rejected)
        logger.info("   cluster-id|pixels: %5d|%d",
                    int(cluster_id) if cluster_id is not None else -1, len(cluster.pixel_arrays))
        logger.info("\t <- rejected")
        stage_timings["total"] = time.perf_counter() - timer_start
        logger.info("-------------------------------------------------------")
        logger.info("Total time: %.2f s", stage_timings["total"])
        logger.info("-------------------------------------------------------")
        return None, None

    # --- Fill detection statistics (reuses CPU module) ---
    _t0 = time.perf_counter()
    if config is not None:
        from pycwb.modules.reconstruction.getMRAwaveform import _create_wdm_set_python
        _wdm_list = _create_wdm_set_python(config)
    else:
        _wdm_list = None
    fill_detection_statistic(
        sky_statistics, skymap_statistics, cluster=cluster, n_ifo=nIFO, xtalk=xtalk,
        network_energy_threshold=network_energy_threshold,
        xgb_rho_mode=xgb_rho_mode, config=config,
        cluster_xtalk=cluster_xtalk,
        cluster_xtalk_lookup=cluster_xtalk_lookup,
        wdm_list=_wdm_list,
    )

    pat0 = (getattr(config, 'pattern', 10) == 0) if config is not None else False
    get_chirp_mass(cluster, xgb_rho_mode=xgb_rho_mode, pat0=pat0)
    get_error_region(cluster)
    stage_timings["post_processing"] = time.perf_counter() - _t0

    # --- Store sky localisation metadata ---
    cluster.cluster_meta.l_max = _l_max
    cluster.cluster_meta.theta = _theta_deg
    cluster.cluster_meta.phi = _phi_deg
    if cluster.cluster_meta.c_time == 0.0:
        cluster.cluster_meta.c_time = cluster.cluster_time
    if cluster.cluster_meta.c_freq == 0.0:
        cluster.cluster_meta.c_freq = cluster.cluster_freq
    cluster.sky_time_delay = [float(ml[i, _l_max]) for i in range(nIFO)]
    cluster.cluster_status = -1

    detected = cluster.cluster_status == -1
    logger.info("   cluster-id|pixels: %5d|%d",
                int(cluster_id) if cluster_id is not None else -1, len(cluster.pixel_arrays))
    if detected:
        logger.info("\t -> SELECTED !!!")
    else:
        logger.info("\t <- rejected")

    stage_timings["total"] = time.perf_counter() - timer_start
    logger.info("-------------------------------------------------------")
    logger.info("Total events: %d", 1 if detected else 0)
    logger.info("Total time: %.2f s", stage_timings["total"])
    logger.info("Stage timings (GPU):")
    for _stage, _t in stage_timings.items():
        if _stage != "total":
            logger.info("  %-30s %.4f s  (%5.1f%%)", _stage, _t,
                        100.0 * _t / stage_timings["total"] if stage_timings["total"] > 0 else 0)
    logger.info("-------------------------------------------------------")

    # Attach stage timings to skymap_statistics for benchmark collection
    skymap_statistics.stage_timings = stage_timings

    return cluster, skymap_statistics


# ---------------------------------------------------------------------------
# likelihood_wrapper — convenience multi-lag wrapper
# ---------------------------------------------------------------------------

def likelihood_wrapper(
    config: Config,
    fragment_clusters: list[FragmentCluster],
    strains: list[TimeSeries],
    MRAcatalog: str,
    nRMS: list[TimeFrequencyMap] | None = None,
    xtalk: XTalk | None = None,
) -> list[list[tuple[Cluster, SkyMapStatistics]]]:
    """Convenience wrapper: same interface as the CPU ``likelihood_wrapper``.

    Calls :func:`setup_likelihood` once (shared with CPU) then routes every
    cluster through the GPU :func:`likelihood`.
    """
    timer_start = time.perf_counter()
    strains = [TimeSeries.from_input(s) for s in strains]

    if xtalk is None:
        xtalk = XTalk.load(MRAcatalog, dump=True)

    likelihood_setup = setup_likelihood(config, strains, config.nIFO)

    results = []
    for fragment_cluster in fragment_clusters:
        lag_results = []
        for k, selected_cluster in enumerate(fragment_cluster.clusters):
            if selected_cluster.cluster_status > 0:
                continue
            selected_cluster.cluster_id = k + 1
            result_cluster, sky_stats = likelihood(
                config.nIFO, selected_cluster, config,
                cluster_id=k + 1, nRMS=nRMS, setup=likelihood_setup, xtalk=xtalk,
            )
            if result_cluster is None or result_cluster.cluster_status != -1:
                logger.info("likelihood rejected cluster %d (%d pixels)",
                            k + 1, len(selected_cluster.pixel_arrays))
                continue
            logger.info("likelihood accepted cluster %d (%d pixels)",
                        k + 1, len(result_cluster.pixel_arrays))
            lag_results.append((result_cluster, sky_stats))
        results.append(lag_results)

    total_accepted = sum(len(r) for r in results)
    logger.info("[GPU] Likelihood wrapper done: %d accepted across %d lag(s)",
                total_accepted, len(fragment_clusters))
    logger.info("[GPU] Likelihood wrapper time: %.2f s", time.perf_counter() - timer_start)

    return results
