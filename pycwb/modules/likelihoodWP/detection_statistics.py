"""Detection statistics: threshold cuts, waveform fill, chirp mass.

Provides post-processing functions applied after the sky scan:
:func:`get_likelihood_rejection_reason`,
:func:`populate_detection_statistics`,
:func:`update_chirp_mass_statistics`, :func:`compute_sky_error_region`,
and supporting Numba JIT kernels.

Legacy aliases ``threshold_cut``, ``fill_detection_statistic``,
``get_chirp_mass``, and ``get_error_region`` remain available.
"""

from __future__ import annotations

import logging
from math import sqrt
import time
import numpy as np
from numba import njit, prange
from pycwb.config.config import Config
from pycwb.types.network_cluster import Cluster
from pycwb.modules.xtalk.monster import _compute_null_likelihood_numba
from pycwb.modules.xtalk.type import XTalk
from pycwb.modules.reconstruction.getMRAwaveform import (
    _create_wdm_set_python, get_MRA_wave, _pa_to_tuple, _build_wdm_njit_data,
)
from .typing import SkyStatistics, SkyMapStatistics

logger = logging.getLogger(__name__)

def populate_detection_statistics(sky_statistics: SkyStatistics, skymap_statistics: SkyMapStatistics,
                                  cluster: Cluster, n_ifo: int,
                                  xtalk: XTalk,
                                  network_energy_threshold: float,
                                  xgb_rho_mode: bool = False,
                                  config: Config = None,
                                  cluster_xtalk: np.ndarray | None = None,
                                  cluster_xtalk_lookup: np.ndarray | None = None,
                                  wdm_list=None) -> None:
    """
    Fill the detection statistics into the cluster and pixels.
    
    Parameters
    ----------
    sky_statistics : SkyStatistics
        The sky statistics object containing the calculated statistics.
    skymap_statistics : SkyMapStatistics
        The skymap statistics object to be filled.
    cluster : Cluster
        The cluster object containing the pixels.
    n_ifo : int
        Number of interferometers.
    xtalk : XTalk
        The XTalk object for cross-talk calculations.
    network_energy_threshold : float
        Energy threshold for the network.
    xgb_rho_mode : bool, optional
        If True, use XGB.rho0 statistics (rho0 without cc division). Default False.
    config : Config
        Pipeline configuration object. Required for MRA waveform reconstruction
        (hrss, strain, accurate gps_time and central_freq). Raises ValueError if None.
    cluster_xtalk : np.ndarray or None, optional
        Pre-computed CSR xtalk coefficient array from ``xtalk.get_xtalk_pixels``.
        When provided together with ``cluster_xtalk_lookup``, the internal
        ``get_xtalk_pixels`` call is skipped (saves ~0.4 s for N=2600).
    cluster_xtalk_lookup : np.ndarray or None, optional
        Pre-computed CSR lookup array (shape (N, 2)) from ``xtalk.get_xtalk_pixels``.
    wdm_list : list or None, optional
        Pre-built WDM filter-bank list from ``_create_wdm_set_python(config)``.
        When provided, ``_create_wdm_set_python`` is not called again (saves ~1 s
        per cluster). Pass from ``likelihood()`` where it is built once.

    Returns
    -------
    None
        Modifies ``cluster`` and ``skymap_statistics`` in place.
    """
    if config is None:
        raise ValueError(
            "populate_detection_statistics(): config is required. Without it, hrss/strain "
            "are zero and gps_time/central_freq use inaccurate supercluster fallback values."
        )
    timing_start = time.perf_counter()
    stage_timings: dict[str, float] = {}

    pixel_mask = sky_statistics.pixel_mask
    energy_array_plus = sky_statistics.energy_array_plus
    energy_array_cross = sky_statistics.energy_array_cross
    packet_data_phase0 = sky_statistics.pd
    packet_data_phase90 = sky_statistics.pD
    packet_signal_phase0 = sky_statistics.ps
    packet_signal_phase90 = sky_statistics.pS
    gaussian_noise_correction = sky_statistics.gaussian_noise_correction
    null_phase0 = sky_statistics.noise_amplitude_00
    null_phase90 = sky_statistics.noise_amplitude_90
    coherent_energy = sky_statistics.coherent_energy
    signal_snr_by_detector = sky_statistics.S_snr
    network_correlation = sky_statistics.Rc
    gaussian_noise = sky_statistics.Gn
    null_energy_packet = sky_statistics.Np
    effective_pixel_count = sky_statistics.N_pix_effective

    event_size = 0 # defined as Mw in cwb

    # --- First pass: set core/likelihood/null flags and per-ifo data arrays ---
    n_pixels = len(cluster.pixel_arrays)

    _t0 = time.perf_counter()
    # Fast path: vectorised update of pixel_arrays (avoids O(n_pixels * n_ifo) Python loop).
    _pa = cluster.pixel_arrays
    _pa.set_waveform_data(
        wave         = np.asarray(packet_data_phase0,  dtype=np.float32),
        w_90         = np.asarray(packet_data_phase90,  dtype=np.float32),
        asnr         = np.asarray(packet_signal_phase0,  dtype=np.float32),
        a_90         = np.asarray(packet_signal_phase90,  dtype=np.float32),
        core_mask    = pixel_mask,
        energy_plus  = np.asarray(energy_array_plus,  dtype=np.float32),
            energy_cross = np.asarray(energy_array_cross, dtype=np.float32),
        )

    # Pre-convert amplitude arrays to 2-D NumPy for fast column access
    null_phase0_arr = np.asarray(null_phase0, dtype=np.float64)
    null_phase90_arr = np.asarray(null_phase90, dtype=np.float64)
    signal_phase0_arr = np.asarray(packet_signal_phase0, dtype=np.float64)
    signal_phase90_arr = np.asarray(packet_signal_phase90, dtype=np.float64)

    # Use pre-computed xtalk arrays when available (avoids redundant O(N²) numba call).
    # Fall back to computing them here only when not passed in (e.g. standalone calls).
    if cluster_xtalk is not None and cluster_xtalk_lookup is not None:
        xtalks_lookup = cluster_xtalk_lookup
        xtalks = cluster_xtalk
    else:
        xtalks_lookup, xtalks = xtalk.get_xtalk_pixels(cluster.pixel_arrays)

    # core flags from pixel_arrays — no Python iteration
    _core = _pa.core
    null_pixel_indices = np.where(_core & (np.asarray(gaussian_noise_correction) > 0))[0].astype(np.int64)
    likelihood_pixel_indices = np.where(_core & (np.asarray(coherent_energy) > 0))[0].astype(np.int64)
    stage_timings["set_waveform_data"] = time.perf_counter() - _t0

    # --- Second pass: compute null and likelihood using the parallel numba kernel ---
    logger.debug("populate_detection_statistics: null_pixel_indices size=%d, likelihood_pixel_indices size=%d, n_pixels=%d",
                 len(null_pixel_indices), len(likelihood_pixel_indices), n_pixels)
    logger.debug("populate_detection_statistics: null_phase0_arr shape=%s, range=[%g, %g]",
                 str(null_phase0_arr.shape), float(np.min(np.abs(null_phase0_arr))), float(np.max(np.abs(null_phase0_arr))))
    logger.debug("populate_detection_statistics: gn range=[%g, %g], ec range=[%g, %g]",
                 float(np.min(gaussian_noise_correction)),
                 float(np.max(gaussian_noise_correction)),
                 float(np.min(coherent_energy)),
                 float(np.max(coherent_energy)))

    # null_out and like_out are written in place for the relevant pixel indices.
    # Initialise to zero so pixels not in the respective sets keep their old value
    # (matches behaviour of the previous Python loops).
    null_out = np.zeros(n_pixels, dtype=np.float64)
    like_out = np.zeros(n_pixels, dtype=np.float64)

    gn_arr = np.asarray(gaussian_noise_correction, dtype=np.float64)
    ec_arr = np.asarray(coherent_energy, dtype=np.float64)

    # Boolean membership masks — mirror the original inner-loop scope:
    #   original null loop:       for k in null_k_set  (core & gn > 0)
    #   original likelihood loop: for k in like_k_set  (core & ec > 0)
    null_mask = np.zeros(n_pixels, dtype=np.bool_)
    null_mask[null_pixel_indices] = True
    like_mask = np.zeros(n_pixels, dtype=np.bool_)
    like_mask[likelihood_pixel_indices] = True

    _t0 = time.perf_counter()
    _compute_null_likelihood_numba(
        null_pixel_indices, likelihood_pixel_indices,
        null_phase0_arr, null_phase90_arr, signal_phase0_arr, signal_phase90_arr,
        gn_arr, ec_arr,
        xtalks_lookup.astype(np.int64),
        xtalks,
        null_mask, like_mask,
        null_out, like_out,
    )
    _kernel_time = time.perf_counter() - _t0

    # Write results back into pixel_arrays
    for i in null_pixel_indices:
        _pa.null[i] = null_out[i]
    for i in likelihood_pixel_indices:
        _pa.likelihood[i] = like_out[i]

    # Count statistic (the set was pre-filtered, so its size is the count).
    event_size = int(len(null_pixel_indices))

    stage_timings["null_xtalk_loop"] = _kernel_time * len(null_pixel_indices) / max(len(null_pixel_indices) + len(likelihood_pixel_indices), 1)
    stage_timings["likelihood_xtalk_loop"] = _kernel_time * len(likelihood_pixel_indices) / max(len(null_pixel_indices) + len(likelihood_pixel_indices), 1)

    # --- Subnetwork statistic ---
    Nmax = 0.0
    Emax = np.max(signal_snr_by_detector)
    Esub = np.sum(signal_snr_by_detector) - Emax
    Esub = Esub * (1 + 2 * network_correlation * Esub / Emax)
    Nmax = gaussian_noise + null_energy_packet - effective_pixel_count * (n_ifo - 1)

    # --- Time-domain waveform statistics via getMRAwave reconstruction ---
    # Mirrors C++ getMRAwave('W') + getMRAwave('S') loop.
    # See docs/math/waveform_likelihood.md for the full derivation.
    #
    # Per-IFO quantities (whitened time-domain waveforms):
    #   sSNR_i = Σ_t z_signal_i(t)²             (signal energy / sSNR)  → Lw = Σ_i sSNR_i
    #   snr_i  = Σ_t z_data_i(t)²               (data   energy / snr)   → Ew_wf
    #   null_i = Σ_t (z_data - z_signal)_i(t)²  (null   energy)         → Nw_wf
    # To/Fo   = sSNR-weighted mean time / frequency over core pixels
    _t0 = time.perf_counter()
    Lw = 0.0
    sSNR_ifo  = np.zeros(n_ifo, dtype=np.float64)
    snr_ifo   = np.zeros(n_ifo, dtype=np.float64)
    null_ifo  = np.zeros(n_ifo, dtype=np.float64)
    signal_energy_physical = np.zeros(n_ifo, dtype=np.float64)
    To = 0.0
    Fo = 0.0

    # if config is not None and len(core_indices) > 0:
    # --- WDM synthesis path: exact getMRAwave equivalent (pure Python, no ROOT) ---
    # Reconstructs whitened time-domain waveforms per IFO:
    #   z_i(t) = Σ_{j∈core} [ a00_ij·ψ00_j(t) + a90_ij·ψ90_j(t) ]

    # Reuse the wdm_list built in likelihood() when provided; otherwise build it
    # here (one-off / standalone calls).  Building it per-cluster was ~1 s overhead.
    if wdm_list is None:
        wdm_list = _create_wdm_set_python(config)
    rate_ana = float(config.rateANA)

    # Pre-build pixel array tuple and WDM kernel data once; shared across all
    # (ifo, a_type, whiten) combinations so get_MRA_wave skips redundant extraction.
    _pixel_arrays  = _pa_to_tuple(cluster.pixel_arrays)
    _wdm_njit_data = _build_wdm_njit_data(wdm_list)

    for ifo_i in range(n_ifo):
        z_sig_ts = get_MRA_wave(cluster, wdm_list, rate_ana, ifo_i,
                                a_type='signal', mode=0, nproc=1, whiten=True,
                                _pixel_arrays=_pixel_arrays, _wdm_njit_data=_wdm_njit_data)
        z_dat_ts = get_MRA_wave(cluster, wdm_list, rate_ana, ifo_i,
                                a_type='strain', mode=0, nproc=1, whiten=True,
                                _pixel_arrays=_pixel_arrays, _wdm_njit_data=_wdm_njit_data)
        # For hrss: get un-whitened signal energy (physical strain units)
        z_sig_physical = get_MRA_wave(cluster, wdm_list, rate_ana, ifo_i,
                                        a_type='signal', mode=0, nproc=1, whiten=False,
                                        _pixel_arrays=_pixel_arrays, _wdm_njit_data=_wdm_njit_data)
        if z_sig_ts is None or z_dat_ts is None:
            continue
        z_sig = np.asarray(z_sig_ts.data, dtype=np.float64)
        z_dat = np.asarray(z_dat_ts.data, dtype=np.float64)
        sSNR_ifo[ifo_i] = np.sum(z_sig ** 2)
        snr_ifo[ifo_i]  = np.sum(z_dat ** 2)
        null_ifo[ifo_i] = np.sum((z_dat - z_sig) ** 2)
        if z_sig_physical is not None:
            z_sig_phys = np.asarray(z_sig_physical.data, dtype=np.float64)
            signal_energy_physical[ifo_i] = np.sum(z_sig_phys ** 2)

        # getWFtime() / getWFfreq() equivalents (mirrors C++ detector::getWFtime/getWFfreq)
        # Used to compute To/Fo exactly as C++: Fo += sSNR_i * getWFfreq_i; To /= Lw
        n_fft = len(z_sig)
        rate_wf = float(z_sig_ts.sample_rate)
        e_sig = z_sig ** 2
        E_sig = float(np.sum(e_sig))
        if E_sig > 0.0:
            t_start = float(z_sig_ts.start_time)
            wf_time_ifo = t_start + float(np.dot(e_sig, np.arange(n_fft))) / (E_sig * rate_wf)
            Z_fft = np.fft.rfft(z_sig)
            power = Z_fft.real ** 2 + Z_fft.imag ** 2
            E_fft = float(np.sum(power))
            if E_fft > 0.0:
                wf_freq_ifo = float(np.dot(power, np.arange(len(power)))) * rate_wf / n_fft / E_fft
            else:
                wf_freq_ifo = 0.0
            To += sSNR_ifo[ifo_i] * wf_time_ifo
            Fo += sSNR_ifo[ifo_i] * wf_freq_ifo

    Lw    = float(np.sum(sSNR_ifo))
    Ew_wf = float(np.sum(snr_ifo))
    Nw_wf = float(np.sum(null_ifo))
    if Lw > 0.0:
        To /= Lw
        Fo /= Lw

    # else:
    #     # Fallback: xtalk-catalog double-sum (used when config is not available).
    #     # Approximate because the catalog may omit weak-overlap pixel pairs.
    #     cross_ifo = np.zeros(n_ifo, dtype=np.float64)
    #     sSNR_ifo  = np.zeros(n_ifo, dtype=np.float64)
    #     snr_ifo   = np.zeros(n_ifo, dtype=np.float64)
    #     _pa_fb    = cluster.pixel_arrays
    #     for i_idx in core_indices:
    #         for k_idx in core_indices:
    #             xt = xtalk.get_xtalk(
    #                 pix1=(_pa_fb.layers[i_idx], _pa_fb.time[i_idx]),
    #                 pix2=(_pa_fb.layers[k_idx], _pa_fb.time[k_idx]),
    #             )
    #             if xt[0] > 2:
    #                 continue
    #             ps_i = ps_arr_np[:, i_idx]
    #             pS_i = pS_arr_np[:, i_idx]
    #             ps_k = ps_arr_np[:, k_idx]
    #             pS_k = pS_arr_np[:, k_idx]
    #             pd_i = pd_arr_np[:, i_idx]
    #             pD_i = pD_arr_np[:, i_idx]
    #             pd_k = pd_arr_np[:, k_idx]
    #             pD_k = pD_arr_np[:, k_idx]
    #             sSNR_ifo += (xt[0]*ps_i*ps_k + xt[1]*ps_i*pS_k + xt[2]*pS_i*ps_k + xt[3]*pS_i*pS_k)
    #             snr_ifo  += (xt[0]*pd_i*pd_k + xt[1]*pd_i*pD_k + xt[2]*pD_i*pd_k + xt[3]*pD_i*pD_k)
    #             cross_ifo += (xt[0]*pd_i*ps_k + xt[1]*pd_i*pS_k + xt[2]*pD_i*ps_k + xt[3]*pD_i*pS_k)
    #         s_snr_pix = float(np.sum(ps_arr_np[:, i_idx] ** 2 + pS_arr_np[:, i_idx] ** 2))
    #         _r  = float(_pa_fb.rate[i_idx])
    #         _ly = float(_pa_fb.layers[i_idx])
    #         pix_time = float(_pa_fb.time[i_idx]) / (_r * _ly) if (_r > 0 and _ly > 0) else 0.0
    #         pix_freq = float(_pa_fb.frequency[i_idx]) * _r / 2.0 if _r > 0 else 0.0
    #         To += s_snr_pix * pix_time
    #         Fo += s_snr_pix * pix_freq
    #     Lw = float(np.sum(sSNR_ifo))
    #     null_ifo = snr_ifo - 2.0 * cross_ifo + sSNR_ifo
    #     Ew_wf = float(np.sum(snr_ifo))
    #     Nw_wf = float(np.sum(null_ifo))
    #     if Lw > 0.0:
    #         To /= Lw
    #         Fo /= Lw

    stage_timings["mra_waveform_reconstruction"] = time.perf_counter() - _t0

    # xSNR per IFO: geometric mean  C++ get_XS() = sqrt(get_XX() * get_SS())
    _t0 = time.perf_counter()
    xSNR_ifo = np.sqrt(np.maximum(snr_ifo * sSNR_ifo, 0.0))

    # --- Detection statistics: netCC, norm, rho (mirrors network.cc likelihoodWP) ---
    # Energy notation:
    #   Eo    — total TF-domain data energy
    #   Eh    — satellite (halo) energy
    #   Em    — pixel-domain xtalk-corrected energy (likesky / neted[3])
    #   Ew_wf — waveform-domain data energy from getMRAwave (neted[2])
    #   Nw_wf — waveform-domain null energy from getMRAwave (neted[1] - Gn)
    # C++ formulas:
    #   ch_wf = (Nw_wf + Gn) / (N * nIFO)
    #   Cp = Ec*Rc / (Ec*Rc + (Dc+Nw_wf+Gn)       - N*(nIFO-1))   # netCC[0]
    #   Cr = Ec*Rc / (Ec*Rc + (Dc+Nw_wf+Gn)*cc_Cr - N*(nIFO-1))   # netCC[1]
    #   norm = (Eo-Eh) / Ew_wf  clamped to ≥ 1, stored as norm*2
    Dc = float(sky_statistics.Dc)
    Ec = float(sky_statistics.Ec)
    Rc_val = float(sky_statistics.Rc)
    Eo = float(sky_statistics.Eo)
    Eh = float(sky_statistics.Eh)
    Gn_val = float(sky_statistics.Gn)
    N_eff = float(effective_pixel_count)
    Nw_for_stats = max(Nw_wf, 0.0)  # clamp to avoid negative chi2
    ch_td = (Nw_for_stats + Gn_val) / (N_eff * n_ifo) if (N_eff * n_ifo) > 0 else 1.0

    # cc_Cr: Cr-specific correction (NOT the simple ch used for rho)
    cc_Cr = 1.0 + (ch_td - 1.0) * 2.0 * (1.0 - Rc_val) if ch_td > 1.0 else 1.0
    denom_r = Ec * Rc_val + (Dc + Nw_for_stats + Gn_val) * cc_Cr - N_eff * (n_ifo - 1)
    denom_p = Ec * Rc_val + (Dc + Nw_for_stats + Gn_val) - N_eff * (n_ifo - 1)
    Cr_td = (Ec * Rc_val / denom_r) if denom_r > 0 else 0.0
    Cp_td = (Ec * Rc_val / denom_p) if denom_p > 0 else 0.0

    norm_td = (Eo - Eh) / Ew_wf if Ew_wf > 0 else 1.0
    if norm_td < 1.0:
        norm_td = 1.0

    # rho is divided by sqrt(cc) using Nw-based chi2 (time-domain null, matches C++ line 939)
    cc_rho_td = ch_td if ch_td > 1.0 else 1.0
    rho_reduced = float(sky_statistics.rho) / sqrt(cc_rho_td)
    stage_timings["detection_statistics"] = time.perf_counter() - _t0

    # --- Store all fields on cluster_meta ---
    _t0 = time.perf_counter()
    cluster.cluster_meta.sky_size = event_size
    cluster.cluster_meta.sub_net = Esub / (Esub + Nmax) if (Esub + Nmax) > 0 else 0.0
    cluster.cluster_meta.sub_net2 = skymap_statistics.nCorrelation[skymap_statistics.l_max]
    cluster.cluster_meta.like_sky = float(sky_statistics.Em)          # Em (neted[3]): pixel-domain xtalk energy
    cluster.cluster_meta.energy_sky = sky_statistics.Eo               # TF-domain data energy (neted[4])
    cluster.cluster_meta.net_ecor = sky_statistics.Ec                 # packet coherent energy
    cluster.cluster_meta.norm_cor = sky_statistics.Ec * sky_statistics.Rc  # normalised coherent energy
    cluster.cluster_meta.like_net = float(Lw)                         # waveform likelihood (likenet)
    cluster.cluster_meta.energy = float(Ew_wf)                        # getMRAwave data energy (neted[2])
    cluster.cluster_meta.net_null = float(Nw_for_stats + Gn_val)      # packet null (neted[1])
    cluster.cluster_meta.net_ed = float(Nw_for_stats + Gn_val + Dc - N_eff * n_ifo)  # residual null (neted[0])
    cluster.cluster_meta.norm = float(norm_td * 2.0)                  # packet norm
    cluster.cluster_meta.net_cc = float(Cp_td)                        # network cc (netcc[0])
    cluster.cluster_meta.sky_cc = float(Cr_td)                        # reduced network cc (netcc[1])
    # c_time / c_freq from Lw-weighted centroid over core pixels
    if Lw > 0.0:
        cluster.cluster_meta.c_time = float(To)
        cluster.cluster_meta.c_freq = float(Fo)

    if not xgb_rho_mode:  # original 2G
        cluster.cluster_meta.net_rho = rho_reduced
        cluster.cluster_meta.net_rho2 = float(sky_statistics.rho)
    else:  # XGB.rho0
        # rho[0] = -netRHO = rho  (XGB rho0, no cc division — C++ netevent.cc line 979)
        cluster.cluster_meta.net_rho = float(sky_statistics.rho)
        # rho[1] = netrho = xrho/sqrt(cc)  (original 2G rho with cc — C++ netevent.cc line 980)
        cluster.cluster_meta.net_rho2 = float(sky_statistics.xrho) / sqrt(cc_rho_td)

    cluster.cluster_meta.g_net = skymap_statistics.nAntennaPrior[skymap_statistics.l_max]
    cluster.cluster_meta.a_net = skymap_statistics.nAlignment[skymap_statistics.l_max]
    cluster.cluster_meta.i_net = 0
    cluster.cluster_meta.ndof = effective_pixel_count
    cluster.cluster_meta.sky_chi2 = skymap_statistics.nDisbalance[skymap_statistics.l_max]
    cluster.cluster_meta.g_noise = sky_statistics.Gn
    cluster.cluster_meta.iota = 0.0
    cluster.cluster_meta.psi = 0.0
    cluster.cluster_meta.ellipticity = 0

    # Per-IFO xtalk-corrected waveform energies (getMRAwave equivalents for snr/sSNR/xSNR)
    cluster.cluster_meta.signal_snr = sSNR_ifo.tolist()   # C++ d->sSNR = get_SS() per IFO
    cluster.cluster_meta.wave_snr   = snr_ifo.tolist()    # C++ d->enrg = get_XX() per IFO
    cluster.cluster_meta.cross_snr  = xSNR_ifo.tolist()   # C++ d->xSNR = get_XS() per IFO
    cluster.cluster_meta.signal_energy_physical = signal_energy_physical.tolist()  # physical strain energy for hrss
    cluster.cluster_meta.null_energy = null_ifo.tolist()  # null energy per IFO (C++ d->null)

    logger.debug(
        "populate_detection_statistics: sky_size=%d sub_net=%.4f net_cc=%.4f sky_cc=%.4f "
        "like_net=%.2f energy=%.2f net_null=%.4f norm=%.4f rho=%.4f "
        "Ew_wf=%.2f Nw_wf=%.4f like_sky=%.2f",
        cluster.cluster_meta.sky_size, cluster.cluster_meta.sub_net,
        cluster.cluster_meta.net_cc, cluster.cluster_meta.sky_cc,
        cluster.cluster_meta.like_net, cluster.cluster_meta.energy,
        cluster.cluster_meta.net_null, cluster.cluster_meta.norm,
        cluster.cluster_meta.net_rho,
        Ew_wf, Nw_wf, cluster.cluster_meta.like_sky,
    )

    stage_timings["store_cluster_meta"] = time.perf_counter() - _t0
    stage_timings["total"] = time.perf_counter() - timing_start
    logger.info("populate_detection_statistics stage timings:")
    for _stage, _t in stage_timings.items():
        if _stage != "total":
            logger.info("  %-30s %.4f s  (%5.1f%%)", _stage, _t,
                        100.0 * _t / stage_timings["total"] if stage_timings["total"] > 0 else 0)


def get_likelihood_rejection_reason(
    sky_statistics: SkyStatistics,
    network_energy_threshold: float,
    netEC_threshold: float,
    net_rho_threshold: float | None = None,
    xgb_rho_mode: bool = False,
) -> str:
    """
    Apply threshold cuts based on the sky statistics and network energy threshold.
    
    Parameters
    ----------
    sky_statistics : SkyStatistics
        The statistics calculated for the sky location.
    network_energy_threshold : float
        The threshold for network energy.
    netEC_threshold : float
        The threshold for net correlation energy (``netEC``).
    net_rho_threshold : float or None, optional
        Absolute ``netRHO`` threshold. In XGB mode C++ compares against
        ``fabs(netRHO)`` directly.
    xgb_rho_mode : bool, optional
        If True, apply XGB.rho0 cuts instead of the original 2G cuts.

    Returns
    -------
    str or None
        A rejection reason string if any cut fails; ``None`` if the cluster passes.
    """
    Lm = sky_statistics.Lm
    Eo = sky_statistics.Eo
    Eh = sky_statistics.Eh
    Ec = sky_statistics.Ec
    Rc = sky_statistics.Rc
    cc = sky_statistics.cc
    rho = sky_statistics.rho
    N = sky_statistics.N_pix_effective   # effective pixel count (_avx_setAMP_ps() - 1)
    if not xgb_rho_mode:
        condition_1 = Lm <= 0.
        condition_2 = (Eo - Eh) <= 0.
        condition_3 = Ec * Rc / cc < netEC_threshold
        condition_4 = N < 1   # C++: N < 1 (pixel count, not null energy)
        if condition_1 or condition_2 or condition_3 or condition_4:
            rejection_reason = ""
            if condition_1:
                rejection_reason += f"Lm > 0 but Lm = {Lm};"
            if condition_2:
                rejection_reason += f" (Eo - Eh) > 0 but (Eo - Eh) = {Eo - Eh};"
            if condition_3:
                rejection_reason += f" Ec * Rc / cc >= netEC_threshold but Ec * Rc / cc = {Ec * Rc / cc:.4f} < {netEC_threshold:.4f};"
            if condition_4:
                rejection_reason += f" N < 1 but N = {N};"
            return rejection_reason
    else:
        # For XGB.rho0 case C++ uses `rho < fabs(netRHO)` directly.
        if net_rho_threshold is None:
            net_rho_threshold = (netEC_threshold / 2.0) ** 0.5
        condition_1 = Lm <= 0.
        condition_2 = (Eo - Eh) <= 0.
        condition_3 = not np.isfinite(rho) or rho < net_rho_threshold
        condition_4 = N < 1   # C++: N < 1 (pixel count)
        if condition_1 or condition_2 or condition_3 or condition_4:
            rejection_reason = ""
            if condition_1:
                rejection_reason += f"Lm > 0 but Lm = {Lm};"
            if condition_2:
                rejection_reason += f" (Eo - Eh) > 0 but (Eo - Eh) = {Eo - Eh};"
            if condition_3:
                rejection_reason += f" rho >= |netRHO| but rho = {rho} < {net_rho_threshold};"
            if condition_4:
                rejection_reason += f" N < 1 but N = {N};"
            return rejection_reason
        
    return None  # No rejection, all conditions passed


def compute_sky_error_region(
    cluster: Cluster,
    sky_probability=None,
    pixel_area_deg2: float | None = None,
    searched_sky_index: int | None = None,
):
    """Populate the compact cWB-compatible ``erA`` sky-area summary.

    Parameters
    ----------
    cluster
        Cluster to update.
    sky_probability
        Normalized probability for the full sky grid.  Zero-probability pixels
        are outside the evaluated support.  If omitted, this function is a
        backward-compatible no-op and returns any existing ``cluster.sky_area``.
    pixel_area_deg2
        Equal-area sky-pixel area in square degrees.
    searched_sky_index
        Optional target or injection pixel.  When supplied, entries 0 and 10
        are respectively sqrt(searched HPD area) and its credible level.

    Returns
    -------
    list[float]
        Eleven legacy values: searched location, 10--90% HPD sqrt-areas, and
        searched credible level.  Area square roots have numerical unit degree.
    """
    if sky_probability is None or pixel_area_deg2 is None:
        return list(getattr(cluster, "sky_area", []) or [])

    probability = np.asarray(sky_probability, dtype=np.float64)
    if probability.ndim != 1:
        raise ValueError("sky_probability must be one-dimensional")
    if not np.isfinite(pixel_area_deg2) or pixel_area_deg2 <= 0.0:
        raise ValueError("pixel_area_deg2 must be positive and finite")
    evaluated = np.where(np.isfinite(probability) & (probability > 0.0))[0]
    if evaluated.size == 0:
        raise ValueError("sky_probability has no positive finite pixels")

    values = probability[evaluated]
    values /= float(np.sum(values))
    order = np.argsort(-values, kind="stable")
    ranked_indices = evaluated[order]
    ranked_probability = values[order]
    cumulative = np.cumsum(ranked_probability, dtype=np.float64)

    legacy = [0.0]
    for level in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        pixel_count = int(np.searchsorted(cumulative, level, side="left")) + 1
        legacy.append(float(np.sqrt(pixel_count * pixel_area_deg2)))
    legacy.append(0.0)

    if searched_sky_index is not None:
        positions = np.where(ranked_indices == int(searched_sky_index))[0]
        if positions.size:
            position = int(positions[0])
            legacy[0] = float(np.sqrt((position + 1) * pixel_area_deg2))
            legacy[10] = float(min(cumulative[position], 1.0))

    cluster.sky_area = legacy
    return legacy


@njit(cache=True, parallel=True)
def _count_chirp_track_overlaps_numba(x, y, xerr, yerr, kk, m_vals):
    """Phase 1 of mchirp Hough transform: compute the max interval-overlap count
    for each mass value (independent → parallelised with prange).

    For each mass m the t-f locus is a line  y = sl*x + b  in (time, F^{-8/3})
    space.  Each pixel defines an error ellipse that, projected onto the b-axis,
    gives an interval [bmin, bmax].  The maximum number of overlapping intervals
    is the Hough vote count for that mass.

    Parameters
    ----------
    x, y, xerr, yerr : 1-D float64 arrays, length n_pts
        Pixel coordinates and their uncertainties.
    kk : float
        Pre-computed chirp-mass constant.
    m_vals : 1-D float64 array, length n_mass
        Mass grid to scan.

    Returns
    -------
    nsel_arr : 1-D int64 array, length n_mass
        Maximum overlap count per mass value.
    """
    n_mass = len(m_vals)
    n_pts  = len(x)
    nsel_arr = np.zeros(n_mass, dtype=np.int64)

    for mi in prange(n_mass):
        m  = m_vals[mi]
        sl = kk * np.abs(m) ** (5.0 / 3.0)
        if m > 0.0:
            sl = -sl

        Db   = np.sqrt(2.0 * (sl * sl * xerr * xerr + yerr * yerr))
        bmin = y - sl * x - Db
        bmax = bmin + 2.0 * Db

        # Build flat endpoint list: opens (+1) followed by closes (-1)
        ep_val  = np.empty(2 * n_pts, dtype=np.float64)
        ep_type = np.empty(2 * n_pts, dtype=np.float64)
        for i in range(n_pts):
            ep_val[i]          = bmin[i]
            ep_type[i]         = 1.0
            ep_val[n_pts + i]  = bmax[i]
            ep_type[n_pts + i] = -1.0

        order = np.argsort(ep_val)

        # Walk sorted endpoints; track running overlap count
        cum    = 0
        maxcum = 0
        for i in range(2 * n_pts):
            idx = order[i]
            cum += int(ep_type[idx])
            if cum > maxcum:
                maxcum = cum

        nsel_arr[mi] = maxcum

    return nsel_arr


@njit(cache=True)
def _fit_chirp_track_candidates_numba(x, y, xerr, yerr, wgt, kk, m_vals, cand_indices, nselmax, chi2_thr):
    """Phase 2 of mchirp Hough transform: fine b-grid search among candidate masses.

    For each candidate mass (those achieving *nselmax* votes in phase 1) the
    b-axis is scanned at step 0.0025 within segments that attain the maximum
    overlap.  The (m, b) pair minimising the likelihood-weighted mean chi2 is
    returned.

    Parameters
    ----------
    x, y, xerr, yerr, wgt : 1-D float64 arrays, length n_pts
    kk : float
    m_vals : 1-D float64 array (full mass grid)
    cand_indices : 1-D int64 array — indices into m_vals with nsel == nselmax
    nselmax : int
    chi2_thr : float

    Returns
    -------
    m0, b0 : float
        Best-fit chirp-mass slope and intercept.
    """
    n_pts   = len(x)
    b_step  = 0.0025
    chi2min = 1e100
    m0 = m_vals[cand_indices[0]]
    b0 = 0.0

    for jj in range(len(cand_indices)):
        mi = cand_indices[jj]
        m  = m_vals[mi]
        sl = kk * np.abs(m) ** (5.0 / 3.0)
        if m > 0.0:
            sl = -sl

        # Per-pixel chi2 denominator
        eps  = sl * sl * xerr * xerr + yerr * yerr

        # Recompute sorted endpoints (cheap — only a few candidate masses)
        Db   = np.sqrt(2.0 * (sl * sl * xerr * xerr + yerr * yerr))
        bmin = y - sl * x - Db
        bmax = bmin + 2.0 * Db

        ep_val  = np.empty(2 * n_pts, dtype=np.float64)
        ep_type = np.empty(2 * n_pts, dtype=np.float64)
        for i in range(n_pts):
            ep_val[i]          = bmin[i]
            ep_type[i]         = 1.0
            ep_val[n_pts + i]  = bmax[i]
            ep_type[n_pts + i] = -1.0

        order = np.argsort(ep_val)
        sorted_val  = np.empty(2 * n_pts, dtype=np.float64)
        sorted_type = np.empty(2 * n_pts, dtype=np.float64)
        for i in range(2 * n_pts):
            sorted_val[i]  = ep_val[order[i]]
            sorted_type[i] = ep_type[order[i]]

        # Build cumulative-type array
        cum_types = np.empty(2 * n_pts, dtype=np.int64)
        cum = 0
        for i in range(2 * n_pts):
            cum += int(sorted_type[i])
            cum_types[i] = cum

        # Walk segments that achieve nselmax and scan b grid
        for k in range(2 * n_pts - 1):
            if cum_types[k] != nselmax:
                continue
            b_lo = sorted_val[k]
            b_hi = sorted_val[k + 1]
            if b_hi <= b_lo:
                continue

            n_b_steps = int((b_hi - b_lo) / b_step)
            if n_b_steps < 1:
                n_b_steps = 1

            for bi in range(n_b_steps + 1):
                b = b_lo + bi * b_step
                if b > b_hi:
                    b = b_hi

                chi2_sum = 0.0
                wgt_sum  = 0.0
                for i in range(n_pts):
                    res      = y[i] - sl * x[i] - b
                    chi2_val = res * res / eps[i]
                    if chi2_val <= chi2_thr:
                        chi2_sum += chi2_val * wgt[i]
                        wgt_sum  += wgt[i]

                if wgt_sum > 0.0:
                    totchi = chi2_sum / wgt_sum
                    if totchi < chi2min:
                        chi2min = totchi
                        m0 = m
                        b0 = b

    return m0, b0


def update_chirp_mass_statistics(cluster: Cluster, xgb_rho_mode: bool = False, pat0: bool = False):
    """Python implementation of C++ netcluster::mchirp().

    Computes chirpEllip and chirpEfrac via Hough-transform + PCA ellipticity
    on the cluster's pixels (which must have .likelihood already set by
    populate_detection_statistics).  Updates cluster.cluster_meta.net_rho2 with
    rho1 = rho0 * chirpEllip * sqrt(chirpEfrac), matching netevent.cc line 977:
        rho[1] = pcd->netRHO * chirp[3] * sqrt(chirp[5])   (pat0=false branch)

    net_rho2 is only updated for original 2G mode (xgb_rho_mode=False)
    with pat0=False.  In XGB mode or pat0=True the value set by
    populate_detection_statistics is preserved (mirrors netevent.cc lines 974-981).
    """
    import math

    # --- C++ watconstants (same as in netcluster::mchirp, from constants.hh) ---
    G  = 6.67259e-11        # WAT_G_SI: gravitational constant [N m^2 kg^-2]
    SM = 1.98892e30         # solar mass [kg]
    C  = 299792458.0        # speed of light [m/s]
    Pi = math.pi
    sF = 128.0              # frequency scaling (units of 128 Hz)
    chi2_thr = 2.5          # default threshold

    kk = 256.0 * Pi / 5.0 * math.pow(G * SM * Pi / (C * C * C), 5.0 / 3.0)
    kk *= math.pow(sF, 8.0 / 3.0)

    # --- Collect pixels (vectorised — no per-pixel object construction) ---
    _pa = cluster.pixel_arrays
    _valid = (_pa.likelihood > 0.0) & (_pa.frequency > 0)

    _rate_v   = _pa.rate[_valid].astype(float)
    _layers_v = _pa.layers[_valid].astype(float)
    _time_v   = _pa.time[_valid].astype(float)
    _freq_v   = _pa.frequency[_valid].astype(float)
    _lh_v     = _pa.likelihood[_valid].astype(float)

    T_v   = np.floor(_time_v / _layers_v) / _rate_v
    eT_v  = (0.5 / _rate_v) * math.sqrt(2.0)

    F_raw_v = _freq_v * _rate_v / 2.0 / sF
    _pos = F_raw_v > 0.0
    T_v, eT_v, F_raw_v, _rate_v, _lh_v = (T_v[_pos], eT_v[_pos], F_raw_v[_pos],
                                            _rate_v[_pos], _lh_v[_pos])

    eF_v = (_rate_v / 4.0 / math.sqrt(3.0)) / sF
    eF_v *= 8.0 / 3.0 / np.power(F_raw_v, 11.0 / 3.0)
    F_t_v = 1.0 / np.power(F_raw_v, 8.0 / 3.0)

    np_pts = len(T_v)
    if np_pts < 5:
        return  # insufficient pixels — leave net_rho2 unchanged

    x    = T_v
    y    = F_t_v
    xerr = eT_v
    yerr = eF_v
    wgt  = _lh_v

    # --- Hough transform: find mass(es) with maximum pixel-overlap ---
    maxM     = 100.0
    stepM    = 0.2
    m_vals   = np.arange(-maxM, maxM + 1e-9, stepM)   # 1001 values

    # Phase 1: parallel Numba scan — O(n_mass * n_pts * log n_pts) with prange
    nsel_arr = _count_chirp_track_overlaps_numba(
        x.astype(np.float64), y.astype(np.float64),
        xerr.astype(np.float64), yerr.astype(np.float64),
        float(kk), m_vals.astype(np.float64),
    )

    nselmax      = int(np.max(nsel_arr))
    cand_indices = np.where(nsel_arr == nselmax)[0].astype(np.int64)

    # Phase 2: fine b-grid search over candidate masses — tight Numba inner loop
    m0, b0 = _fit_chirp_track_candidates_numba(
        x.astype(np.float64), y.astype(np.float64),
        xerr.astype(np.float64), yerr.astype(np.float64),
        wgt.astype(np.float64),
        float(kk), m_vals.astype(np.float64),
        cand_indices, int(nselmax), float(chi2_thr),
    )

    # --- Compute Efrac ---
    sl  = kk * math.pow(abs(m0), 5.0 / 3.0)
    if m0 > 0:
        sl = -sl

    eps = sl * sl * xerr * xerr + yerr * yerr
    residuals = y - sl * x - b0
    chi2_all  = residuals * residuals / eps
    sel_mask  = chi2_all <= chi2_thr

    totEn = float(np.sum(wgt))
    selEn = float(np.sum(wgt[sel_mask]))
    Efrac = selEn / totEn if totEn > 0.0 else 0.0

    # --- Filter to selected pixels and compute PCA ellipticity ---
    x_sel = x[sel_mask]
    y_sel = y[sel_mask]
    np_sel = len(x_sel)

    if np_sel >= 2:
        xcm = np.mean(x_sel)
        ycm = np.mean(y_sel)
        dx  = x_sel - xcm
        dy  = y_sel - ycm
        qxx = float(np.sum(dx * dx))
        qyy = float(np.sum(dy * dy))
        qxy = float(np.sum(dx * dy))

        sq_delta = math.sqrt((qxx - qyy) ** 2 + 4.0 * qxy * qxy)
        lam1 = math.sqrt((qxx + qyy + sq_delta) / 2.0)
        lam2_sq = (qxx + qyy - sq_delta) / 2.0
        lam2 = math.sqrt(max(lam2_sq, 0.0))
        denom = lam1 + lam2
        chirpEllip = abs(lam1 - lam2) / denom if denom > 0.0 else 0.0
    else:
        chirpEllip = 0.0

    # --- Update cluster metadata ---
    chrho = chirpEllip * math.sqrt(Efrac)
    rho1  = cluster.cluster_meta.net_rho * chrho

    # C++ netevent.cc lines 974-981:
    #   chrho = chirp[3] * sqrt(chirp[5])                    (always computed)
    #   if netRHO >= 0 (original 2G):
    #       rho[1] = pat0 ? netrho : netRHO * chrho           (only pat0=false uses chirp)
    #   else (XGB.rho0):
    #       rho[1] = netrho                                   (chirp result ignored)
    if not xgb_rho_mode and not pat0:
        cluster.cluster_meta.net_rho2 = rho1
    # else: net_rho2 already set correctly by populate_detection_statistics; do not overwrite



# Legacy aliases
threshold_cut = get_likelihood_rejection_reason
fill_detection_statistic = populate_detection_statistics
get_chirp_mass = update_chirp_mass_statistics
get_error_region = compute_sky_error_region
_hough_count_overlaps_numba = _count_chirp_track_overlaps_numba
_fine_search_numba = _fit_chirp_track_candidates_numba

__all__ = [
    "threshold_cut", "fill_detection_statistic", "get_chirp_mass",
    "get_error_region",
    "get_likelihood_rejection_reason", "populate_detection_statistics",
    "update_chirp_mass_statistics", "compute_sky_error_region",
]
