"""Event reconstruction, detection cuts, and sky-localization updates.

populate_detection_statistics reconstructs detector waveforms and populates
cluster/pixel event quantities. get_likelihood_rejection_reason evaluates cuts;
populate_sky_localization attaches posterior probabilities and error regions.
Hough chirp fitting and its cluster updates live in chirp_hough.py.
"""

from __future__ import annotations
from pycwb.constants.execution_profile import execution_profile

import logging
from math import sqrt
import time
import numpy as np
from pycwb.config.config import Config
from pycwb.types.network_cluster import Cluster
from pycwb.modules.xtalk.monster import _compute_null_likelihood_numba
from pycwb.modules.xtalk.type import XTalk
from pycwb.modules.reconstruction.getMRAwaveform import (
    _create_wdm_set_python,
    get_MRA_wave,
    _pa_to_tuple,
    _build_wdm_njit_data,
)
from .results import SkyStatistics, SkyMapStatistics

logger = logging.getLogger(__name__)


def populate_detection_statistics(
    sky_statistics: SkyStatistics,
    skymap_statistics: SkyMapStatistics,
    cluster: Cluster,
    n_ifo: int,
    xtalk: XTalk,
    network_energy_threshold: float,
    xgb_rho_mode: bool = False,
    config: Config = None,
    cluster_xtalk: np.ndarray | None = None,
    cluster_xtalk_lookup: np.ndarray | None = None,
    wdm_list=None,
) -> None:
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

    event_size = 0  # defined as Mw in cwb

    # --- First pass: set core/likelihood/null flags and per-ifo data arrays ---
    n_pixels = len(cluster.pixel_arrays)

    _t0 = time.perf_counter()
    # Fast path: vectorised update of pixel_arrays (avoids O(n_pixels * n_ifo) Python loop).
    _pa = cluster.pixel_arrays
    _pa.set_waveform_data(
        wave=np.asarray(packet_data_phase0, dtype=np.float32),
        w_90=np.asarray(packet_data_phase90, dtype=np.float32),
        asnr=np.asarray(packet_signal_phase0, dtype=np.float32),
        a_90=np.asarray(packet_signal_phase90, dtype=np.float32),
        core_mask=pixel_mask,
        energy_plus=np.asarray(energy_array_plus, dtype=np.float32),
        energy_cross=np.asarray(energy_array_cross, dtype=np.float32),
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
    # cWB skips satellites before likelihood writeback. Their negative energy
    # markers must survive, even when their coherent energy is positive.
    likelihood_pixel_indices = np.where(
        _core & (np.asarray(gaussian_noise_correction) > 0) & (np.asarray(coherent_energy) > 0)
    )[0].astype(np.int64)
    stage_timings["set_waveform_data"] = time.perf_counter() - _t0

    # --- Second pass: compute null and likelihood using the parallel numba kernel ---
    logger.debug(
        "populate_detection_statistics: null_pixel_indices size=%d, likelihood_pixel_indices size=%d, n_pixels=%d",
        len(null_pixel_indices),
        len(likelihood_pixel_indices),
        n_pixels,
    )
    logger.debug(
        "populate_detection_statistics: null_phase0_arr shape=%s, range=[%g, %g]",
        str(null_phase0_arr.shape),
        float(np.min(np.abs(null_phase0_arr))),
        float(np.max(np.abs(null_phase0_arr))),
    )
    logger.debug(
        "populate_detection_statistics: gn range=[%g, %g], ec range=[%g, %g]",
        float(np.min(gaussian_noise_correction)),
        float(np.max(gaussian_noise_correction)),
        float(np.min(coherent_energy)),
        float(np.max(coherent_energy)),
    )

    # null_out and like_out are written in place for the relevant pixel indices.
    # Initialise to zero so pixels not in the respective sets keep their old value
    # (matches behaviour of the previous Python loops).
    null_out = np.zeros(n_pixels, dtype=np.float64)
    like_out = np.zeros(n_pixels, dtype=np.float64)

    gn_arr = np.asarray(gaussian_noise_correction, dtype=np.float64)
    ec_arr = np.asarray(coherent_energy, dtype=np.float64)

    # Cross-talk neighbors have a broader likelihood gate than writeback:
    # cWB requires core & ec > 0 for neighbors, without the outer gn > 0 cut.
    null_mask = np.zeros(n_pixels, dtype=np.bool_)
    null_mask[null_pixel_indices] = True
    like_mask = np.zeros(n_pixels, dtype=np.bool_)
    like_mask[:] = _core & (np.asarray(coherent_energy) > 0)

    _t0 = time.perf_counter()
    _compute_null_likelihood_numba(
        null_pixel_indices,
        likelihood_pixel_indices,
        null_phase0_arr,
        null_phase90_arr,
        signal_phase0_arr,
        signal_phase90_arr,
        gn_arr,
        ec_arr,
        xtalks_lookup.astype(np.int64),
        xtalks,
        null_mask,
        like_mask,
        null_out,
        like_out,
    )
    _kernel_time = time.perf_counter() - _t0

    # Write results back into pixel_arrays
    for i in null_pixel_indices:
        _pa.null[i] = null_out[i]
    for i in likelihood_pixel_indices:
        _pa.likelihood[i] = like_out[i]

    # Count statistics (sets were pre-filtered, so counts equal set sizes)
    event_size = int(len(null_pixel_indices))

    stage_timings["null_xtalk_loop"] = (
        _kernel_time * len(null_pixel_indices) / max(len(null_pixel_indices) + len(likelihood_pixel_indices), 1)
    )
    stage_timings["likelihood_xtalk_loop"] = (
        _kernel_time * len(likelihood_pixel_indices) / max(len(null_pixel_indices) + len(likelihood_pixel_indices), 1)
    )

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
    release_waveform_stats = execution_profile(config).release_waveform_stats
    from .waveform_statistics import (
        waveform_rms,
        waveform_time,
        waveform_frequency,
        network_centroids,
        compute_final_detection_statistics,
    )

    wf_times = np.zeros(n_ifo)
    wf_frequencies = np.zeros(n_ifo)
    cross_rms = np.zeros(n_ifo)
    Lw = 0.0
    sSNR_ifo = np.zeros(n_ifo, dtype=np.float64)
    snr_ifo = np.zeros(n_ifo, dtype=np.float64)
    null_ifo = np.zeros(n_ifo, dtype=np.float64)
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
    _pixel_arrays = _pa_to_tuple(cluster.pixel_arrays)
    _wdm_njit_data = _build_wdm_njit_data(wdm_list)

    for ifo_i in range(n_ifo):
        z_sig_ts = get_MRA_wave(
            cluster,
            wdm_list,
            rate_ana,
            ifo_i,
            a_type="signal",
            mode=0,
            nproc=1,
            whiten=True,
            _pixel_arrays=_pixel_arrays,
            _wdm_njit_data=_wdm_njit_data,
        )
        z_dat_ts = get_MRA_wave(
            cluster,
            wdm_list,
            rate_ana,
            ifo_i,
            a_type="strain",
            mode=0,
            nproc=1,
            whiten=True,
            _pixel_arrays=_pixel_arrays,
            _wdm_njit_data=_wdm_njit_data,
        )
        # For hrss: get un-whitened signal energy (physical strain units)
        z_sig_physical = get_MRA_wave(
            cluster,
            wdm_list,
            rate_ana,
            ifo_i,
            a_type="signal",
            mode=0,
            nproc=1,
            whiten=False,
            _pixel_arrays=_pixel_arrays,
            _wdm_njit_data=_wdm_njit_data,
        )
        if z_sig_ts is None or z_dat_ts is None:
            continue
        z_sig = np.asarray(z_sig_ts.data, dtype=np.float64)
        z_dat = np.asarray(z_dat_ts.data, dtype=np.float64)
        sSNR_ifo[ifo_i] = np.sum(z_sig**2)
        snr_ifo[ifo_i] = np.sum(z_dat**2)
        null_ifo[ifo_i] = np.sum((z_dat - z_sig) ** 2)
        if release_waveform_stats:
            rs, rd, rn = waveform_rms(z_sig), waveform_rms(z_dat), waveform_rms(z_dat - z_sig)
            sSNR_ifo[ifo_i] = np.float32(rs * rs * len(z_sig))
            snr_ifo[ifo_i] = np.float32(rd * rd * len(z_dat))
            null_ifo[ifo_i] = np.float32(rn * rn * len(z_dat))
            cross_rms[ifo_i] = np.float32(rd * rs * len(z_dat))
        if z_sig_physical is not None:
            z_sig_phys = np.asarray(z_sig_physical.data, dtype=np.float64)
            signal_energy_physical[ifo_i] = np.sum(z_sig_phys**2)

        # getWFtime() / getWFfreq() equivalents (mirrors C++ detector::getWFtime/getWFfreq)
        # Used to compute To/Fo exactly as C++: Fo += sSNR_i * getWFfreq_i; To /= Lw
        n_fft = len(z_sig)
        rate_wf = float(z_sig_ts.sample_rate)
        e_sig = z_sig**2
        E_sig = float(np.sum(e_sig))
        if E_sig > 0.0:
            t_start = float(z_sig_ts.start_time)
            if release_waveform_stats:
                wf_time_ifo = waveform_time(z_sig, t_start, rate_wf)
                wf_freq_ifo = waveform_frequency(z_sig, rate_wf)
                wf_times[ifo_i] = wf_time_ifo
                wf_frequencies[ifo_i] = wf_freq_ifo
            else:
                wf_time_ifo = t_start + float(np.dot(e_sig, np.arange(n_fft))) / (E_sig * rate_wf)
                Z_fft = np.fft.rfft(z_sig)
                power = Z_fft.real**2 + Z_fft.imag**2
                E_fft = float(np.sum(power))
                wf_freq_ifo = (
                    float(np.dot(power, np.arange(len(power)))) * rate_wf / n_fft / E_fft if E_fft > 0.0 else 0.0
                )
            To += sSNR_ifo[ifo_i] * wf_time_ifo
            Fo += sSNR_ifo[ifo_i] * wf_freq_ifo

    Lw = float(np.sum(sSNR_ifo))
    Ew_wf = float(np.sum(snr_ifo))
    Nw_wf = float(np.sum(null_ifo))
    if Lw > 0.0:
        To /= Lw
        Fo /= Lw

    if release_waveform_stats:
        Lw, To, Fo = map(float, network_centroids(sSNR_ifo, wf_times, wf_frequencies))
        Ew_wf = float(np.add.accumulate(snr_ifo.astype(np.float32))[-1]) if n_ifo else 0.0
        Nw_wf = float(np.add.accumulate(null_ifo.astype(np.float32))[-1]) if n_ifo else 0.0

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
    xSNR_ifo = cross_rms if release_waveform_stats else np.sqrt(np.maximum(snr_ifo * sSNR_ifo, 0.0))

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
    #   norm = (Eo-Eh) / Ew_wf, stored as norm*2 (no final floor)
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
    # The earlier sky-loop floor does not apply to final waveform normalization.

    # rho is divided by sqrt(cc) using Nw-based chi2 (time-domain null, matches C++ line 939)
    cc_rho_td = ch_td if ch_td > 1.0 else 1.0
    rho_reduced = float(sky_statistics.rho) / sqrt(cc_rho_td)
    stage_timings["detection_statistics"] = time.perf_counter() - _t0

    # --- Store all fields on cluster_meta ---
    _t0 = time.perf_counter()
    cluster.cluster_meta.sky_size = event_size
    cluster.cluster_meta.sub_net = Esub / (Esub + Nmax) if (Esub + Nmax) > 0 else 0.0
    cluster.cluster_meta.sub_net2 = skymap_statistics.nCorrelation[skymap_statistics.l_max]
    cluster.cluster_meta.like_sky = float(sky_statistics.Em)  # Em (neted[3]): pixel-domain xtalk energy
    cluster.cluster_meta.energy_sky = sky_statistics.Eo  # TF-domain data energy (neted[4])
    cluster.cluster_meta.net_ecor = sky_statistics.Ec  # packet coherent energy
    cluster.cluster_meta.norm_cor = sky_statistics.Ec * sky_statistics.Rc  # normalised coherent energy
    cluster.cluster_meta.like_net = float(Lw)  # waveform likelihood (likenet)
    cluster.cluster_meta.energy = float(Ew_wf)  # getMRAwave data energy (neted[2])
    cluster.cluster_meta.net_null = float(Nw_for_stats + Gn_val)  # packet null (neted[1])
    cluster.cluster_meta.net_ed = float(Nw_for_stats + Gn_val + Dc - N_eff * n_ifo)  # residual null (neted[0])
    cluster.cluster_meta.norm = float(norm_td * 2.0)  # packet norm
    cluster.cluster_meta.net_cc = float(Cp_td)  # network cc (netcc[0])
    cluster.cluster_meta.sky_cc = float(Cr_td)  # reduced network cc (netcc[1])
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

    if release_waveform_stats:
        final = compute_final_detection_statistics(
            Eo, Eh, Ew_wf, Nw_wf, Gn_val, Dc, Ec, Rc_val, N_eff, n_ifo, sky_statistics.rho, sky_statistics.xrho
        )
        meta = cluster.cluster_meta
        meta.net_null = final["null"]
        meta.net_ed = final["residual"]
        meta.norm = final["norm"]
        meta.net_cc = final["cp"]
        meta.sky_cc = final["cr"]
        meta.norm_cor = final["norm_cor"]
        meta.net_rho2 = final["xrho_reduced"] if xgb_rho_mode else float(np.float32(sky_statistics.rho))
        if not xgb_rho_mode:
            meta.net_rho = final["rho_reduced"]

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
    cluster.cluster_meta.signal_snr = sSNR_ifo.tolist()  # C++ d->sSNR = get_SS() per IFO
    cluster.cluster_meta.wave_snr = snr_ifo.tolist()  # C++ d->enrg = get_XX() per IFO
    cluster.cluster_meta.cross_snr = xSNR_ifo.tolist()  # C++ d->xSNR = get_XS() per IFO
    cluster.cluster_meta.signal_energy_physical = signal_energy_physical.tolist()  # physical strain energy for hrss
    cluster.cluster_meta.null_energy = null_ifo.tolist()  # null energy per IFO (C++ d->null)

    logger.debug(
        "populate_detection_statistics: sky_size=%d sub_net=%.4f net_cc=%.4f sky_cc=%.4f "
        "like_net=%.2f energy=%.2f net_null=%.4f norm=%.4f rho=%.4f "
        "Ew_wf=%.2f Nw_wf=%.4f like_sky=%.2f",
        cluster.cluster_meta.sky_size,
        cluster.cluster_meta.sub_net,
        cluster.cluster_meta.net_cc,
        cluster.cluster_meta.sky_cc,
        cluster.cluster_meta.like_net,
        cluster.cluster_meta.energy,
        cluster.cluster_meta.net_null,
        cluster.cluster_meta.norm,
        cluster.cluster_meta.net_rho,
        Ew_wf,
        Nw_wf,
        cluster.cluster_meta.like_sky,
    )

    stage_timings["store_cluster_meta"] = time.perf_counter() - _t0
    stage_timings["total"] = time.perf_counter() - timing_start
    logger.info("populate_detection_statistics stage timings:")
    for _stage, _t in stage_timings.items():
        if _stage != "total":
            logger.info(
                "  %-30s %.4f s  (%5.1f%%)",
                _stage,
                _t,
                100.0 * _t / stage_timings["total"] if stage_timings["total"] > 0 else 0,
            )


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
    N = sky_statistics.N_pix_effective  # effective pixel count (_avx_setAMP_ps() - 1)
    if not xgb_rho_mode:
        condition_1 = Lm <= 0.0
        condition_2 = (Eo - Eh) <= 0.0
        condition_3 = Ec * Rc / cc < netEC_threshold
        condition_4 = N < 1  # C++: N < 1 (pixel count, not null energy)
        if condition_1 or condition_2 or condition_3 or condition_4:
            rejection_reason = ""
            if condition_1:
                rejection_reason += f"Lm > 0 but Lm = {Lm};"
            if condition_2:
                rejection_reason += f" (Eo - Eh) > 0 but (Eo - Eh) = {Eo - Eh};"
            if condition_3:
                rejection_reason += (
                    f" Ec * Rc / cc >= netEC_threshold but Ec * Rc / cc = {Ec * Rc / cc:.4f} < {netEC_threshold:.4f};"
                )
            if condition_4:
                rejection_reason += f" N < 1 but N = {N};"
            return rejection_reason
    else:
        # For XGB.rho0 case C++ uses `rho < fabs(netRHO)` directly.
        if net_rho_threshold is None:
            net_rho_threshold = (netEC_threshold / 2.0) ** 0.5
        condition_1 = Lm <= 0.0
        condition_2 = (Eo - Eh) <= 0.0
        condition_3 = not np.isfinite(rho) or rho < net_rho_threshold
        condition_4 = N < 1  # C++: N < 1 (pixel count)
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


def populate_sky_localization(cluster: Cluster, skymap_statistics=None, sky_statistics=None, config=None):
    """Populate background sky probabilities and area deciles after reconstruction."""
    if skymap_statistics is None or sky_statistics is None:
        return None
    from .sky_localization import localize_sky

    detection_index = int(skymap_statistics.l_max)
    statistic = skymap_statistics.nLikelihood if getattr(config, "delta", 0.5) < 0 else skymap_statistics.nSkyStat
    # Mo is the count returned by the GW projection, not Mw/sky_size (which
    # counts pixels surviving later waveform writeback gates).
    mo = int(np.count_nonzero(sky_statistics.pixel_mask > 0))
    chi2 = float(skymap_statistics.nDisbalance[detection_index])
    scale = float(cluster.cluster_meta.norm) / 2.0 * float(sky_statistics.Rc) * np.sqrt(mo) * (1.0 + abs(1.0 - chi2))
    if execution_profile(config).release_waveform_stats:
        from .waveform_statistics import compute_sky_posterior_scale

        scale = compute_sky_posterior_scale(cluster.cluster_meta.norm / 2.0, sky_statistics.Rc, mo, chi2)
    result = localize_sky(
        statistic,
        skymap_statistics.nAntennaPrior,
        scale,
        use_prior=getattr(config, "gamma", getattr(config, "cfg_gamma", 0.0)) < 0,
        n_sky=getattr(config, "nSky", 0),
    )
    if result is None:
        skymap_statistics.nProbability = None
        return None
    cluster.sky_area = result.error_regions.tolist()
    cluster.sky_pixel_index = result.indices.tolist()
    cluster.sky_pixel_map = result.selected_probability.tolist()
    skymap_statistics.nProbability = result.probability
    return result

__all__ = [
    "get_likelihood_rejection_reason",
    "populate_detection_statistics",
    "populate_sky_localization",
]
