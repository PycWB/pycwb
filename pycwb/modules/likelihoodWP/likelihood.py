from math import sqrt
import logging
import time
import numpy as np
from numba import njit, prange, float32
from numba.typed import List
from wdm_wavelet.wdm import WDM as WDMWavelet
from pycwb.types.network_cluster import Cluster
from pycwb.types.time_series import TimeSeries
from pycwb.types.detector import compute_sky_delay_and_patterns, _build_sky_directions
from .dpf import calculate_dpf, dpf_np_loops_vec
from .sky_stat import avx_GW_ps, avx_ort_ps, avx_stat_ps, load_data_from_td
from .utils import avx_packet_ps, packet_norm_numpy, gw_norm_numpy, avx_noise_ps, \
        avx_setAMP_ps, avx_pol_ps, avx_loadNULL_ps
from .pixel_batch_ops import load_data_from_pixels_vectorized, batch_ensure_td_amp
from pycwb.modules.xtalk.type import XTalk
from .typing import SkyStatistics, SkyMapStatistics


logger = logging.getLogger(__name__)


def _expected_td_vec_len(td_size):
    return 4 * int(td_size) + 2


def _normalize_wdm_layers(layer_tag):
    layer_tag = int(layer_tag)
    if layer_tag <= 1:
        return 1
    candidate = layer_tag - 1
    return candidate if candidate % 2 == 0 else layer_tag


def _normalize_strains(strains):
    normalized = []
    for strain in strains:
        if isinstance(strain, TimeSeries):
            normalized.append(strain)
        elif hasattr(strain, "data") and isinstance(getattr(strain, "data"), TimeSeries):
            normalized.append(getattr(strain, "data"))
        else:
            normalized.append(TimeSeries.from_input(strain))
    return normalized


def _resolve_runtime_parameters(config, nIFO):
    if config is None:
        raise ValueError("config is required for pure-Python likelihood")
    acor = float(getattr(config, "Acore"))
    gamma = float(getattr(config, "gamma", 0.0))
    delta = float(getattr(config, "delta", 0.0))
    net_rho = float(getattr(config, "netRHO", 0.0))
    net_cc = float(getattr(config, "netCC", 0.0))

    network_energy_threshold = 2 * acor * acor * nIFO
    gamma_regulator = gamma * gamma * 2 / 3
    delta_regulator = abs(delta) if abs(delta) < 1 else 1
    netEC_threshold = net_rho * net_rho * 2
    return network_energy_threshold, gamma_regulator, delta_regulator, netEC_threshold, net_cc


def _ensure_td_amp(cluster, nIFO, strains=None, config=None, wdm_td_cache=None):
    if len(cluster.pixels) == 0:
        return False

    has_td = True
    for pixel in cluster.pixels:
        td_amp = getattr(pixel, "td_amp", None)
        if td_amp is None or len(td_amp) < nIFO:
            has_td = False
            break

    if has_td:
        return False

    if strains is None or config is None:
        raise ValueError("likelihood requires `strains` and `config` when cluster pixels do not contain td_amp")

    # Batch JAX extraction replaces the per-pixel serial loop
    return batch_ensure_td_amp(cluster, nIFO, strains, config, wdm_td_cache=wdm_td_cache)

def _populate_pixel_noise_rms(pixels, nRMS):
    """
    Populate each ``pixel.data[i].noise_rms`` from the per-IFO TF noise maps.

    The nRMS maps come from the highest-resolution whitening step.  For pixels at
    other resolutions the frequency bin is scaled proportionally to the nRMS grid.

    Parameters
    ----------
    pixels : list[Pixel]
        Cluster pixels.
    nRMS : list[TimeFrequencyMap]
        One TF noise map per IFO from whitening_python.  ``data`` shape is
        ``(n_freq_bins, n_time_bins)`` where n_freq_bins covers [0, fNyq].
    """
    n_ifo = len(nRMS)
    # Precompute nRMS data arrays once
    nrms_data = []
    nrms_shapes = []
    for i in range(n_ifo):
        arr = np.asarray(nRMS[i].data, dtype=np.float64)
        nrms_data.append(arr)
        nrms_shapes.append(arr.shape)  # (n_freq, n_time)

    for pixel in pixels:
        freq_bin = int(pixel.frequency)
        n_freq_pix = int(pixel.layers)  # number of frequency bins at this resolution
        # Derive time bin from composite pixel.time = time_idx * n_freq + freq_bin
        if n_freq_pix > 0:
            time_bin_pix = int(pixel.time) // n_freq_pix
        else:
            time_bin_pix = 0

        for i in range(n_ifo):
            try:
                nf, nt = nrms_shapes[i]
                # Map pixel freq_bin (at resolution n_freq_pix) to nRMS freq bin
                if n_freq_pix > 0 and nf > 0:
                    fb = int(round(freq_bin * nf / n_freq_pix))
                    fb = min(max(fb, 0), nf - 1)
                else:
                    fb = 0
                # Map time bin 
                tb = min(time_bin_pix, nt - 1) if nt > 0 else 0
                val = float(np.abs(nrms_data[i][fb, tb]))
                if val > 0.0:
                    pixel.data[i].noise_rms = val
            except Exception:
                pass  # keep noise_rms=1.0 on failure


def likelihood(nIFO, cluster, MRAcatalog, strains=None, config=None, ml=None, FP=None, FX=None, cluster_id=None,
               wdm_td_cache=None, nRMS=None):
    """
    Main function to calculate the likelihood for a given network and cluster.

    Args:
        nIFO (int): Number of interferometers.
        cluster (Cluster): The cluster object containing pixel data.
        MRAcatalog (str): Path to the MRA catalog for xtalk information.
        wdm_td_cache (dict | None): Optional pre-built TD-input cache from
            supercluster_wrapper (Priority 2 optimisation).  When provided,
            ``batch_ensure_td_amp`` skips the expensive WDM context rebuild.
        nRMS (list | None): List of TimeFrequencyMap objects (one per IFO) from
            data conditioning.  When provided, each pixel's ``noise_rms`` field
            is populated from the TF map so that downstream physical-unit
            quantities (hrss, noise) are correct.

    Returns:
        tuple[Cluster | None, SkyMapStatistics | None]: 
            The updated cluster object with filled detection statistics and the full skymap statistics.
            Returns ``(None, None)`` if the cluster is rejected by threshold cuts.
    """
    timer_start = time.perf_counter()
    logger.info("-------------------------------------------------------")
    logger.info("-> Processing cluster-id=%d|pixels=%d", int(cluster_id) if cluster_id is not None else -1, len(cluster.pixels))
    logger.info("   ----------------------------------------------------")

    t_tdamp = time.perf_counter()
    td_amp_reloaded = _ensure_td_amp(cluster, nIFO, strains=strains, config=config, wdm_td_cache=wdm_td_cache)
    setattr(cluster, "_td_amp_reloaded", bool(td_amp_reloaded))
    logger.info("td_amp reload: %s  (%.4f s)", bool(td_amp_reloaded), time.perf_counter() - t_tdamp)

    # Populate pixel noise_rms from the nRMS TF maps so downstream physical-unit quantities
    # (hrss, noise) are correct.  Each pixel stores the noise floor at its (freq_bin, time_bin).
    if nRMS is not None and len(nRMS) == nIFO:
        _populate_pixel_noise_rms(cluster.pixels, nRMS)

    # prepare runtime parameters
    network_energy_threshold, gamma_regulator, delta_regulator, netEC_threshold, netCC = _resolve_runtime_parameters(
        config, nIFO
    )

    REG = np.array([delta_regulator * np.sqrt(2), 0., 0.])
    n_pix = len(cluster.pixels)

    # Load xtalk catalog
    xtalk = XTalk.load(MRAcatalog, dump=True)
    cluster_xtalk_lookup, cluster_xtalk = xtalk.get_xtalk_pixels(cluster.pixels, True)

    # Extract data from python object to numpy arrays for numba
    ml, FP, FX = load_data_from_ifo(
        nIFO=nIFO,
        strains=strains,
        config=config,
        ml=ml,
        FP=FP,
        FX=FX,
    )
    n_sky = int(ml.shape[1])

    rms, td00, td90, td_energy = load_data_from_pixels(cluster.pixels, nIFO)

    # Transpose array and convert to float32 for speedup
    td00 = np.transpose(td00.astype(np.float32), (2, 0, 1))  # (ndelay, nifo, npix)
    td90 = np.transpose(td90.astype(np.float32), (2, 0, 1))  # (ndelay, nifo, npix)
    FP = FP.T.astype(np.float32)
    FX = FX.T.astype(np.float32)
    rms = rms.T.astype(np.float32)

    # Note: What are the two regulators for?
    REG[1] = calculate_dpf(FP, FX, rms, n_sky, nIFO, gamma_regulator, network_energy_threshold)

    # loop over the sky locations to find the optimal sky localization, 
    # l_max and sky statistics will be returned in tuple due to the limitations of numba
    skymap_statistics = find_optimal_sky_localization(nIFO, n_pix, n_sky, FP, FX, rms, td00, td90, ml, REG, netCC,
                                          delta_regulator, network_energy_threshold)
    # Convert the tuple to SkyMapStatistics dataclass for better structure and IDE friendly
    skymap_statistics = SkyMapStatistics.from_tuple(skymap_statistics)

    # Compute normalised sky probability from nSkyStat (softmax)
    _sky_stat_f64 = skymap_statistics.nSkyStat.astype(np.float64)
    _sky_stat_shifted = _sky_stat_f64 - _sky_stat_f64.max()
    _exp_stat = np.exp(_sky_stat_shifted)
    skymap_statistics.nProbability = (_exp_stat / _exp_stat.sum()).astype(np.float32)

    # Compute sky direction angles at l_max from the same grid used by load_data_from_ifo
    _healpix_order = int(getattr(config, 'healpix', 0)) if hasattr(config, 'healpix') else None
    _ra_arr, _dec_arr = _build_sky_directions(n_sky, _healpix_order)
    _l_max = int(skymap_statistics.l_max)
    # cWB theta: co-latitude in degrees [0, 180]; phi: longitude in degrees [0, 360)
    _theta_rad = float(np.pi / 2.0 - _dec_arr[_l_max])  # co-latitude from declination
    _phi_rad = float(_ra_arr[_l_max])
    _theta_deg = float(np.degrees(_theta_rad)) % 180.0
    _phi_deg = float(np.degrees(_phi_rad)) % 360.0

    # calculate sky statistics for the cluster at the optimal sky location l_max,
    # dozens of parameters will be returned in SkyStatistics dataclass
    sky_statistics: SkyStatistics = calculate_sky_statistics(skymap_statistics.l_max, nIFO, n_pix, 
                                                             FP, FX, rms, td00, td90, ml, REG, 
                                                             network_energy_threshold, 
                                                             cluster_xtalk, cluster_xtalk_lookup)

    # Check if the cluster is rejected based on the threshold cuts, 
    # the function will return the reason for rejection. If the cluster is not rejected, it will return None.
    selected_core_pixels = int(np.count_nonzero(np.asarray(sky_statistics.pixel_mask) > 0))
    logger.info("Selected core pixels: %d / %d", selected_core_pixels, n_pix)

    rejected = threshold_cut(sky_statistics, network_energy_threshold, netEC_threshold)
    if rejected:
        logger.info("Cluster rejected due to threshold cuts: %s", rejected)
        logger.info("   cluster-id|pixels: %5d|%d", int(cluster_id) if cluster_id is not None else -1, len(cluster.pixels))
        logger.info("\t <- rejected    ")
        timer_end = time.perf_counter()
        logger.info("-------------------------------------------------------")
        logger.info("Total events: %d", 0)
        logger.info("Total time: %.2f s", timer_end - timer_start)
        logger.info("-------------------------------------------------------")
        return None, None

    # Fill the detection statistics into the cluster and pixels for return
    fill_detection_statistic(sky_statistics, skymap_statistics, cluster=cluster, 
                             n_ifo=nIFO, xtalk=xtalk,
                             network_energy_threshold=network_energy_threshold)
    
    # Placeholder: Get the chirp mass
    get_chirp_mass(cluster)

    # Placeholder: Get the error region
    get_error_region(cluster)

    # Store sky localisation metadata on cluster for downstream Event construction
    cluster.cluster_meta.l_max = _l_max
    cluster.cluster_meta.theta = _theta_deg
    cluster.cluster_meta.phi = _phi_deg
    # c_time / c_freq from supercluster if not yet set by fill_detection_statistic
    if cluster.cluster_meta.c_time == 0.0:
        cluster.cluster_meta.c_time = cluster.cluster_time
    if cluster.cluster_meta.c_freq == 0.0:
        cluster.cluster_meta.c_freq = cluster.cluster_freq
    # Time delay indices at l_max — one entry per IFO (used by getMRAwaveform for ToF correction)
    # ml has shape (nIFO, n_sky) here; the transposed copy is a different variable
    # Retrieve the original (pre-transpose) ml via the closure variable
    cluster.sky_time_delay = [float(ml[i, _l_max]) for i in range(nIFO)]

    detected = cluster.cluster_status == -1
    logger.info("   cluster-id|pixels: %5d|%d", int(cluster_id) if cluster_id is not None else -1, len(cluster.pixels))
    if detected:
        logger.info("\t -> SELECTED !!!")
    else:
        logger.info("\t <- rejected    ")

    timer_end = time.perf_counter()
    logger.info("-------------------------------------------------------")
    logger.info("Total events: %d", 1 if detected else 0)
    logger.info("Total time: %.2f s", timer_end - timer_start)
    logger.info("-------------------------------------------------------")

    return cluster, skymap_statistics


def load_data_from_pixels(pixels, nifo):
    """
    Load data from pixels into numpy arrays for numba processing.

    Delegates to the vectorised implementation in pixel_batch_ops which
    avoids the per-pixel Python loops and uses bulk numpy operations.
    """
    return load_data_from_pixels_vectorized(pixels, nifo)


def load_data_from_ifo(nIFO, strains=None, config=None, ml=None, FP=None, FX=None):
    """
    Load the sky delay/pattern data into numpy arrays for numba processing.
    Args:
        nIFO (int): Number of interferometers.

    Returns:
        tuple: ml, FP, FX
            - ml (np.ndarray): Array of indices for each sky location.
            - FP (np.ndarray): Array of f+ polarization data for each interferometer.
            - FX (np.ndarray): Array of fx polarization data for each interferometer.
    """
    if ml is not None and FP is not None and FX is not None:
        return np.asarray(ml), np.asarray(FP), np.asarray(FX)

    if strains is None or config is None:
        raise ValueError("strains and config are required when ml/FP/FX are not provided")

    normalized_strains = _normalize_strains(strains)
    gps_time = float(normalized_strains[0].t0)
    ml_arr, fp_arr, fx_arr = compute_sky_delay_and_patterns(
        ifos=getattr(config, "ifo"),
        ref_ifo=getattr(config, "refIFO"),
        sample_rate=float(getattr(config, "rateANA")),
        td_size=int(getattr(config, "TDSize")),
        gps_time=gps_time,
        healpix_order=int(getattr(config, "healpix", 0)) if hasattr(config, "healpix") else None,
        n_sky=None,
    )
    return ml_arr, fp_arr, fx_arr


@njit(cache=True, parallel=True)
def find_optimal_sky_localization(n_ifo, n_pix, n_sky, FP, FX, rms, td00, td90, ml, REG, netCC, delta_regulator, network_energy_threshold):
    """
    Find the optimal sky localization by calculating sky statistics for each sky location.
    
    Args:
        n_ifo (int): Number of interferometers.
        n_pix (int): Number of pixels.
        n_sky (int): Number of sky locations.
        FP (np.ndarray): f+ polarization data for each interferometer.
        FX (np.ndarray): fx polarization data for each interferometer.
        rms (np.ndarray): RMS values for each interferometer and pixel.
        td00 (np.ndarray): Time delayed data for 00 polarization.
        td90 (np.ndarray): Time delayed data for 90 polarization.
        ml (np.ndarray): Array of indices for each sky location.
        REG (np.ndarray): Regularization parameters.
        netCC (float): Network correlation coefficient threshold.
        delta_regulator (float): Delta regulator value.
        network_energy_threshold (float): Energy threshold for the network.

    Returns:
        tuple: (l_max, nAntenaPrior, nAlignment, nLikelihood, nNullEnergy, nCorrEnergy, 
                nCorrelation, nSkyStat, nDisbalance, nNetIndex, nEllipticity, nPolarisation)
            - l_max (int): Index of the sky location with maximum likelihood.
            - nAntenaPrior (np.ndarray): Antenna prior values for each sky location.
            - nAlignment (np.ndarray): Alignment values for each sky location.
            - nLikelihood (np.ndarray): Likelihood values for each sky location.
            - nNullEnergy (np.ndarray): Null energy values for each sky location.
            - nCorrEnergy (np.ndarray): Correlation energy values for each sky location.
            - nCorrelation (np.ndarray): Correlation values for each sky location.
            - nSkyStat (np.ndarray): Sky statistics for each sky location.
            - nDisbalance (np.ndarray): Disbalance values for each sky location.
            - nNetIndex (np.ndarray): Network index values for each sky location.
            - nEllipticity (np.ndarray): Ellipticity values for each sky location.
            - nPolarisation (np.ndarray): Polarization values for each sky location.
    """
    # td00 = np.transpose(td00.astype(np.float32), (2, 0, 1))  # (ndelay, nifo, npix)
    # td90 = np.transpose(td90.astype(np.float32), (2, 0, 1))  # (ndelay, nifo, npix)
    # FP = FP.T.astype(np.float32)
    # FX = FX.T.astype(np.float32)
    # rms = rms.T.astype(np.float32)
    REG = REG.astype(np.float32)


    # get unique list of ml.T for each nsky for numba
    # ml_set = set()
    # for i in range(n_sky):
    #     ml_set.add(tuple(ml.T[i]))

    nAlignment = np.zeros(n_sky, dtype=float32)
    nLikelihood = np.zeros(n_sky, dtype=float32)
    nNullEnergy = np.zeros(n_sky, dtype=float32)
    nCorrEnergy = np.zeros(n_sky, dtype=float32)
    nCorrelation = np.zeros(n_sky, dtype=float32)
    nSkyStat = np.zeros(n_sky, dtype=float32)
    nProbability = np.zeros(n_sky, dtype=float32)
    nDisbalance = np.zeros(n_sky, dtype=float32)
    nNetIndex = np.zeros(n_sky, dtype=float32)
    nEllipticity = np.zeros(n_sky, dtype=float32)
    nPolarisation = np.zeros(n_sky, dtype=float32)
    nAntenaPrior = np.zeros(n_sky, dtype=float32)

    Eh = float32(0.0)
    # sky = 0.0
    # l_max = 0
    # STAT=-1.e12
    offset = int(td00.shape[0] / 2)
    # TODO: sky sky mask
    AA_array = np.zeros(n_sky, dtype=float32)
    for l in prange(n_sky):
        # get time delayed data slice at sky location l, make sure it is numpy float32 array
        v00 = np.empty((n_ifo, n_pix), dtype=float32)
        v90 = np.empty((n_ifo, n_pix), dtype=float32)
        for i in range(n_ifo):
            v00[i] = td00[ml[i, l] + offset, i]
            v90[i] = td90[ml[i, l] + offset, i]

        # calculate data stats for time delayed data slice
        Eo, NN, energy_total, mask = load_data_from_td(v00, v90, network_energy_threshold)
        # print Eo at 0, 1000, 2000
        # if l == 0 or l == 1000 or l == 2000:
        #     print(f"Eo({l}): ", Eo)
        #     print(f"mask[{l}]: ", np.sum(mask))
        # print(f"v00[{l}]:" , np.max(v00[0]), np.max(v00[1]), f", v90[{l}]:", np.max(v90[0]), np.max(v90[1]))
        # print(f"ml[0, {l}]: {ml[0, l]}, ml[1, {l}]: {ml[1, l]}")

        # calculate DPF f+,fx and their norms
        _, f, F, fp, fx, si, co, ni = dpf_np_loops_vec(FP[l], FX[l], rms)

        # gw strain packet, return number of selected pixels
        Mo, ps, pS, mask, au, AU, av, AV = avx_GW_ps(v00, v90, f, F, fp, fx, ni, energy_total, mask, REG)
        # print Mo at 0, 1000, 2000
        # if l == 0 or l == 1000 or l == 2000:
        #     print(f"Mo({l}): ", Mo)
        #     print(f"ps[{l}]: ", np.max(ps[0]), np.max(ps[1])), print(f"pS[{l}]: ", np.max(pS[0]), np.max(pS[1]))
        #     print(f"mask[{l}]: ", np.sum(mask))
        #     print(f"fp[{l}]: ", np.max(fp), f" fx[{l}]: ", np.max(fx))


        # othogonalize signal amplitudes
        Lo, si, co, ee, EE = avx_ort_ps(ps, pS, mask)
        # print Lo at 0, 1000, 2000
        # if l == 0 or l == 1000 or l == 2000:
        #     print(f"Lo({l}): ", Lo)

        # coherent statistics
        Cr, Ec, Mp, No, coherent_energy, _, _ = avx_stat_ps(v00, v90, ps, pS, si, co, mask)

        CH = No / (n_ifo * Mo + sqrt(Mo))  # chi2 in TF domain
        cc = CH if CH > float(1.0) else 1.0  # noise correction factor in TF domain
        Co = Ec / (Ec + No * cc - Mo * (n_ifo - 1))  # network correlation coefficient in TF

        if Cr < netCC:
            continue

        aa = Eo - No if Eo > float32(0.) else float32(0.)  # likelihood skystat
        AA = aa * Co  # x-correlation skystat
        # if l == 0 or l == 1000 or l == 2000:
        #     print(f"Cr({l}): ", Cr, f" Ec[{l}]: ", Ec, f" Mp[{l}]: ", Mp, f" No[{l}]: ", No)
        #     print(f"CH({l}): ", CH, f" cc[{l}]: ", cc, f" Co[{l}]: ", Co)
        #     print(f"aa({l}): ", aa, f" AA[{l}]: ", AA)
        nProbability[l] = aa if delta_regulator < 0 else AA

        ff, FF, ee = float32(0.), float32(0.), float32(0.)

        for j in range(n_pix):
            if mask[j] <= 0:
                continue
            ee += energy_total[j]  # total energy
            ff += fp[j] * energy_total[j]  # |f+|^2
            FF += fx[j] * energy_total[j]  # |fx|^2
        ff = ff / ee if ee > float32(0.) else float32(0.)
        FF = FF / ee if ee > float32(0.) else float32(0.)

        nAntenaPrior[l] = sqrt(ff + FF)
        nAlignment[l] = sqrt(FF / ff) if ff > float32(0.) else float32(0.)
        nLikelihood[l] = Eo - No
        nNullEnergy[l] = No
        nCorrEnergy[l] = Ec
        nCorrelation[l] = Co
        nSkyStat[l] = AA
        nDisbalance[l] = CH
        nNetIndex[l] = cc
        nEllipticity[l] = Cr
        nPolarisation[l] = Mp

        AA_array[l] = AA
        # if AA >= STAT:
        #     STAT = AA
        #     l_max = l
        #     Em = Eo - Eh
        #
        # if nProbability[l] > sky:
        #     sky = nProbability[l]  # find max of skyloc stat
    STAT = np.max(AA_array)
    l_max = np.argmax(AA_array)
    sky = np.max(nProbability)

    return (l_max, nAntenaPrior, nAlignment, nLikelihood, nNullEnergy, nCorrEnergy, \
              nCorrelation, nSkyStat, nDisbalance, nNetIndex, nEllipticity, nPolarisation)
    # return {
    #     'l_max': l_max,
    #     'nAntennaPrior': nAntenaPrior,
    #     'nAlignment': nAlignment,
    #     'nLikelihood': nLikelihood,
    #     'nNullEnergy': nNullEnergy,
    #     'nCorrEnergy': nCorrEnergy,
    #     'nCorrelation': nCorrelation,
    #     'nSkyStat': nSkyStat,
    #     'nDisbalance': nDisbalance,
    #     'nNetIndex': nNetIndex,
    #     'nEllipticity': nEllipticity,
    #     'nPolarisation': nPolarisation,
    # }


# @njit(cache=True)
def calculate_sky_statistics(l, n_ifo, n_pix, FP, FX, rms, td00, td90, ml, REG, 
                             network_energy_threshold, cluster_xtalk, cluster_xtalk_lookup_table,
                             DEBUG=False) -> SkyStatistics:
    """
    Calculate the sky statistics for a specific sky location l.
    Args:
        l (int): Index of the sky location.
        n_ifo (int): Number of interferometers.
        n_pix (int): Number of pixels.
        FP (np.ndarray): f+ polarization data for each interferometer.
        FX (np.ndarray): fx polarization data for each interferometer.
        rms (np.ndarray): RMS values for each interferometer and pixel.
        td00 (np.ndarray): Time delayed data for 00 polarization.
        td90 (np.ndarray): Time delayed data for 90 polarization.
        ml (np.ndarray): Array of indices for each sky location.
        REG (np.ndarray): Regularization parameters.
        network_energy_threshold (float): Energy threshold for the network.
        cluster_xtalk (XTalk): Cluster XTalk object containing xtalk information.
        cluster_xtalk_lookup_table: Lookup table for xtalk.

    Returns:
        SkyStatistics: Dataclass containing the sky statistics for the specified sky location.
    """
    # from numpy import float32
    # td00 = np.transpose(td00.astype(np.float32), (2, 0, 1))  # (ndelay, nifo, npix)
    # td90 = np.transpose(td90.astype(np.float32), (2, 0, 1))  # (ndelay, nifo, npix)
    v00 = np.empty((n_ifo, n_pix), dtype=np.float32)
    v90 = np.empty((n_ifo, n_pix), dtype=np.float32)
    td_energy = np.zeros((n_ifo, n_pix), dtype=np.float32)

    offset = int(td00.shape[0] / 2)

    # get time delayed data slice at sky location l
    for i in range(n_ifo):
        v00[i] = td00[ml[i, l] + offset, i]
        v90[i] = td90[ml[i, l] + offset, i]

    # compute the energy of the time delayed data slice
    for i in range(n_ifo):
        for j in range(n_pix):
            td_energy[i, j] = v00[i, j] * v00[i, j] + v90[i, j] * v90[i, j]

    # calculate the total energy, active pixels and mask
    Eo, NN, energy_total, mask = load_data_from_td(v00, v90, network_energy_threshold)

    # calculate DPF f+,fx and their norms for the sky location l
    _, f, F, fp, fx, si, co, ni = dpf_np_loops_vec(FP[l], FX[l], rms)

    # gw strain packet, return number of selected pixels
    Mo, ps, pS, mask, au, AU, av, AV = avx_GW_ps(v00, v90, f, F, fp, fx, ni, energy_total, mask, REG)

    # othogonalize signal amplitudes
    Lo, si, co, ee, EE = avx_ort_ps(ps, pS, mask)

    # coherent statistics
    _, _, _, _, coherent_energy, gn, rn = avx_stat_ps(v00, v90, ps, pS, si, co, mask)

    ##############################
    # Eo = _avx_packet_ps(pd, pD, _AVX, V4);            // get data packet
    # Lo = _avx_packet_ps(ps, pS, _AVX, V4);            // get signal packet
    # D_snr = _avx_norm_ps(wdmMRA, pd, pD, _AVX, V4);           // data packet energy snr
    # S_snr = _avx_norm_ps(pS, pD, p_ec, V4);           // set signal norms, return signal SNR
    # Ep = D_snr[0];
    # Lp = S_snr[0];
    ##############################
    Eo, pd, pD, pD_E, pD_si, pD_co, pD_a, pD_A = avx_packet_ps(v00, v90, mask)  # get data packet
    Lo, ps, pS, pS_E, pS_si, pS_co, pS_a, pS_A = avx_packet_ps(ps, pS, mask)  # get signal packet

    detector_snr, pD_E, rn, pD_norm = packet_norm_numpy(pd, pD, cluster_xtalk, cluster_xtalk_lookup_table, mask, pD_E)
    D_snr = np.sum(detector_snr)
    S_snr, signal_snr, pS_E, pS_norm = gw_norm_numpy(pD_norm, pD_E, pS_E, coherent_energy) # return signal norms and signal SNR
    # S_snr = np.sum(signal_norm)
    if DEBUG:
        print(S_snr, signal_snr)
        print("Eo = ", Eo, ", Lo = ", Lo, ", Ep = ", D_snr, ", Lp = ", S_snr)

    ############### G-noise correction ##################
    # _CC = _avx_noise_ps(pS, pD, _AVX, V4);            // get G-noise correction
    # _mm256_storeu_ps(vvv, _CC);                     // extract coherent statistics
    # Gn = vvv[0];                                   // gaussian noise correction
    # Ec = vvv[1];                                   // core coherent energy in TF domain
    # Dc = vvv[2];                                   // signal-core coherent energy in TF domain
    # Rc = vvv[3];                                   // EC normalization
    # Eh = vvv[4];                                   // satellite energy in TF domain
    #################################

    # TODO: one more pixel selected, need to be fixed
    # Gn: gaussian noise correction
    # Ec: core coherent energy in TF domain
    # Dc: signal-core coherent energy in TF domain
    # Rc: EC normalization
    # Eh: satellite energy in TF domain    
    Gn, Ec, Dc, Rc, Eh, Es, NC, NS = avx_noise_ps(pS_norm, pD_norm, energy_total, mask, coherent_energy, gn, rn)

    if DEBUG:
        print("Gn = ", Gn, ", Ec = ", Ec, ", Dc = ", Dc, ", Rc = ", Rc, ", Eh = ", Eh, ", Es = ", Es, ", NC = ", NC, ", NS = ", NS)
    # # DEBUG:
    # print(f"pD[0][393]: {pD[0][393]}, pS[0][393]: {pS[0][393]}")
    # print(f"pD[1][393]: {pD[1][393]}, pS[1][393]: {pS[1][393]}")

    ##################################
    # N = _avx_setAMP_ps(pd, pD, _AVX, V4) - 1;           // set data packet amplitudes
    # _avx_setAMP_ps(ps, pS, _AVX, V4);                 // set signal packet amplitudes
    # _avx_loadNULL_ps(pn, pN, pd, pD, ps, pS, V4);        // load noise TF domain amplitudes
    # D_snr = _avx_norm_ps(wdmMRA, pd, pD, _AVX, -V4);          // data packet energy snr
    # N_snr = _avx_norm_ps(wdmMRA, pn, pN, _AVX, -V4);          // noise packet energy snr
    # Np = N_snr.data[0];                            // time-domain NULL
    # Em = D_snr.data[0];                            // time domain energy
    # Lm = Em - Np - Gn;                                 // time domain signal energy
    # norm = Em > 0 ? (Eo - Eh) / Em : 1.e9;               // norm
    # if (norm < 1) norm = 1;                           // corrected norm
    # Ec /= norm;                                   // core coherent energy in time domain
    # Dc /= norm;                                   // signal-core coherent energy in time domain
    # ch = (Np + Gn) / (N * nIFO);                         // chi2
    ##################################

    # set data packet amplitudes
    N, pd, pD = avx_setAMP_ps(pd, pD, pD_norm, pD_si, pD_co, pD_a, pD_A, mask)  # set data packet amplitudes
    N = N - 1  # effective number of pixels
    # set signal packet amplitudes
    _, ps, pS = avx_setAMP_ps(ps, pS, pS_norm, pS_si, pS_co, pS_a, pS_A, mask)
    # load noise TF domain amplitudes
    pn, pN = avx_loadNULL_ps(pd, pD, ps, pS)
    # data packet energy snr
    _, pD_E, rn, _ = packet_norm_numpy(pd, pD, cluster_xtalk, cluster_xtalk_lookup_table, mask, pD_E)
    D_snr = np.sum(pD_E)  # data packet energy snr
    _, pN_E, rn, _ = packet_norm_numpy(pn, pN, cluster_xtalk, cluster_xtalk_lookup_table, mask, pS_E)
    N_snr = np.sum(pN_E)  # noise packet energy snr

    Np = N_snr  # time-domain NULL
    Em = D_snr  # time domain energy
    Lm = Em - Np - Gn  # time domain signal energy
    norm = (Eo - Eh) / Em if (Eo - Eh) > 0 else 1.e9  # norm
    if norm < 1:
        norm = 1
    Ec /= norm  # core coherent energy in time domain
    Dc /= norm  # signal-core coherent energy in time domain
    ch = (Np + Gn) / (N * n_ifo)  # chi
    if DEBUG:
        print("Np = ", Np, ", Em = ", Em, ", Lm = ", Lm, ", norm = ", norm, ", Ec = ", Ec, ", Dc = ", Dc, ", ch = ", ch)

    #################################
    # if (netRHO >= 0) {    // original 2G
    #     cc = ch > 1 ? ch : 1;                          // rho correction factor
    #     rho = Ec > 0 ? sqrt(Ec * Rc / 2.) : 0.;           // cWB detection stat
    # } else {        // (XGB.rho0)
    #     penalty = ch;
    #     ecor = Ec;
    #     rho = sqrt(ecor / (1 + penalty * (max((float) 1., penalty) - 1)));
    #     // original 2G rho statistic: only for test
    #     cc = ch > 1 ? ch : 1;                          // rho correction factor
    #     xrho = Ec > 0 ? sqrt(Ec * Rc / 2.) : 0.;          // cWB detection stat
    # }
    #################################
    xrho = 0.
    penalty = 0.
    ecor = 0.
    if network_energy_threshold >= 0: # original 2G
        cc = ch if ch > 1 else 1  # rho correction factor
        rho = np.sqrt(Ec * Rc / 2.) if Ec > 0 else 0  # cWB detection stat
        if DEBUG:
            print("cc = ", cc, ", rho = ", rho)
    else:  # (XGB.rho0)
        penalty = ch
        ecor = Ec
        rho = np.sqrt(ecor / (1 + penalty * (max(float(1), penalty) - 1)))
        # original 2G rho statistic: only for test
        cc = ch if ch > 1 else 1  # rho correction factor
        xrho = np.sqrt(Ec * Rc / 2.) if Ec > 0 else 0  # cWB detection stat
        if DEBUG:
            print("cc = ", cc, ", rho = ", rho, ", ecor = ", ecor, ", penalty = ", penalty, ", xrho = ", xrho)
    #################################
    # // save projection on network plane in polar coordinates
    # // The Dual Stream Transform (DSP) is applied to v00,v90
    # _avx_pol_ps(v00, v90, p00_POL, p90_POL, _APN, _AVX, V4);
    # // save DSP components in polar coordinates
    # _avx_pol_ps(v00, v90, r00_POL, r90_POL, _APN, _AVX, V4);
    #################################
    # save projection on network plane in polar coordinates
    v00, v90, p00_POL, p90_POL = avx_pol_ps(v00, v90, mask, fp, fx, f, F)
    # save DSP components in polar coordinates
    v00, v90, r00_POL, r90_POL = avx_pol_ps(v00, v90, mask, fp, fx, f, F)

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
        v00=v00,
        v90=v90,
        gaussian_noise_correction=gn,
        coherent_energy=coherent_energy,
        N_pix_effective=N,
        noise_amplitude_00=pn,
        noise_amplitude_90=pN,
        pd=pd,
        pD=pD,
        ps=ps,
        pS=pS,
        p00_POL=p00_POL,
        p90_POL=p90_POL,
        r00_POL=r00_POL,
        r90_POL=r90_POL,
        S_snr=signal_snr,
        f = f,
        F = F,
    )


def fill_detection_statistic(sky_statistics: SkyStatistics, skymap_statistics: SkyMapStatistics, 
                             cluster: Cluster, n_ifo: int, 
                             xtalk: XTalk,
                             network_energy_threshold: float):
    """
    Fill the detection statistics into the cluster and pixels.
    
    Args:
        sky_statistics (SkyStatistics): The sky statistics object containing the calculated statistics.
        skymap_statistics (SkyMapStatistics): The skymap statistics object to be filled.
        cluster (Cluster): The cluster object containing the pixels.
        n_ifo (int): Number of interferometers.
        xtalk (XTalk): The XTalk object for cross-talk calculations.
        network_energy_threshold (float): Energy threshold for the network.
    
    Returns:
        None: The function modifies the cluster and skymap_statistics in place.
    """
    pixel_mask = sky_statistics.pixel_mask
    energy_array_plus = sky_statistics.energy_array_plus
    energy_array_cross = sky_statistics.energy_array_cross
    pd = sky_statistics.pd
    pD = sky_statistics.pD
    ps = sky_statistics.ps
    pS = sky_statistics.pS
    gaussian_noise_correction = sky_statistics.gaussian_noise_correction
    pn = sky_statistics.noise_amplitude_00
    pN = sky_statistics.noise_amplitude_90
    coherent_energy = sky_statistics.coherent_energy
    S_snr = sky_statistics.S_snr
    Rc = sky_statistics.Rc
    Gn = sky_statistics.Gn
    Np = sky_statistics.Np
    N_pix_effective = sky_statistics.N_pix_effective

    event_size = 0 # defined as Mw in cwb
    n_coherent_pixels = 0

    # --- First pass: set core/likelihood/null flags and per-ifo data arrays ---
    n_pix = len(cluster.pixels)
    for i, pixel in enumerate(cluster.pixels):
        pixel.core = False
        pixel.likelihood = 0.0
        pixel.null = 0.0

        if pixel_mask[i] > 0:
            pixel.core = True
            pixel.likelihood = - (energy_array_plus[i] + energy_array_cross[i]) / 2.0

        for j in range(n_ifo):
            pixel.data[j].wave = pd[j][i]
            pixel.data[j].w_90 = pD[j][i]
            pixel.data[j].asnr = ps[j][i]
            pixel.data[j].a_90 = pS[j][i]

    # Pre-convert amplitude arrays to 2-D NumPy for fast column access
    pn_arr = np.asarray(pn, dtype=np.float64)  # (n_ifo, n_pix)
    pN_arr = np.asarray(pN, dtype=np.float64)
    ps_arr = np.asarray(ps, dtype=np.float64)
    pS_arr = np.asarray(pS, dtype=np.float64)

    # Precompute per-pixel xtalk once using the vectorised lookup
    xtalks_lookup, xtalks = xtalk.get_xtalk_pixels(cluster.pixels)

    # Prefilter outer-loop eligible pixels (avoids the inner xpix.core test each time)
    # null inner condition : core AND gnc > 0  (mirrors C++: skip if !core || p_gn[k]<=0)
    null_k_set = np.array(
        [k for k, xpix in enumerate(cluster.pixels)
         if xpix.core and gaussian_noise_correction[k] > 0],
        dtype=np.int64
    )
    # likelihood inner condition : core AND coherent_energy > 0  (mirrors C++: skip if !core || p_ec[k]<=0)
    like_k_set = np.array(
        [k for k, xpix in enumerate(cluster.pixels)
         if xpix.core and coherent_energy[k] > 0],
        dtype=np.int64
    )

    # --- Second pass: compute null and likelihood using vectorised inner sums ---
    logger.debug("fill_detection_statistic: null_k_set size=%d, like_k_set size=%d, n_pix=%d",
                 len(null_k_set), len(like_k_set), n_pix)
    logger.debug("fill_detection_statistic: pn_arr shape=%s, pn range=[%g, %g]",
                 str(pn_arr.shape), float(np.min(np.abs(pn_arr))), float(np.max(np.abs(pn_arr))))
    logger.debug("fill_detection_statistic: gn range=[%g, %g], ec range=[%g, %g]",
                 float(np.min(gaussian_noise_correction)),
                 float(np.max(gaussian_noise_correction)),
                 float(np.min(coherent_energy)),
                 float(np.max(coherent_energy)))

    for i, pixel in enumerate(cluster.pixels):
        if not pixel.core or gaussian_noise_correction[i] <= 0:
            continue

        event_size += 1

        # null computation — inner loop over (core, gnc > 0) pixels
        if len(null_k_set) > 0:
            null_acc = 0.0
            pn_i = pn_arr[:, i]   # (n_ifo,)
            pN_i = pN_arr[:, i]
            for k in null_k_set:
                xt = xtalk.get_xtalk(pix1=pixel, pix2=cluster.pixels[k])
                if xt[0] > 2:
                    continue
                # Vectorised over ifo dimension
                term = (xt[0] * np.dot(pn_i, pn_arr[:, k])
                        + xt[1] * np.dot(pn_i, pN_arr[:, k])
                        + xt[2] * np.dot(pN_i, pn_arr[:, k])
                        + xt[3] * np.dot(pN_i, pN_arr[:, k]))
                null_acc += term
            pixel.null = null_acc

        if coherent_energy[i] <= 0:
            continue    # skip the incoherent pixels

        n_coherent_pixels += 1
        pixel.likelihood = 0

        # likelihood computation — inner loop over (core, coherent_energy<=0) pixels
        if len(like_k_set) > 0:
            like_acc = 0.0
            ps_i = ps_arr[:, i]   # (n_ifo,)
            pS_i = pS_arr[:, i]
            for k in like_k_set:
                xt = xtalk.get_xtalk(pix1=pixel, pix2=cluster.pixels[k])
                if xt[0] > 2:
                    continue
                like_acc += (xt[0] * np.dot(ps_i, ps_arr[:, k])
                             + xt[1] * np.dot(ps_i, pS_arr[:, k])
                             + xt[2] * np.dot(pS_i, ps_arr[:, k])
                             + xt[3] * np.dot(pS_i, pS_arr[:, k]))
            pixel.likelihood = like_acc

    # subnetwork statistic
    Nmax = 0.0
    Emax = np.max(S_snr)

    Esub = np.sum(S_snr) - Emax
    Esub = Esub * (1 + 2 * Rc * Esub / Emax)
    Nmax = Gn + Np - N_pix_effective * (n_ifo - 1)

    # --- Accumulate time-domain waveform statistics (mirrors C++ getMRAwave('W') + ('S') loop)
    #
    # Lw  —  waveform likelihood  (C++: cData[id-1].likenet, stored in Event.likelihood)
    # ─────────────────────────────────────────────────────────────────────────────────────
    # C++ computes Lw by:
    #   1. getMRAwave('S') → reconstructs the whitened signal time series for each IFO:
    #          s_i(t) = Σ_{j∈core} [ asnr_ij · ψ00_j(t) + a_90_ij · ψ90_j(t) ]
    #      where ψ00_j / ψ90_j are the 0°/90° WDM basis functions for pixel j (netcluster.cc).
    #   2. get_SS() = Σ_t s_i(t)²  — squared norm of the reconstructed waveform.
    #   3. Lw = Σ_i  get_SS_i
    #
    # Expanding the squared norm reveals WDM cross-pixel inner products (xtalk):
    #
    #   Lw = Σ_i  Σ_{j,k ∈ core, CC0_jk ≤ 2}
    #               [ CC0_jk · asnr_ij · asnr_ik          (00–00 cross)
    #               + CC1_jk · asnr_ij · a_90_ik          (00–90 cross)
    #               + CC2_jk · a_90_ij · asnr_ik          (90–00 cross)
    #               + CC3_jk · a_90_ij · a_90_ik ]        (90–90 cross)
    #
    # where CC0–CC3 = WDM basis inner products from the xtalk catalog:
    #   CC0 = <ψ00_j, ψ00_k>,  CC1 = <ψ00_j, ψ90_k>,
    #   CC2 = <ψ90_j, ψ00_k>,  CC3 = <ψ90_j, ψ90_k>.
    # The cut CC0 > 2 skips non-overlapping pairs (sentinel value in the catalog).
    #
    # For a coherent GW signal spanning V core pixels, the O(V²) cross-terms can be as
    # large as the O(V) diagonal, so the diagonal-only approximation underestimates Lw
    # by a factor O(V) for detection-strength signals.
    #
    # See docs/math/waveform_likelihood.md for the full derivation.
    #
    # Ew = Σ_{i,j} (wave²_ij + w_90²_ij)  — time-domain data energy  (diagonal, no xtalk)
    # Nw = Σ_{i,j} (pn²_ij  + pN²_ij)   — time-domain null energy   (diagonal, no xtalk)
    # To = Σ_j w_j·t_j / Lw              — Lw-weighted mean time
    # Fo = Σ_j w_j·f_j / Lw              — Lw-weighted mean frequency
    #   where w_j = Σ_i (asnr²_ij + a_90²_ij)  (per-pixel diagonal sSNR used as centroid weight)
    ps_arr_np = np.asarray(ps, dtype=np.float64)   # (n_ifo, n_pix)
    pS_arr_np = np.asarray(pS, dtype=np.float64)
    pn_arr_np = np.asarray(pn, dtype=np.float64)
    pN_arr_np = np.asarray(pN, dtype=np.float64)

    # Core pixel indices — only core pixels contribute to getMRAwave (ps/pS are 0 for non-core
    # after avx_setAMP_ps, but using explicit core mask matches the C++ getMRAwave logic)
    core_indices = [k for k, pix in enumerate(cluster.pixels) if pix.core]

    # Lw: xtalk-corrected double sum over core pixels (mirrors C++ get_SS on reconstructed waveForm)
    Lw = 0.0
    for i_idx in core_indices:
        for k_idx in core_indices:
            xt = xtalk.get_xtalk(pix1=cluster.pixels[i_idx], pix2=cluster.pixels[k_idx])
            if xt[0] > 2:
                continue
            ps_i = ps_arr_np[:, i_idx]   # (n_ifo,)
            pS_i = pS_arr_np[:, i_idx]
            ps_k = ps_arr_np[:, k_idx]
            pS_k = pS_arr_np[:, k_idx]
            Lw += (xt[0] * float(np.dot(ps_i, ps_k))
                 + xt[1] * float(np.dot(ps_i, pS_k))
                 + xt[2] * float(np.dot(pS_i, ps_k))
                 + xt[3] * float(np.dot(pS_i, pS_k)))

    # Ew, Nw: diagonal sums (no xtalk — kept for time/freq centroid bookkeeping only).
    # meta.energy uses sky_statistics.Em (xtalk-corrected proxy for C++ getMRAwave Ew).
    # meta.net_null / meta.net_ed use sky_statistics.Np (xtalk-corrected proxy for C++ getMRAwave Nw),
    # exactly analogous to using Em for energy.
    Ew = 0.0
    Nw = 0.0
    To = 0.0
    Fo = 0.0
    for i, pixel in enumerate(cluster.pixels):
        # enrg = sum_j (pd[j][i]^2 + pD[j][i]^2)
        enrg_pix = sum(
            pixel.data[j].wave ** 2 + pixel.data[j].w_90 ** 2
            for j in range(n_ifo)
        )
        # null = sum_j (pn[j][i]^2 + pN[j][i]^2)
        null_pix = float(np.sum(pn_arr_np[:, i] ** 2 + pN_arr_np[:, i] ** 2))
        Ew += enrg_pix
        Nw += null_pix
        # time/freq weights: pixel.time is sample index; physical time = idx / (rate * layers)
        # physical freq: freq_bin in [0, layers/2]; nyquist = rate/2 (since rate = rateANA/layers)
        # actual_hz = freq_bin * nyquist / (layers/2) = freq_bin * rate / 2
        pix_time = float(pixel.time) / (float(pixel.rate) * float(pixel.layers)) if (pixel.rate > 0 and pixel.layers > 0) else 0.0
        pix_freq = float(pixel.frequency) * float(pixel.rate) / 2.0 if pixel.rate > 0 else 0.0
        # Centroid numerator: use diagonal sSNR per pixel as weight (xtalk correction is global)
        s_snr_pix = float(np.sum(ps_arr_np[:, i] ** 2 + pS_arr_np[:, i] ** 2))
        To += s_snr_pix * pix_time
        Fo += s_snr_pix * pix_freq

    if Lw > 0.0:
        To /= Lw
        Fo /= Lw

    # -----------------------------------------------------------------------
    # Detection statistics: netCC, norm
    # -----------------------------------------------------------------------
    # Notation (mirrors network.cc likelihoodWP):
    #   Eo  — total TF-domain data energy (all resolutions, at best sky position)
    #   Eh  — satellite (halo) energy in TF domain; from avx_noise_ps vvv[4]
    #   Em  — xtalk-corrected time-domain packet energy; from _avx_norm_ps(pd,pD,-V4)
    #           In Python: sky_statistics.Em = np.sum(pD_E) after packet_norm_numpy with
    #           xtalk corrections — this is the best proxy for C++ getMRAwave-based Ew.
    #   Nw  — time-domain null energy accumulated over core pixels (see loop above)
    #   Gn  — Gaussian noise correction
    #   Ec  — core coherent energy (TF domain, normalised by norm)
    #   Dc  — signal-core coherent energy (TF domain)
    #   Rc  — EC normalisation factor (Ec*Rc = normalised coherent energy)
    #   N   — effective number of pixels
    #
    # C++ formulas:
    #   ch     = (Nw + Gn) / (N * nIFO)
    #   cc_corr= ch > 1 ? ch : 1
    #   Cp     = Ec*Rc / (Ec*Rc + (Dc+Nw+Gn)         - N*(nIFO-1))   # netCC  (netcc[0])
    #   Cr     = Ec*Rc / (Ec*Rc + (Dc+Nw+Gn)*cc_corr - N*(nIFO-1))   # skyCC  (netcc[1])
    #   norm   = (Eo - Eh) / Em          # raw packet norm in TF domain
    #   norm   = max(norm, 1)            # clamped from below
    #   stored = norm * 2                # factor of 2 accounts for two quadratures (00+90)
    #
    # Why Em and not the raw TF diagonal sum Ew = Σ(wave²+w_90²)?
    #   Ew is a diagonal-only approximation that ignores WDM xtalk cross-terms between
    #   pixels.  For multi-pixel clusters those cross-terms are non-zero, so Ew
    #   underestimates the true packet energy and inflates norm.  Em (from
    #   packet_norm_numpy which includes the full xtalk double-sum) matches the C++
    #   _avx_norm_ps energy and gives the correct denominator.
    # -----------------------------------------------------------------------
    Dc = float(sky_statistics.Dc)
    Ec = float(sky_statistics.Ec)
    Rc_val = float(sky_statistics.Rc)
    Eo = float(sky_statistics.Eo)
    Eh = float(sky_statistics.Eh)
    Gn_val = float(sky_statistics.Gn)
    N_eff = float(N_pix_effective)
    ch_td = (Nw + Gn_val) / (N_eff * n_ifo) if (N_eff * n_ifo) > 0 else 1.0
    cc_corr = ch_td if ch_td > 1.0 else 1.0
    denom_r = Ec * Rc_val + (Dc + Nw + Gn_val) * cc_corr - N_eff * (n_ifo - 1)
    denom_p = Ec * Rc_val + (Dc + Nw + Gn_val) - N_eff * (n_ifo - 1)
    Cr_td = (Ec * Rc_val / denom_r) if denom_r > 0 else 0.0
    Cp_td = (Ec * Rc_val / denom_p) if denom_p > 0 else 0.0
    # Use Em (xtalk-corrected energy from packet_norm_numpy) instead of Ew (raw TF diagonal sum).
    # Em matches the C++ _avx_norm_ps(pd,pD,-V4) energy used to compute norm in the sky loop,
    # which is the best Python equivalent of the C++ getMRAwave-based Ew.
    Em_td = float(sky_statistics.Em)
    norm_td = (Eo - Eh) / Em_td if Em_td > 0 else 1.0
    if norm_td < 1.0:
        norm_td = 1.0

    # C++ rho uses /sqrt(cc) in original 2G: netRHO = rho/sqrt(cc)
    cc_rho = float(sky_statistics.cc)  # rho correction factor from sky loop
    rho_reduced = float(sky_statistics.rho) / sqrt(max(cc_rho, 1.0))

    # --- Store all fields on cluster_meta ---
    cluster.cluster_meta.sky_size = event_size             # event size in the skyloop
    cluster.cluster_meta.sub_net = Esub / (Esub + Nmax) if (Esub + Nmax) > 0 else 0.0  # sub-network statistic
    cluster.cluster_meta.sub_net2 = skymap_statistics.nCorrelation[skymap_statistics.l_max]    # sky cc
    # neted[3]: likesky = Em = Eo_loadata - Eh at best sky (C++ multi-sky sky-loop energy).
    # sky_statistics.Eo is the packet Eo (best available proxy for loadata Eo in Python).
    cluster.cluster_meta.like_sky = float(sky_statistics.Eo - sky_statistics.Eh)  # Eo-Eh at best sky (neted[3])
    cluster.cluster_meta.energy_sky = sky_statistics.Eo    # TF-domain all-res energy (neted[4])
    cluster.cluster_meta.net_ecor = sky_statistics.Ec      # packet (signal) coherent energy
    cluster.cluster_meta.norm_cor = sky_statistics.Ec * sky_statistics.Rc   # normalized coherent energy
    cluster.cluster_meta.like_net = float(Lw)              # waveform likelihood (likenet)
    # neted[2]: energy = Ew from getMRAwave in C++ = xtalk-corrected time-domain data energy.
    # sky_statistics.Em (D_snr from packet_norm_numpy) is the best Python proxy for getMRAwave Ew.
    # Raw pixel diagonal Ew underestimates due to missing xtalk cross-terms.
    cluster.cluster_meta.energy = float(sky_statistics.Em)  # xtalk-corrected time-domain energy (neted[2])
    # neted[1] and neted[0]: Nw in C++ is getMRAwave-based xtalk-corrected time-domain null.
    # sky_statistics.Np (N_snr from packet_norm_numpy(pn,pN,xtalk)) is the best Python proxy,
    # exactly analogous to sky_statistics.Em for energy. Raw diagonal Nw would underestimate.
    Nw_xtalk = float(sky_statistics.Np)  # xtalk-corrected time-domain null (proxy for getMRAwave Nw)
    cluster.cluster_meta.net_null = float(Nw_xtalk + Gn_val)    # packet NULL (neted[1])
    cluster.cluster_meta.net_ed = float(Nw_xtalk + Gn_val + Dc - N_eff * n_ifo)  # residual NULL (neted[0])
    cluster.cluster_meta.norm = float(norm_td * 2.0)       # packet norm (norm)
    cluster.cluster_meta.net_cc = float(Cp_td)             # network cc (netcc[0])
    cluster.cluster_meta.sky_cc = float(Cr_td)             # reduced network cc (netcc[1])
    # c_time / c_freq from sSNR-weighted centroid over core pixels
    if Lw > 0.0:
        cluster.cluster_meta.c_time = float(To)
        cluster.cluster_meta.c_freq = float(Fo)

    if network_energy_threshold >= 0:  # original 2G
        cluster.cluster_meta.net_rho = rho_reduced                    # reduced rho (netRHO = rho/sqrt(cc))
        cluster.cluster_meta.net_rho2 = float(sky_statistics.rho)     # raw rho before cc reduction (netrho)
    else:  # (XGB.rho0)
        pass

    cluster.cluster_meta.g_net = skymap_statistics.nAntennaPrior[skymap_statistics.l_max]  # antenna prior
    cluster.cluster_meta.a_net = skymap_statistics.nAlignment[skymap_statistics.l_max]  # alignment
    cluster.cluster_meta.i_net = 0   # degrees of freedom
    cluster.cluster_meta.ndof = N_pix_effective  # degrees of freedom
    cluster.cluster_meta.sky_chi2 = skymap_statistics.nDisbalance[skymap_statistics.l_max]  # disbalance
    cluster.cluster_meta.g_noise = sky_statistics.Gn  # gaussian noise correction
    cluster.cluster_meta.iota = 0.0 
    cluster.cluster_meta.psi = 0.0
    cluster.cluster_meta.ellipticity = 0

    logger.debug(
        "fill_detection_statistic: sky_size=%d sub_net=%.4f net_cc=%.4f sky_cc=%.4f "
        "like_net=%.2f energy=%.2f net_null=%.4f norm=%.4f rho=%.4f",
        cluster.cluster_meta.sky_size, cluster.cluster_meta.sub_net,
        cluster.cluster_meta.net_cc, cluster.cluster_meta.sky_cc,
        cluster.cluster_meta.like_net, cluster.cluster_meta.energy,
        cluster.cluster_meta.net_null, cluster.cluster_meta.norm,
        cluster.cluster_meta.net_rho,
    )



def threshold_cut(sky_statistics: SkyStatistics, network_energy_threshold: float, netEC_threshold: float) -> str:
    """
    Apply threshold cuts based on the sky statistics and network energy threshold.
    
    Parameters:
        sky_statistics (SkyStatistics): The statistics calculated for the sky location.
        network_energy_threshold (float): The threshold for network energy.
        netEC_threshold (float): The threshold for net EC.
    
    Returns:
        str: A rejection reason if any condition is not met, otherwise None.
    """
    # if (this->netRHO >= 0)
    #   { // original 2G
    #     if (Lm <= 0. || (Eo - Eh) <= 0. || Ec * Rc / cc < netEC || N < 1)
    #     {
    #       pwc->sCuts[id - 1] = 1;
    #       count = 0; // reject cluster
    #       pwc->clean(id);
    #       continue;
    #     }
    #   }
    #   else
    #   { // (XGB.rho0)
    #     if (Lm <= 0. || (Eo - Eh) <= 0. || rho < fabs(this->netRHO) || N < 1)
    #     {
    #       pwc->sCuts[id - 1] = 1;
    #       count = 0; // reject cluster
    #       pwc->clean(id);
    #       continue;
    #     }
    #   }
    Lm = sky_statistics.Lm
    Eo = sky_statistics.Eo
    Np = sky_statistics.Np
    Eh = sky_statistics.Eh
    Ec = sky_statistics.Ec
    Rc = sky_statistics.Rc
    cc = sky_statistics.cc
    rho = sky_statistics.rho
    if network_energy_threshold > 0:
        condition_1 = Lm <= 0.
        condition_2 = (Eo - Eh) <= 0.
        condition_3 = Ec * Rc / cc < netEC_threshold
        condition_4 = Np < 1
        if condition_1 or condition_2 or condition_3 or condition_4:
            rejection_reason = ""
            if condition_1:
                rejection_reason += f"Lm > 0 but Lm = {Lm};"
            if condition_2:
                rejection_reason += f" (Eo - Eh) > 0 but (Eo - Eh) = {Eo - Eh};"
            if condition_3:
                rejection_reason += f" Ec * Rc / cc >= netEC_threshold but Ec * Rc / cc = {Ec * Rc / cc};"
            if condition_4:
                rejection_reason += f" Np > 1 but Np = {Np};"
            return rejection_reason
    else:
        # For XGB.rho0 case
        condition_1 = Lm <= 0.
        condition_2 = (Eo - Eh) <= 0.
        condition_3 = rho < abs(network_energy_threshold)
        condition_4 = Np < 1
        if condition_1 or condition_2 or condition_3 or condition_4:
            rejection_reason = ""
            if condition_1:
                rejection_reason += f"Lm > 0 but Lm = {Lm};"
            if condition_2:
                rejection_reason += f" (Eo - Eh) > 0 but (Eo - Eh) = {Eo - Eh};"
            if condition_3:
                rejection_reason += f" rho >= abs(network_energy_threshold) but rho = {rho} < {abs(network_energy_threshold)};"
            if condition_4:
                rejection_reason += f" Np > 1 but Np = {Np};"
            return rejection_reason
        
    return None  # No rejection, all conditions passed


def likelihood_by_pixel():
    pass


def subnetwork_statistic():
    pass


def detection_statistic():
    pass


def get_error_region(cluster: Cluster):
    # pwc->p_Ind[id - 1].push_back(Mo);
    # double T = To + pwc->start;                          // trigger time
    # std::vector<float> sArea;
    # pwc->sArea.push_back(sArea);
    # pwc->p_Map.push_back(sArea);
    #
    # double var = norm * Rc * sqrt(Mo) * (1 + fabs(1 - CH));
    #
    # // TODO: fix this
    # if (iID <= 0 || ID == id) {
    # network::getSkyArea(id, lag, T, var);       // calculate error regions
    # }
    pass


def get_chirp_mass(cluster: Cluster):
    # if (netRHO >= 0) {
    # ee = pwc->mchirp(id);        // original mchirp 2G
    # cc = Ec / (fabs(Ec) + ee);            // chirp cc
    # printf("mchirp_2g : %d %g %.2e %.3f %.3f %.3f %.3f \n\n",
    # int(id), cc, pwc->cData[id - 1].mchirp,
    # pwc->cData[id - 1].mchirperr, pwc->cData[id - 1].tmrgr,
    # pwc->cData[id - 1].tmrgrerr, pwc->cData[id - 1].chi2chirp);
    # } else {                // Enabled only for Search=CBC/BBH/IMBHB
    # if (m_chirp && (TString(Search) == "CBC" || TString(Search) == "BBH" || TString(Search) == "IMBHB")) {
    # ee = pwc->mchirp_upix(id, nRun);        // mchirp micropixel version
    # }
    # }
    pass
