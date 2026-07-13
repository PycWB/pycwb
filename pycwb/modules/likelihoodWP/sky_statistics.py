"""Sky statistics: compute detailed statistics at a sky position.

Provides :func:`compute_statistics_at_sky_position` which computes per-pixel
statistics (packet energies, SNRs, noise corrections) at the
best-fit sky direction.

Legacy alias ``calculate_sky_statistics`` remains available.
"""

from __future__ import annotations

import logging
from math import sqrt
import numpy as np
from .typing import SkyStatistics
from .dpf import dpf_np_loops_vec
from .packet_ops import (
    avx_packet_ps, packet_norm_numpy, gw_norm_numpy, avx_noise_ps,
    avx_setAMP_ps, avx_pol_ps, avx_loadNULL_ps, xtalk_energy_sum_numpy,
)
from .sky_stat import avx_GW_ps, avx_ort_ps, avx_stat_ps, load_data_from_td
from pycwb.modules.xtalk.type import XTalk

logger = logging.getLogger(__name__)

def compute_statistics_at_sky_position(
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
    """
    Compute detailed statistics for a specific sky location.
    Parameters
    ----------
    sky_idx : int
        Index of the sky location.
    n_ifo : int
        Number of interferometers.
    n_pix : int
        Number of pixels.
    FP : np.ndarray
        f+ polarization data for each interferometer, shape (n_sky, nIFO).
    FX : np.ndarray
        fx polarization data for each interferometer, shape (n_sky, nIFO).
    rms : np.ndarray
        RMS values, shape (nIFO, n_pix).
    td00 : np.ndarray
        Time-delayed in-phase data, shape (ndelay, nIFO, n_pix).
    td90 : np.ndarray
        Time-delayed quadrature data, shape (ndelay, nIFO, n_pix).
    ml : np.ndarray
        Sky-delay index array, shape (nIFO, n_sky).
    REG : np.ndarray
        Regularization parameters, shape (3,).
    network_energy_threshold : float
        Energy threshold for the network.
    cluster_xtalk : object
        Cluster XTalk object containing cross-talk coefficients.
    cluster_xtalk_lookup_table : object
        Lookup table for cross-talk.
    DEBUG : bool, optional
        If True, emit extra debug output. Default is False.

    Returns
    -------
    SkyStatistics
        Dataclass containing the sky statistics for the specified sky location.
    """
    # Keep legacy parameter names for keyword compatibility, but use readable
    # local names inside the calculation.
    plus_antenna_patterns = FP
    cross_antenna_patterns = FX
    noise_weights = rms
    td_phase0 = td00
    td_phase90 = td90
    sky_delay_samples = ml
    regularization = REG

    data_phase0 = np.empty((n_ifo, n_pix), dtype=np.float32)
    data_phase90 = np.empty((n_ifo, n_pix), dtype=np.float32)
    td_energy = np.zeros((n_ifo, n_pix), dtype=np.float32)

    offset = int(td_phase0.shape[0] / 2)

    # --- Apply time delay for this sky direction ---
    for i in range(n_ifo):
        data_phase0[i] = td_phase0[sky_delay_samples[i, sky_idx] + offset, i]
        data_phase90[i] = td_phase90[sky_delay_samples[i, sky_idx] + offset, i]

    # Per-pixel TF energy (data_phase0² + data_phase90²)
    for i in range(n_ifo):
        for j in range(n_pix):
            td_energy[i, j] = data_phase0[i, j] * data_phase0[i, j] + data_phase90[i, j] * data_phase90[i, j]

    # --- Compute total energy, pixel activity mask ---
    total_data_energy, _, energy_total, mask = load_data_from_td(
        data_phase0, data_phase90, network_energy_threshold,
    )

    # --- Dominant polarisation frame: f+/fx projections and norms ---
    _, dominant_plus, dominant_cross, plus_norm, cross_norm, rotation_sin, rotation_cos, network_index = dpf_np_loops_vec(
        plus_antenna_patterns[sky_idx], cross_antenna_patterns[sky_idx], noise_weights,
    )

    # --- Project onto GW strain packet; select pixels above threshold ---
    active_pixel_count, signal_phase0, signal_phase90, mask, _, _, _, _ = avx_GW_ps(
        data_phase0, data_phase90,
        dominant_plus, dominant_cross,
        plus_norm, cross_norm, network_index,
        energy_total, mask, regularization,
    )

    # --- Orthogonalise signal amplitudes (+ and x polarisations) ---
    signal_packet_energy, rotation_sin, rotation_cos, energy_array_plus, energy_array_cross = avx_ort_ps(signal_phase0, signal_phase90, mask)

    # --- Coherent network statistics ---
    _, _, _, _, coherent_energy, gaussian_noise_per_pixel, residual_noise_per_pixel = avx_stat_ps(
        data_phase0, data_phase90, signal_phase0, signal_phase90, rotation_sin, rotation_cos, mask
    )

    # --- Build data and signal packets; compute xtalk-corrected SNRs ---
    total_data_energy, packet_data_phase0, packet_data_phase90, data_packet_energy, data_rotation_sin, data_rotation_cos, data_amplitude0, data_amplitude90 = avx_packet_ps(data_phase0, data_phase90, mask)
    total_signal_packet_energy, packet_signal_phase0, packet_signal_phase90, signal_packet_energy_by_detector, signal_rotation_sin, signal_rotation_cos, signal_amplitude0, signal_amplitude90 = avx_packet_ps(signal_phase0, signal_phase90, mask)

    detector_snr, data_packet_energy, residual_noise_per_pixel, data_packet_norm = packet_norm_numpy(
        packet_data_phase0, packet_data_phase90, cluster_xtalk, cluster_xtalk_lookup_table, mask, data_packet_energy
    )
    total_detector_snr = np.sum(detector_snr)
    total_signal_snr, signal_snr_by_detector, signal_packet_energy_by_detector, signal_packet_norm = gw_norm_numpy(
        data_packet_norm, data_packet_energy, signal_packet_energy_by_detector, coherent_energy
    )
    if DEBUG:
        print(total_signal_snr, signal_snr_by_detector)
        print("Eo = ", total_data_energy, ", Lo = ", total_signal_packet_energy, ", Ep = ", total_detector_snr, ", Lp = ", total_signal_snr)

    # --- Gaussian-noise correction and coherent energy decomposition ---
    # Returns: Gn (Gaussian noise), Ec (core coherent energy), Dc (signal-core coherent energy),
    #          Rc (EC normalisation), Eh (satellite/halo energy), Es, NC, NS
    # TODO: one more pixel selected, need to be fixed
    gaussian_noise, core_coherent_energy, signal_core_coherent_energy, network_correlation, halo_energy, signal_energy, network_count, signal_count = avx_noise_ps(
        signal_packet_norm, data_packet_norm, energy_total, mask,
        coherent_energy, gaussian_noise_per_pixel, residual_noise_per_pixel,
    )

    if DEBUG:
        print(
            "Gn = ", gaussian_noise,
            ", Ec = ", core_coherent_energy,
            ", Dc = ", signal_core_coherent_energy,
            ", Rc = ", network_correlation,
            ", Eh = ", halo_energy,
            ", Es = ", signal_energy,
            ", NC = ", network_count,
            ", NS = ", signal_count,
        )

    # --- Set packet amplitudes and compute time-domain null / energy ---
    effective_pixel_count, packet_data_phase0, packet_data_phase90 = avx_setAMP_ps(
        packet_data_phase0, packet_data_phase90, data_packet_norm,
        data_rotation_sin, data_rotation_cos, data_amplitude0, data_amplitude90, mask,
    )
    effective_pixel_count = effective_pixel_count - 1
    _, packet_signal_phase0, packet_signal_phase90 = avx_setAMP_ps(
        packet_signal_phase0, packet_signal_phase90, signal_packet_norm,
        signal_rotation_sin, signal_rotation_cos, signal_amplitude0, signal_amplitude90, mask,
    )
    null_phase0, null_phase90 = avx_loadNULL_ps(
        packet_data_phase0, packet_data_phase90, packet_signal_phase0, packet_signal_phase90
    )

    # Raw xtalk sums (no clamping, mirrors C++ _avx_norm_ps(-V4))
    _, data_packet_energy, residual_noise_per_pixel, _ = packet_norm_numpy(
        packet_data_phase0, packet_data_phase90, cluster_xtalk, cluster_xtalk_lookup_table, mask, data_packet_energy
    )
    xtalk_data_energy = xtalk_energy_sum_numpy(
        packet_data_phase0, packet_data_phase90, cluster_xtalk, cluster_xtalk_lookup_table, mask,
    )
    xtalk_null_energy = xtalk_energy_sum_numpy(
        null_phase0, null_phase90, cluster_xtalk, cluster_xtalk_lookup_table, mask,
    )
    total_detector_snr = xtalk_data_energy  # legacy Ep equivalent
    time_domain_signal_energy = xtalk_data_energy - xtalk_null_energy - gaussian_noise
    energy_normalization = (total_data_energy - halo_energy) / xtalk_data_energy if xtalk_data_energy > 0 else 1.e9
    if energy_normalization < 1:
        energy_normalization = 1
    core_coherent_energy /= energy_normalization  # core coherent energy normalised to time domain
    signal_core_coherent_energy /= energy_normalization  # signal-core coherent energy normalised to time domain
    chi_square_noise = (xtalk_null_energy + gaussian_noise) / (effective_pixel_count * n_ifo)
    if DEBUG:
        print(
            "Np = ", xtalk_null_energy,
            ", Em = ", xtalk_data_energy,
            ", Lm = ", time_domain_signal_energy,
            ", norm = ", energy_normalization,
            ", Ec = ", core_coherent_energy,
            ", Dc = ", signal_core_coherent_energy,
            ", ch = ", chi_square_noise,
        )

    # --- Detection statistic rho (mode-dependent) ---
    reference_rho = 0.
    xgb_penalty = 0.
    xgb_coherent_energy = 0.
    if not xgb_rho_mode:  # original 2G
        noise_correction_factor = chi_square_noise if chi_square_noise > 1 else 1
        detection_rho = np.sqrt(core_coherent_energy * network_correlation / 2.) if core_coherent_energy > 0 else 0
        if DEBUG:
            print("cc = ", noise_correction_factor, ", rho = ", detection_rho)
    else:  # XGB.rho0
        xgb_penalty = chi_square_noise
        xgb_coherent_energy = core_coherent_energy
        # TODO: The ecor can be negative for certain cases, which causes rho to be NaN. 
        # And in Python NaN < netRHO is always False, which means the cluster will never be rejected by the rho threshold cut. 
        # This is fixed in cWB 6.9.6.9. The investigation of the root cause of negative ecor is leave to the future,
        # for now we clamp it to zero to avoid NaN issues and ensure proper thresholding.
        detection_rho = (
            np.sqrt(xgb_coherent_energy / (1 + xgb_penalty * (max(float(1), xgb_penalty) - 1)))
            if xgb_coherent_energy > 0 else 0
        )
        noise_correction_factor = chi_square_noise if chi_square_noise > 1 else 1
        reference_rho = (
            np.sqrt(core_coherent_energy * network_correlation / 2.) if core_coherent_energy > 0 else 0
        )
        if DEBUG:
            print(
                "cc = ", noise_correction_factor,
                ", rho = ", detection_rho,
                ", ecor = ", xgb_coherent_energy,
                ", penalty = ", xgb_penalty,
                ", xrho = ", reference_rho,
            )

    # --- Project residuals onto network polarisation plane (Dual Stream Transform) ---
    data_phase0, data_phase90, p00_POL, p90_POL = avx_pol_ps(
        data_phase0, data_phase90, mask, plus_norm, cross_norm, dominant_plus, dominant_cross
    )
    data_phase0, data_phase90, r00_POL, r90_POL = avx_pol_ps(
        data_phase0, data_phase90, mask, plus_norm, cross_norm, dominant_plus, dominant_cross
    )

    return SkyStatistics(
        Gn=np.float32(gaussian_noise),
        Ec=np.float32(core_coherent_energy),
        Dc=np.float32(signal_core_coherent_energy),
        Rc=np.float32(network_correlation),
        Eh=np.float32(halo_energy),
        Es=np.float32(signal_energy),
        Np=np.float32(xtalk_null_energy),
        Em=np.float32(xtalk_data_energy),
        Lm=np.float32(time_domain_signal_energy),
        norm=np.float32(energy_normalization),
        cc=np.float32(noise_correction_factor),
        rho=np.float32(detection_rho),
        xrho=np.float32(reference_rho),
        Lo=np.float32(total_signal_packet_energy),
        Eo=np.float32(total_data_energy),
        energy_array_plus=energy_array_plus,
        energy_array_cross=energy_array_cross,
        pixel_mask=mask,
        v00=data_phase0,
        v90=data_phase90,
        gaussian_noise_correction=gaussian_noise_per_pixel,
        coherent_energy=coherent_energy,
        N_pix_effective=effective_pixel_count,
        noise_amplitude_00=null_phase0,
        noise_amplitude_90=null_phase90,
        pd=packet_data_phase0,
        pD=packet_data_phase90,
        ps=packet_signal_phase0,
        pS=packet_signal_phase90,
        p00_POL=p00_POL,
        p90_POL=p90_POL,
        r00_POL=r00_POL,
        r90_POL=r90_POL,
        S_snr=signal_snr_by_detector,
        f=dominant_plus,
        F=dominant_cross,
    )



# Legacy alias
calculate_sky_statistics = compute_statistics_at_sky_position

__all__ = [
    "calculate_sky_statistics",
    "compute_statistics_at_sky_position",
]
