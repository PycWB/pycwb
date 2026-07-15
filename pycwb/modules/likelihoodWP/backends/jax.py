"""JAX likelihood kernels and their declarative selection bundle.

The orchestration, cuts, reconstruction, confidence products, and metadata are
owned by :mod:`pycwb.modules.likelihoodWP.likelihood`. This module implements
the JAX best-fit statistic and selects the three direct JAX kernel functions.
"""

from __future__ import annotations

import numpy as np

from .base import LikelihoodKernels
from ..packet_ops import avx_packet_ps, gw_norm_numpy, packet_norm_numpy
from ..typing import SkyStatistics

from .jax_dpf import (
    calculate_dpf_regulator,
    compute_dpf,
)
from .jax_sky_scan import find_optimal_sky_localization
from .jax_sky_stat import (
    compute_coherent_statistics,
    compute_pixel_energy,
    orthogonalise_polarisations,
    project_gw_packet,
)
from .jax_packet_ops import (
    compute_noise_correction,
    compute_null_packet,
    project_polarisation,
    set_packet_amplitudes,
    xtalk_energy_sum,
)


def calculate_sky_statistics_jax(
    sky_idx: int,
    n_ifo: int,
    n_pix: int,
    plus_patterns: np.ndarray,
    cross_patterns: np.ndarray,
    noise_weights: np.ndarray,
    td_phase0: np.ndarray,
    td_phase90: np.ndarray,
    sky_delays: np.ndarray,
    regularization: np.ndarray,
    network_energy_threshold: float,
    cluster_xtalk: np.ndarray,
    cluster_xtalk_lookup: np.ndarray,
    debug: bool = False,
    xgb_rho_mode: bool = False,
) -> SkyStatistics:
    """Detailed best-fit statistics using JAX kernels and sparse Numba ops."""

    import jax.numpy as jnp

    offset = int(td_phase0.shape[0] / 2)
    v00 = np.empty((n_ifo, n_pix), dtype=np.float32)
    v90 = np.empty((n_ifo, n_pix), dtype=np.float32)
    for detector in range(n_ifo):
        delay = int(sky_delays[detector, sky_idx]) + offset
        v00[detector] = td_phase0[delay, detector]
        v90[detector] = td_phase90[delay, detector]

    v00_j = jnp.asarray(v00, dtype=jnp.float32)
    v90_j = jnp.asarray(v90, dtype=jnp.float32)
    rms_j = jnp.asarray(noise_weights, dtype=jnp.float32)
    plus_j = jnp.asarray(plus_patterns[sky_idx], dtype=jnp.float32)
    cross_j = jnp.asarray(cross_patterns[sky_idx], dtype=jnp.float32)
    reg_j = jnp.asarray(regularization, dtype=jnp.float32)

    pixel_info = compute_pixel_energy(
        v00_j, v90_j, jnp.float32(network_energy_threshold)
    )
    total_energy_j = pixel_info["total_energy"]
    e_original = float(pixel_info["Eo"])

    dpf = compute_dpf(plus_j, cross_j, rms_j)
    gw = project_gw_packet(
        v00_j,
        v90_j,
        dpf["f"],
        dpf["F"],
        dpf["fp"],
        dpf["fx"],
        dpf["network_index"],
        total_energy_j,
        pixel_info["mask"],
        reg_j,
    )
    orthogonal = orthogonalise_polarisations(
        gw["signal_00"], gw["signal_90"], gw["mask"]
    )
    coherent = compute_coherent_statistics(
        v00_j,
        v90_j,
        gw["signal_00"],
        gw["signal_90"],
        orthogonal["psi_sin"],
        orthogonal["psi_cos"],
        gw["mask"],
    )

    mask = np.asarray(gw["mask"])
    total_energy = np.asarray(total_energy_j)
    f = np.asarray(dpf["f"])
    F = np.asarray(dpf["F"])
    fp = np.asarray(dpf["fp"])
    fx = np.asarray(dpf["fx"])
    signal0 = np.asarray(gw["signal_00"])
    signal90 = np.asarray(gw["signal_90"])
    coherent_energy = np.asarray(coherent["ec"])
    gaussian_noise = np.asarray(coherent["gn"])
    residual_noise = np.asarray(coherent["rn"])
    energy_plus = np.asarray(orthogonal["energy_plus"])
    energy_cross = np.asarray(orthogonal["energy_cross"])
    l_original = float(orthogonal["total_energy"])

    _, pd, pD, data_energy, data_sin, data_cos, data_a, data_A = avx_packet_ps(
        np.asarray(v00), np.asarray(v90), np.asarray(mask)
    )
    _, ps, pS, signal_energy, signal_sin, signal_cos, signal_a, signal_A = avx_packet_ps(
        signal0, signal90, np.asarray(mask)
    )

    detector_snr, data_energy, _, data_norm = packet_norm_numpy(
        pd, pD, cluster_xtalk, cluster_xtalk_lookup, mask, data_energy
    )
    data_snr = np.sum(detector_snr)
    signal_snr, per_detector_signal_snr, signal_energy, signal_norm = gw_norm_numpy(
        data_norm, data_energy, signal_energy, coherent_energy
    )

    if debug:
        print(signal_snr, per_detector_signal_snr)
        print("Eo =", e_original, ", Lo =", l_original, ", Ep =", data_snr)

    Gn, Ec, Dc, Rc, Eh, Es, _, _ = compute_noise_correction(
        signal_norm,
        data_norm,
        total_energy,
        mask,
        coherent_energy,
        gaussian_noise,
        residual_noise,
    )

    n_effective, pd, pD = set_packet_amplitudes(
        pd, pD, data_norm, data_sin, data_cos, data_a, data_A, mask
    )
    n_effective -= 1
    _, ps, pS = set_packet_amplitudes(
        ps, pS, signal_norm, signal_sin, signal_cos, signal_a, signal_A, mask
    )
    pn, pN = compute_null_packet(pd, pD, ps, pS)

    Em = xtalk_energy_sum(pd, pD, cluster_xtalk, cluster_xtalk_lookup, mask)
    Np = xtalk_energy_sum(pn, pN, cluster_xtalk, cluster_xtalk_lookup, mask)
    Lm = Em - Np - Gn
    norm = (e_original - Eh) / Em if Em > 0 else 1.0e9
    norm = max(norm, 1.0)
    Ec /= norm
    Dc /= norm
    disbalance = (Np + Gn) / (n_effective * n_ifo)

    xrho = 0.0
    if not xgb_rho_mode:
        cc = max(disbalance, 1.0)
        rho = np.sqrt(Ec * Rc / 2.0) if Ec > 0 else 0.0
    else:
        penalty = disbalance
        rho = np.sqrt(Ec / (1 + penalty * (max(1.0, penalty) - 1)))
        cc = max(disbalance, 1.0)
        xrho = np.sqrt(Ec * Rc / 2.0) if Ec > 0 else 0.0

    v00_out, v90_out, p00_pol, p90_pol = project_polarisation(
        v00, v90, mask, fp, fx, f, F
    )
    v00_out, v90_out, r00_pol, r90_pol = project_polarisation(
        v00_out, v90_out, mask, fp, fx, f, F
    )

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
        Lo=np.float32(l_original),
        Eo=np.float32(e_original),
        N_pix_effective=np.float32(n_effective),
        energy_array_plus=energy_plus,
        energy_array_cross=energy_cross,
        v00=v00_out,
        v90=v90_out,
        pd=pd,
        pD=pD,
        ps=ps,
        pS=pS,
        pixel_mask=mask,
        gaussian_noise_correction=gaussian_noise,
        noise_amplitude_00=pn,
        noise_amplitude_90=pN,
        coherent_energy=coherent_energy,
        p00_POL=p00_pol,
        p90_POL=p90_pol,
        r00_POL=r00_pol,
        r90_POL=r90_pol,
        S_snr=per_detector_signal_snr,
        f=f,
        F=F,
    )


KERNELS = LikelihoodKernels(
    name="jax",
    calculate_dpf_regulator=calculate_dpf_regulator,
    scan_sky=find_optimal_sky_localization,
    statistics_at_best_fit=calculate_sky_statistics_jax,
)

__all__ = [
    "calculate_sky_statistics_jax",
    "KERNELS",
]
