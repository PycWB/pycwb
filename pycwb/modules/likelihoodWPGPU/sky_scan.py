"""
Sky scan — JAX-vectorised search over all sky directions.

The sky scan is the computational hot path of the likelihood pipeline.
This module fuses the per-sky-direction kernels from ``dpf`` and ``sky_stat``
into a single function, then ``vmap``-s it over all sky directions.

The result is an O(n_sky) batch of coherent statistics from which the
optimal sky location (l_max) is selected.

Mathematical reference: docs/likelihood/likelihoodWP.md
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from .dpf import compute_dpf
from .sky_stat import (
    compute_pixel_energy,
    project_gw_packet,
    orthogonalise_polarisations,
    compute_coherent_statistics,
)


# ---------------------------------------------------------------------------
# Per-sky-direction kernel (to be vmap-ped)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=())
def _sky_direction_statistics(Fp_sky: jnp.ndarray,
                              Fx_sky: jnp.ndarray,
                              rms: jnp.ndarray,
                              v00: jnp.ndarray,
                              v90: jnp.ndarray,
                              REG: jnp.ndarray,
                              netCC: jnp.ndarray,
                              delta_regulator: jnp.ndarray,
                              energy_threshold: jnp.ndarray,
                              n_ifo: jnp.ndarray) -> dict:
    """Compute all statistics for a single sky direction.

    Parameters
    ----------
    Fp_sky : (n_ifo,)
    Fx_sky : (n_ifo,)
    rms    : (n_pix, n_ifo) — same for all sky dirs, broadcast by vmap
    v00    : (n_ifo, n_pix) — time-delay-shifted data for THIS sky dir
    v90    : (n_ifo, n_pix)
    REG    : (3,) — regularisation vector
    netCC  : scalar
    delta_regulator : scalar
    energy_threshold : scalar
    n_ifo  : scalar int

    Returns
    -------
    dict of per-sky scalars used to build the skymap statistics arrays.
    """
    # 1. Pixel energy and mask
    pixel_info = compute_pixel_energy(v00, v90, energy_threshold)
    Eo = pixel_info["Eo"]
    total_energy = pixel_info["total_energy"]
    mask = pixel_info["mask"]

    # 2. DPF
    dpf = compute_dpf(Fp_sky, Fx_sky, rms)

    # 3. GW packet projection
    gw = project_gw_packet(
        v00, v90,
        dpf["f"], dpf["F"], dpf["fp"], dpf["fx"],
        dpf["network_index"], total_energy, mask, REG,
    )

    # 4. Orthogonalisation
    ort = orthogonalise_polarisations(gw["signal_00"], gw["signal_90"], gw["mask"])

    # 5. Coherent statistics
    stats = compute_coherent_statistics(
        v00, v90, gw["signal_00"], gw["signal_90"],
        ort["psi_sin"], ort["psi_cos"], gw["mask"],
    )

    Cr = stats["correlation"]
    Ec = stats["Ec"]
    No = stats["null_energy"]
    Mp = stats["Mp"]
    Mo = gw["n_active"].astype(jnp.float32)

    # Chi² in TF domain
    CH = No / (n_ifo * Mo + jnp.sqrt(Mo) + jnp.float32(1e-9))
    cc_factor = jnp.maximum(CH, jnp.float32(1.0))
    Co = Ec / (Ec + No * cc_factor - Mo * (n_ifo - 1) + jnp.float32(1e-9))

    # Likelihood and cross-correlation sky statistics
    aa = jnp.maximum(Eo - No, jnp.float32(0.0))
    AA = aa * Co

    # Antenna sensitivity (energy-weighted f+/fx norms)
    masked_energy = total_energy * mask.astype(jnp.float32)
    ee_sum = jnp.sum(masked_energy)
    ff_weighted = jnp.sum(dpf["fp"] * masked_energy)
    FF_weighted = jnp.sum(dpf["fx"] * masked_energy)
    safe_ee = jnp.maximum(ee_sum, jnp.float32(1e-12))
    ff_norm = ff_weighted / safe_ee
    FF_norm = FF_weighted / safe_ee

    antenna_prior = jnp.sqrt(ff_norm + FF_norm)
    alignment = jnp.where(ff_norm > 0, jnp.sqrt(FF_norm / ff_norm), jnp.float32(0.0))

    # Gate: zero ALL stats for directions below netCC threshold (mirrors CPU `continue`)
    passed = (Cr >= netCC)
    AA           = jnp.where(passed, aa * Co,         jnp.float32(0.0))
    antenna_prior = jnp.where(passed, antenna_prior,  jnp.float32(0.0))
    alignment     = jnp.where(passed, alignment,      jnp.float32(0.0))
    likelihood_s  = jnp.where(passed, Eo - No,        jnp.float32(0.0))
    null_energy_s = jnp.where(passed, No,             jnp.float32(0.0))
    coh_energy_s  = jnp.where(passed, Ec,             jnp.float32(0.0))
    correlation_s = jnp.where(passed, Co,             jnp.float32(0.0))
    disbalance_s  = jnp.where(passed, CH,             jnp.float32(0.0))
    net_index_s   = jnp.where(passed, cc_factor,      jnp.float32(0.0))
    ellipticity_s = jnp.where(passed, Cr,             jnp.float32(0.0))
    polarisation_s = jnp.where(passed, Mp,            jnp.float32(0.0))

    return {
        "AA": AA,                    # cross-correlation skystat (used for l_max selection)
        "antenna_prior": antenna_prior,
        "alignment": alignment,
        "likelihood": likelihood_s,
        "null_energy": null_energy_s,
        "coherent_energy": coh_energy_s,
        "correlation": correlation_s,
        "sky_stat": AA,
        "disbalance": disbalance_s,
        "net_index": net_index_s,
        "ellipticity": ellipticity_s,
        "polarisation": polarisation_s,
    }


# ---------------------------------------------------------------------------
# Full sky scan
# ---------------------------------------------------------------------------

def find_optimal_sky_localization(n_ifo: int,
                                 n_pix: int,
                                 n_sky: int,
                                 FP: np.ndarray,
                                 FX: np.ndarray,
                                 rms: np.ndarray,
                                 td00: np.ndarray,
                                 td90: np.ndarray,
                                 ml: np.ndarray,
                                 REG: np.ndarray,
                                 netCC: float,
                                 delta_regulator: float,
                                 network_energy_threshold: float,
                                 sky_batch_size: int = 8192):
    """Find the sky direction that maximises the cross-correlation statistic.

    This is the JAX equivalent of the Numba ``find_optimal_sky_localization``.
    It uses ``jax.vmap`` to evaluate all sky directions in parallel, processing
    them in batches of ``sky_batch_size`` to bound peak VRAM usage.

    Parameters
    ----------
    n_ifo : int
    n_pix : int
    n_sky : int
    FP, FX : shape (n_sky, n_ifo) float32
    rms : shape (n_pix, n_ifo) float32  — NOTE: (n_pix, n_ifo) for GPU layout
    td00, td90 : shape (n_delay, n_ifo, n_pix) float32
    ml : shape (n_ifo, n_sky) int
    REG : shape (3,) float32
    netCC : float
    delta_regulator : float
    network_energy_threshold : float
    sky_batch_size : int
        Number of sky directions processed per vmap call.  Reduce this if
        GPU/XLA runs out of memory for large clusters.  Default: 8192.

    Returns
    -------
    SkyMapStatistics-compatible tuple:
        (l_max, nAntennaPrior, nAlignment, nLikelihood, nNullEnergy,
         nCorrEnergy, nCorrelation, nSkyStat, nDisbalance, nNetIndex,
         nEllipticity, nPolarisation)
    """
    # Convert inputs to JAX arrays
    FP_j = jnp.asarray(FP, dtype=jnp.float32)
    FX_j = jnp.asarray(FX, dtype=jnp.float32)
    rms_j = jnp.asarray(rms, dtype=jnp.float32)  # (n_pix, n_ifo)
    td00_j = jnp.asarray(td00, dtype=jnp.float32)
    td90_j = jnp.asarray(td90, dtype=jnp.float32)
    ml_j = jnp.asarray(ml, dtype=jnp.int32)
    REG_j = jnp.asarray(REG, dtype=jnp.float32)
    netCC_j = jnp.float32(netCC)
    delta_j = jnp.float32(delta_regulator)
    ethr_j = jnp.float32(network_energy_threshold)
    nifo_j = jnp.float32(n_ifo)

    offset = td00_j.shape[0] // 2

    # --- Build per-sky delayed data: v00[l], v90[l] ---
    # ml_j: (n_ifo, n_sky)
    # For each sky l and ifo i: v00[i,j] = td00[ml[i,l]+offset, i, j]
    # We gather in a vmap-friendly way.
    def _gather_delayed_data(ml_col):
        """Gather delayed data for one sky direction.

        ml_col: (n_ifo,) — delay indices for this sky direction.
        Returns v00, v90 each (n_ifo, n_pix).
        """
        # For each ifo i, select td00[ml_col[i]+offset, i, :]
        delay_indices = ml_col + offset  # (n_ifo,)
        # Use advanced indexing: td00[delay_indices, ifo_range, :]
        ifo_range = jnp.arange(td00_j.shape[1])
        v00_sky = td00_j[delay_indices, ifo_range, :]  # (n_ifo, n_pix)
        v90_sky = td90_j[delay_indices, ifo_range, :]
        return v00_sky, v90_sky

    # ml_j: (n_ifo, n_sky) → transpose to (n_sky, n_ifo) for vmap axis 0
    ml_t = ml_j.T  # (n_sky, n_ifo)

    # --- Batched vmap over sky directions ---
    # Process sky_batch_size directions at a time to bound peak VRAM usage.
    # For each batch: gather delayed data, compute sky statistics, transfer to numpy.
    result_keys = [
        "AA", "antenna_prior", "alignment", "likelihood", "null_energy",
        "coherent_energy", "correlation", "sky_stat", "disbalance",
        "net_index", "ellipticity", "polarisation",
    ]
    result_parts = {k: [] for k in result_keys}

    for _start in range(0, n_sky, sky_batch_size):
        _end = min(_start + sky_batch_size, n_sky)

        # Gather delayed data for this batch: (batch, n_ifo, n_pix) each
        _all_v00, _all_v90 = jax.vmap(_gather_delayed_data)(ml_t[_start:_end])

        # Compute per-sky statistics for this batch
        _chunk = jax.vmap(
            _sky_direction_statistics,
            in_axes=(0, 0, None, 0, 0, None, None, None, None, None),
        )(FP_j[_start:_end], FX_j[_start:_end], rms_j,
          _all_v00, _all_v90,
          REG_j, netCC_j, delta_j, ethr_j, nifo_j)

        # Transfer batch results to CPU numpy to free device memory
        for k in result_keys:
            result_parts[k].append(np.asarray(_chunk[k]))

    # Concatenate all batches
    sky_results_np = {k: np.concatenate(result_parts[k]) for k in result_keys}

    # --- Find l_max: last index with maximum AA (mirrors C++ tie-breaking) ---
    AA_np = sky_results_np["AA"]
    STAT = np.float32(-1e12)
    l_max = 0
    for _l in range(n_sky):
        if AA_np[_l] >= STAT:
            STAT = AA_np[_l]
            l_max = _l

    # --- Collect per-sky arrays ---
    nAntennaPrior = sky_results_np["antenna_prior"].astype(np.float32)
    nAlignment    = sky_results_np["alignment"].astype(np.float32)
    nLikelihood   = sky_results_np["likelihood"].astype(np.float32)
    nNullEnergy   = sky_results_np["null_energy"].astype(np.float32)
    nCorrEnergy   = sky_results_np["coherent_energy"].astype(np.float32)
    nCorrelation  = sky_results_np["correlation"].astype(np.float32)
    nSkyStat      = sky_results_np["sky_stat"].astype(np.float32)
    nDisbalance   = sky_results_np["disbalance"].astype(np.float32)
    nNetIndex     = sky_results_np["net_index"].astype(np.float32)
    nEllipticity  = sky_results_np["ellipticity"].astype(np.float32)
    nPolarisation = sky_results_np["polarisation"].astype(np.float32)

    return (l_max, nAntennaPrior, nAlignment, nLikelihood, nNullEnergy,
            nCorrEnergy, nCorrelation, nSkyStat, nDisbalance, nNetIndex,
            nEllipticity, nPolarisation)
