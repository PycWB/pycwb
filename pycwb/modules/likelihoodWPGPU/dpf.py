"""
Dominant Polarization Frame (DPF) construction — JAX implementation.

The DPF rotates the antenna response vectors (F+, Fx) at each pixel into a
basis where the plus-like axis captures maximum signal power.  This module
provides:

- ``compute_dpf``: per-sky-direction DPF solving (f+, fx norms, rotation
  angles, network index).  Designed to be ``vmap``-ped over sky directions.
- ``calculate_dpf_regulator``: sky-averaged energy regulator (REG[1]).

Mathematical reference: docs/likelihood/likelihoodWP.md §"Dominant Polarization Frame"

Variable naming conventions (matching the math doc):
    Fp_sky, Fx_sky  — antenna patterns for one sky direction, shape (n_ifo,)
    rms             — noise-weighted detector response, shape (n_pix, n_ifo)
    f, F            — rotated response vectors in DPF basis, shape (n_pix, n_ifo)
    fp, fx          — squared norms |f+|² and |fx|², shape (n_pix,)
    psi_sin, psi_cos — DPF rotation sin/cos per pixel, shape (n_pix,)
    network_index   — per-pixel network index, shape (n_pix,)
"""

import jax
import jax.numpy as jnp
from functools import partial


# ---------------------------------------------------------------------------
# Core DPF for a single sky direction
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=())
def compute_dpf(Fp_sky: jnp.ndarray,
                Fx_sky: jnp.ndarray,
                rms: jnp.ndarray) -> dict:
    """Compute the Dominant Polarization Frame for one sky direction.

    Parameters
    ----------
    Fp_sky : jnp.ndarray, shape (n_ifo,)
        Plus antenna pattern for this sky direction.
    Fx_sky : jnp.ndarray, shape (n_ifo,)
        Cross antenna pattern for this sky direction.
    rms : jnp.ndarray, shape (n_pix, n_ifo)
        Per-pixel noise-weighted detector response.

    Returns
    -------
    dict with keys:
        f          — rotated plus response, shape (n_pix, n_ifo)
        F          — rotated cross response (orthogonalised), shape (n_pix, n_ifo)
        fp         — |f+|² per pixel, shape (n_pix,)
        fx         — |fx|² per pixel, shape (n_pix,)
        psi_sin    — sin(ψ) per pixel, shape (n_pix,)
        psi_cos    — cos(ψ) per pixel, shape (n_pix,)
        network_index — per-pixel network index, shape (n_pix,)
        dpf_quality   — scalar DPF quality measure (NI)
    """
    EPS = jnp.float32(1e-4)

    # --- Weighted response vectors: f0 = rms * Fp, F0 = rms * Fx ---
    # rms: (n_pix, n_ifo), Fp_sky: (n_ifo,) → broadcast: (n_pix, n_ifo)
    f0 = rms * Fp_sky[jnp.newaxis, :]
    F0 = rms * Fx_sky[jnp.newaxis, :]

    # --- Inner products per pixel (sum over ifo axis) ---
    ff = jnp.sum(f0 * f0, axis=1)     # (n_pix,)
    FF = jnp.sum(F0 * F0, axis=1)     # (n_pix,)
    fF = jnp.sum(f0 * F0, axis=1)     # (n_pix,)

    # --- Rotation angle ---
    sin_2psi = 2.0 * fF               # rotation 2·sin·cos·norm
    cos_2psi = ff - FF                 # rotation (cos²-sin²)·norm
    total_antenna = ff + FF            # total antenna power
    norm_2psi = jnp.sqrt(cos_2psi ** 2 + sin_2psi ** 2)

    # Dominant polarisation energy
    fp = (total_antenna + norm_2psi) / 2.0

    # cos(2ψ) normalised
    cos_2psi_n = cos_2psi / (norm_2psi + EPS)

    # Recover sin(ψ) and cos(ψ) from cos(2ψ)
    sign = jnp.where(sin_2psi > 0, jnp.float32(1.0), jnp.float32(-1.0))
    psi_sin = jnp.sqrt((1.0 - cos_2psi_n) / 2.0)  # |sin(ψ)|
    psi_cos = jnp.sqrt((1.0 + cos_2psi_n) / 2.0) * sign  # cos(ψ) with sign

    # --- Rotate into DPF basis ---
    # f_rot = f0·cos(ψ) + F0·sin(ψ)
    # F_rot = F0·cos(ψ) − f0·sin(ψ)
    f_rot = f0 * psi_cos[:, jnp.newaxis] + F0 * psi_sin[:, jnp.newaxis]
    F_rot = F0 * psi_cos[:, jnp.newaxis] - f0 * psi_sin[:, jnp.newaxis]

    # --- Orthogonalise F w.r.t. f ---
    # Project out f-component: F⊥ = F − f·(f·F)/(|f|²)
    fF_rot = jnp.sum(f_rot * F_rot, axis=1)  # (n_pix,)
    projection = fF_rot / (fp + EPS)
    F_orth = F_rot - f_rot * projection[:, jnp.newaxis]

    # Cross polarisation norm
    fx = jnp.sum(F_orth * F_orth, axis=1)  # (n_pix,)

    # --- Network index ---
    f4_sum = jnp.sum(f_rot ** 4, axis=1)   # Σ_i f_i⁴
    network_index = f4_sum / (fp ** 2 + EPS)

    # --- DPF quality statistic ---
    N_plus = jnp.sum(jnp.where(fp > 0, jnp.float32(1.0), jnp.float32(0.0)))
    NI_sum = jnp.sum(fx / (network_index + EPS))
    dpf_quality = jnp.sqrt(NI_sum / (N_plus + jnp.float32(0.01)))

    return {
        "f": f_rot,
        "F": F_orth,
        "fp": fp,
        "fx": fx,
        "psi_sin": psi_sin,
        "psi_cos": psi_cos,
        "network_index": network_index,
        "dpf_quality": dpf_quality,
    }


# ---------------------------------------------------------------------------
# Sky-averaged DPF energy regulator
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=())
def _dpf_quality_single(Fp_sky, Fx_sky, rms):
    """Return the scalar DPF quality for one sky direction."""
    return compute_dpf(Fp_sky, Fx_sky, rms)["dpf_quality"]


def calculate_dpf_regulator(FP: jnp.ndarray,
                            FX: jnp.ndarray,
                            rms: jnp.ndarray,
                            gamma_regulator: float,
                            network_energy_threshold: float,
                            sky_batch_size: int = 8192) -> float:
    """Compute the DPF-based energy regulator REG[1].

    Scans all sky directions, counts how many have DPF quality above
    ``gamma_regulator``, and returns:

        REG[1] = (N_sky² / (N_gamma² + ε) − 1) · E_thr

    Parameters
    ----------
    FP : jnp.ndarray, shape (n_sky, n_ifo)
        Plus antenna patterns for all sky directions.
    FX : jnp.ndarray, shape (n_sky, n_ifo)
        Cross antenna patterns for all sky directions.
    rms : jnp.ndarray, shape (n_pix, n_ifo)
        Per-pixel noise-weighted detector response.
    gamma_regulator : float
        DPF quality threshold (γ²·2/3).
    network_energy_threshold : float
        Per-pixel energy threshold E_thr.
    sky_batch_size : int
        Number of sky directions processed per vmap call.  Reduce this if
        GPU/XLA runs out of memory for large clusters.  Default: 8192.

    Returns
    -------
    float
        The energy regulator value REG[1].
    """
    import numpy as _np

    n_sky_total = FP.shape[0]
    FP_j = jnp.asarray(FP, dtype=jnp.float32)
    FX_j = jnp.asarray(FX, dtype=jnp.float32)
    rms_j = jnp.asarray(rms, dtype=jnp.float32)

    quality_parts = []
    for start in range(0, n_sky_total, sky_batch_size):
        end = min(start + sky_batch_size, n_sky_total)
        batch_qualities = jax.vmap(
            _dpf_quality_single, in_axes=(0, 0, None)
        )(FP_j[start:end], FX_j[start:end], rms_j)
        quality_parts.append(_np.asarray(batch_qualities))

    dpf_qualities = _np.concatenate(quality_parts)

    n_sky = float(n_sky_total)
    n_gamma = float(_np.sum(dpf_qualities > gamma_regulator))

    reg = (n_sky ** 2 / (n_gamma ** 2 + 1e-9) - 1.0) * network_energy_threshold
    return float(reg)
