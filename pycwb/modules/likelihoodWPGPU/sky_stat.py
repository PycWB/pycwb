"""
Sky-direction statistics kernels — JAX implementation.

These functions compute the per-pixel coherent statistics at a single sky
direction.  They are designed to be composed and ``vmap``-ped over sky
directions for the full sky scan.

Kernel pipeline (per sky direction):
    1. ``compute_pixel_energy``    — total energy and active-pixel mask
    2. ``project_gw_packet``       — project data onto DPF, build coherent signal packet
    3. ``orthogonalise_polarisations`` — orthogonalise + / × amplitudes
    4. ``compute_coherent_statistics`` — correlation, coherent energy, noise correction

Mathematical reference: docs/likelihood/likelihoodWP.md
"""

import jax
import jax.numpy as jnp
from functools import partial


# ---------------------------------------------------------------------------
# 1. Pixel energy and mask
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=())
def compute_pixel_energy(v00: jnp.ndarray,
                         v90: jnp.ndarray,
                         energy_threshold: jnp.ndarray) -> dict:
    """Compute per-pixel network energy and active-pixel mask.

    Parameters
    ----------
    v00 : jnp.ndarray, shape (n_ifo, n_pix)
        In-phase delayed data at this sky direction.
    v90 : jnp.ndarray, shape (n_ifo, n_pix)
        Quadrature delayed data at this sky direction.
    energy_threshold : float
        Network energy threshold E_thr.

    Returns
    -------
    dict with keys:
        total_energy — per-pixel total energy e_j, shape (n_pix,)
        mask         — active-pixel mask m_j ∈ {0, 1}, shape (n_pix,) int32
        Eo           — total masked energy / 2 (scalar)
        n_active     — number of active pixels (scalar, int32)
    """
    # e_j = Σ_i [v00²_ij + v90²_ij]
    energy = jnp.sum(v00 ** 2 + v90 ** 2, axis=0) + jnp.float32(1e-12)
    mask = (energy > energy_threshold).astype(jnp.int32)
    n_active = jnp.sum(mask)
    masked_energy = energy * mask
    Eo = jnp.sum(masked_energy) / jnp.float32(2.0)

    return {
        "total_energy": energy,
        "mask": mask,
        "Eo": Eo,
        "n_active": n_active,
    }


# ---------------------------------------------------------------------------
# 2. GW packet projection onto DPF
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=())
def project_gw_packet(v00: jnp.ndarray,
                      v90: jnp.ndarray,
                      f: jnp.ndarray,
                      F: jnp.ndarray,
                      fp: jnp.ndarray,
                      fx: jnp.ndarray,
                      network_index: jnp.ndarray,
                      total_energy: jnp.ndarray,
                      mask: jnp.ndarray,
                      REG: jnp.ndarray) -> dict:
    """Project data onto the DPF basis and build the regularised GW packet.

    This is the JAX equivalent of ``avx_GW_ps``.

    Parameters
    ----------
    v00, v90 : shape (n_ifo, n_pix) — delayed data
    f, F     : shape (n_pix, n_ifo) — DPF response vectors
    fp, fx   : shape (n_pix,) — |f+|², |fx|²
    network_index : shape (n_pix,)
    total_energy  : shape (n_pix,)
    mask     : shape (n_pix,) int32 — active-pixel mask
    REG      : shape (3,) — regularisation vector [δ·√2, DPF_reg, 0]

    Returns
    -------
    dict with keys:
        signal_00   — reconstructed signal (00), shape (n_ifo, n_pix)
        signal_90   — reconstructed signal (90), shape (n_ifo, n_pix)
        mask        — updated mask, shape (n_pix,) float32
        amplitude_plus_00, amplitude_plus_90   — plus amplitudes, (n_pix,)
        amplitude_cross_00, amplitude_cross_90 — cross amplitudes, (n_pix,)
        n_active    — number of active pixels (scalar)
    """
    EPS = jnp.float32(1e-5)
    reg_delta = REG[0]  # δ·√2
    reg_dpf = REG[1]    # DPF energy regulator

    # Inner products: x_p = Σ_i v00_ij · f_ji, etc.
    # v00: (n_ifo, n_pix), f: (n_pix, n_ifo) → need (n_pix,)
    xp = jnp.sum(v00 * f.T, axis=0)   # (n_pix,) — projections (data, f+)
    Xp = jnp.sum(v90 * f.T, axis=0)   # (n_pix,) — projections (data90, f+)
    xx = jnp.sum(v00 * F.T, axis=0)   # (n_pix,) — projections (data, fx)
    Xx = jnp.sum(v90 * F.T, axis=0)   # (n_pix,) — projections (data90, fx)

    # --- Plus-polarisation regularised inverse norm ---
    # α = mask / (fp + max(√(ni·(xp²+Xp²)/(e+ε))·REG[0] - fp, 0) + ε)
    plus_energy = xp ** 2 + Xp ** 2
    reg_plus = jnp.sqrt(network_index * plus_energy / (total_energy + EPS)) * reg_delta - fp
    reg_plus = jnp.maximum(reg_plus, jnp.float32(0.0))
    alpha = mask.astype(jnp.float32) / (fp + reg_plus + EPS)

    # --- Cross-polarisation regularised inverse norm ---
    # R = 0.1 + REG[1] / (e + ε)  → dynamic cross regulator
    # β = mask / (fx + max(√(xx²+Xx²/(h+ε))·R - fx, 0) + ε)
    h_plus = xp * alpha
    H_plus = Xp * alpha
    h_energy = h_plus ** 2 + H_plus ** 2
    cross_raw = xx ** 2 + Xx ** 2
    F_ratio = jnp.sqrt(cross_raw / (h_energy + EPS))
    R_dyn = jnp.float32(0.1) + reg_dpf / (total_energy + EPS)
    reg_cross = F_ratio * R_dyn - fx
    reg_cross = jnp.maximum(reg_cross, jnp.float32(0.0))
    beta = mask.astype(jnp.float32) / (fx + reg_cross + EPS)

    # Plus and cross amplitudes per pixel
    au = xp * alpha   # plus amplitude 00
    AU = Xp * alpha   # plus amplitude 90
    av = xx * beta    # cross amplitude 00
    AV = Xx * beta    # cross amplitude 90

    # Gaussian noise correction per pixel
    gnc = alpha * fp + beta * fx

    # Updated mask: gnc + mask - 1 (≥0 → accepted, <0 → rejected)
    mask_updated = gnc + mask.astype(jnp.float32) - jnp.float32(1.0)
    n_active = jnp.sum(mask.astype(jnp.int32))

    # Reconstruct signal packet: s_ij = f_ji·au_j + F_ji·av_j  (per ifo, per pixel)
    # f: (n_pix, n_ifo) → f.T: (n_ifo, n_pix)
    signal_00 = f.T * au[jnp.newaxis, :] + F.T * av[jnp.newaxis, :]
    signal_90 = f.T * AU[jnp.newaxis, :] + F.T * AV[jnp.newaxis, :]

    return {
        "signal_00": signal_00,
        "signal_90": signal_90,
        "mask": mask_updated,
        "amplitude_plus_00": au,
        "amplitude_plus_90": AU,
        "amplitude_cross_00": av,
        "amplitude_cross_90": AV,
        "n_active": n_active,
    }


# ---------------------------------------------------------------------------
# 3. Orthogonalise polarisation amplitudes
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=())
def orthogonalise_polarisations(v00: jnp.ndarray,
                                v90: jnp.ndarray,
                                mask: jnp.ndarray) -> dict:
    """Orthogonalise the 00/90 data vectors and compute polarisation energies.

    JAX equivalent of ``avx_ort_ps``.

    Parameters
    ----------
    v00, v90 : shape (n_ifo, n_pix)
    mask : shape (n_pix,) float32 — updated mask from ``project_gw_packet``

    Returns
    -------
    dict with keys:
        total_energy — scalar, total signal energy (e + E)
        psi_sin      — sin(ψ) per pixel, shape (n_pix,)
        psi_cos      — cos(ψ) per pixel, shape (n_pix,)
        energy_plus  — first (dominant) component energy ee, shape (n_pix,)
        energy_cross — second component energy EE, shape (n_pix,)
    """
    EPS = jnp.float32(1e-21)

    # Per-pixel norms and cross product (sum over ifo)
    aa = jnp.sum(v00 * v00, axis=0)   # (n_pix,)
    AA = jnp.sum(v90 * v90, axis=0)   # (n_pix,)
    aA = jnp.sum(v00 * v90, axis=0)   # (n_pix,)

    sin_2psi = 2.0 * aA
    cos_2psi = aa - AA
    total_e = aa + AA + EPS

    norm = jnp.sqrt(cos_2psi ** 2 + sin_2psi ** 2)
    energy_plus = (total_e + norm) / 2.0
    energy_cross = (total_e - norm) / 2.0

    cos_2psi_n = cos_2psi / (norm + EPS)
    sign = jnp.where(sin_2psi > 0, jnp.float32(1.0), jnp.float32(-1.0))
    psi_sin = jnp.sqrt((1.0 - cos_2psi_n) / 2.0)
    psi_cos = jnp.sqrt((1.0 + cos_2psi_n) / 2.0) * sign

    mk = jnp.where(mask > 0, jnp.float32(1.0), jnp.float32(0.0))
    e_total = jnp.sum(mk * energy_plus) + jnp.sum(mk * energy_cross)

    return {
        "total_energy": e_total,
        "psi_sin": psi_sin,
        "psi_cos": psi_cos,
        "energy_plus": energy_plus,
        "energy_cross": energy_cross,
    }


# ---------------------------------------------------------------------------
# 4. Coherent statistics
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=())
def compute_coherent_statistics(v00: jnp.ndarray,
                                v90: jnp.ndarray,
                                signal_00: jnp.ndarray,
                                signal_90: jnp.ndarray,
                                psi_sin: jnp.ndarray,
                                psi_cos: jnp.ndarray,
                                mask: jnp.ndarray) -> dict:
    """Compute coherent network statistics.

    JAX equivalent of ``avx_stat_ps``.

    Parameters
    ----------
    v00, v90       : shape (n_ifo, n_pix) — data
    signal_00, signal_90 : shape (n_ifo, n_pix) — reconstructed signal
    psi_sin, psi_cos : shape (n_pix,) — orthogonalisation angles
    mask : shape (n_pix,) float32 — pixel mask

    Returns
    -------
    dict with keys:
        correlation  — network correlation Cr (scalar)
        Ec           — total coherent energy (scalar)
        n_active     — number of active pixels
        null_energy  — total noise No (scalar)
        ec           — per-pixel coherent energy, (n_pix,)
        gn           — per-pixel G-noise correction, (n_pix,)
        rn           — per-pixel residual noise, (n_pix,)
        Mp           — polarisation statistic
    """
    EPS = jnp.float32(0.001)

    # Rotate data and signal by orthogonalisation angle
    # s' = s·cos + S·sin,  S' = S·cos - s·sin  (for both signal and data)
    cos_p = psi_cos[jnp.newaxis, :]
    sin_p = psi_sin[jnp.newaxis, :]

    s_rot = signal_00 * cos_p + signal_90 * sin_p
    S_rot = signal_90 * cos_p - signal_00 * sin_p
    x_rot = v00 * cos_p + v90 * sin_p
    X_rot = v90 * cos_p - v00 * sin_p

    # Per-pixel statistics (reduced over ifo)
    sx = jnp.sum(s_rot * x_rot, axis=0)   # (n_pix,)
    SX = jnp.sum(S_rot * X_rot, axis=0)
    c_sq = jnp.sum((s_rot * x_rot) ** 2, axis=0)
    C_sq = jnp.sum((S_rot * X_rot) ** 2, axis=0)
    ss = jnp.sum(s_rot * s_rot, axis=0)
    SS = jnp.sum(S_rot * S_rot, axis=0)
    residual_00 = jnp.sum((signal_00 - v00) ** 2, axis=0)
    residual_90 = jnp.sum((signal_90 - v90) ** 2, axis=0)

    mk = jnp.where(mask >= 0, jnp.float32(1.0), jnp.float32(0.0))

    # Incoherent energy ratios
    c_incoherent = c_sq / (sx ** 2 + EPS)
    C_incoherent = C_sq / (SX ** 2 + EPS)

    # Coherent energy
    signal_energy = mk * (ss + SS)
    ec_00 = ss * (1.0 - c_incoherent)
    ec_90 = SS * (1.0 - C_incoherent)
    ec = mk * (ec_00 + ec_90)

    # G-noise correction and residual noise
    gn = mk * 2.0 * mask
    rn = mk * (residual_00 + residual_90)

    # Correlation coefficient and reduced likelihood
    a_2ec = 2.0 * jnp.abs(ec)
    A_null = rn + gn + EPS
    cc = ec / (a_2ec + A_null)
    Lr = jnp.sum(signal_energy * cc)

    # Accumulated statistics — use ALL masked pixels (matching CPU avx_stat_ps)
    ec_mask = jnp.where(ec > EPS, jnp.float32(1.0), jnp.float32(0.0))
    Ec = jnp.sum(mk * ec)              # all masked pixels, incl. negative ec
    LL = jnp.sum(mk * signal_energy)   # all masked pixels
    GN = jnp.sum(mk * gn)             # all masked pixels
    RN = jnp.sum(mk * rn)             # all masked pixels
    No = (GN + RN) / jnp.float32(2.0) # (G-noise + residual) / 2, matching CPU total_noise
    NN = jnp.sum(ec_mask)             # count of pixels with positive ec

    # Correlation coefficient: CPU uses 2*Lr/LL
    Cr = jnp.float32(2.0) * Lr / (LL + EPS)
    Mp = NN  # polarisation stat = count of positive-ec pixels (matches CPU)

    return {
        "correlation": Cr,
        "Ec": Ec,
        "n_active": NN,
        "null_energy": No,
        "ec": ec,
        "gn": gn,
        "rn": rn,
        "Mp": Mp,
    }
