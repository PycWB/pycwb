"""
Packet operations and utility kernels — JAX implementation.

These functions operate at a single sky direction (the best-fit l_max) to
compute detection-level statistics: packet rotation, amplitude normalisation,
noise correction, and xtalk-convolved energy sums.

JAX equivalents of the functions in ``likelihoodWP/utils.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


# ---------------------------------------------------------------------------
# Packet rotation and normalisation (avx_packet_ps equivalent)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=())
def compute_packet_rotation(v00: jnp.ndarray,
                            v90: jnp.ndarray,
                            mask: jnp.ndarray) -> dict:
    """Rotate data into principal-axis frame and compute per-IFO amplitudes.

    JAX equivalent of ``avx_packet_ps``.

    Parameters
    ----------
    v00, v90 : shape (n_ifo, n_pix)
    mask : shape (n_pix,) — pixel mask (>0 active)

    Returns
    -------
    dict with keys:
        Ep       — total packet energy / 2 (scalar)
        v00_rot  — rotated + normalised v00, shape (n_ifo, n_pix)
        v90_rot  — rotated + normalised v90, shape (n_ifo, n_pix)
        energy   — per-IFO packet energy, shape (n_ifo,)
        sin_rot  — per-IFO rotation sin, shape (n_ifo,)
        cos_rot  — per-IFO rotation cos, shape (n_ifo,)
        amp_a    — per-IFO first component amplitude, shape (n_ifo,)
        amp_A    — per-IFO second component amplitude, shape (n_ifo,)
    """
    EPS = jnp.float32(1e-4)

    mk = jnp.where(mask > 0, jnp.float32(1.0), jnp.float32(0.0))  # (n_pix,)
    mk_2d = mk[jnp.newaxis, :]  # (1, n_pix)

    # Per-IFO sums over masked pixels
    aa = jnp.sum(mk_2d * v00 * v00, axis=1)  # (n_ifo,)
    AA = jnp.sum(mk_2d * v90 * v90, axis=1)  # (n_ifo,)
    aA = jnp.sum(mk_2d * v00 * v90, axis=1)  # (n_ifo,)

    sin_2p = 2.0 * aA
    cos_2p = aa - AA
    total_e = aa + AA + EPS
    norm_2p = jnp.sqrt(cos_2p ** 2 + sin_2p ** 2)

    amp_a = jnp.sqrt((total_e + norm_2p) / 2.0)  # first component amplitude
    amp_A = jnp.sqrt(jnp.abs((total_e - norm_2p) / 2.0))  # second component

    cos_2p_n = cos_2p / (norm_2p + EPS)
    sign = jnp.where(sin_2p > 0, jnp.float32(1.0), jnp.float32(-1.0))
    sin_rot = jnp.sqrt((1.0 - cos_2p_n) / 2.0)
    cos_rot = jnp.sqrt((1.0 + cos_2p_n) / 2.0) * sign

    energy = (amp_a + amp_A) ** 2 / 2.0
    Ep = jnp.sum(energy) / 2.0

    # Inverse amplitudes for normalisation
    inv_a = 1.0 / (amp_a + EPS)  # (n_ifo,)
    inv_A = 1.0 / (amp_A + EPS)

    # Rotate and normalise
    # _a = v00·cos + v90·sin,  _A = v90·cos - v00·sin
    v00_rot_raw = v00 * cos_rot[:, jnp.newaxis] + v90 * sin_rot[:, jnp.newaxis]
    v90_rot_raw = v90 * cos_rot[:, jnp.newaxis] - v00 * sin_rot[:, jnp.newaxis]
    v00_rot = mk_2d * v00_rot_raw * inv_a[:, jnp.newaxis]
    v90_rot = mk_2d * v90_rot_raw * inv_A[:, jnp.newaxis]

    return {
        "Ep": Ep,
        "v00_rot": v00_rot,
        "v90_rot": v90_rot,
        "energy": energy,
        "sin_rot": sin_rot,
        "cos_rot": cos_rot,
        "amp_a": amp_a,
        "amp_A": amp_A,
    }


# ---------------------------------------------------------------------------
# Gaussian noise correction (avx_noise_ps equivalent)
# ---------------------------------------------------------------------------

def compute_noise_correction(signal_norm: np.ndarray,
                             data_norm: np.ndarray,
                             total_energy: np.ndarray,
                             mask: np.ndarray,
                             coherent_energy: np.ndarray,
                             gn: np.ndarray,
                             rn: np.ndarray) -> tuple:
    """Compute Gaussian noise correction and energy decomposition.

    Numpy implementation (called once at best sky only, not in hot path).

    Parameters
    ----------
    signal_norm : (n_ifo, n_pix) — normalised signal packet norms
    data_norm   : (n_ifo, n_pix) — normalised data packet norms
    total_energy : (n_pix,)
    mask : (n_pix,) — pixel mask
    coherent_energy : (n_pix,)
    gn : (n_pix,) — G-noise correction
    rn : (n_pix,) — residual noise

    Returns
    -------
    tuple (Gn, Ec, Dc, Rc, Eh, Es, NC, NS)
    """
    p_arr = np.asarray(signal_norm, dtype=np.float64)
    q_arr = np.asarray(data_norm, dtype=np.float64)
    n_ifos = p_arr.shape[0]

    ns = p_arr.sum(axis=0) / n_ifos
    nx = q_arr.sum(axis=0) / n_ifos

    mk = (np.asarray(mask) > 0).astype(np.float64)
    nm_core = mk * (nx > 0).astype(np.float64)

    EC = float(np.sum(nm_core * coherent_energy))
    NC = float(np.sum(nm_core))

    nm_halo = mk - nm_core
    ES = float(np.sum(nm_halo * rn))

    gn_arr = np.asarray(gn, dtype=np.float64)
    rc = np.where(gn_arr < 2.0, 1.0, 0.0)
    nn = gn_arr * (1.0 - rc)
    rc = rc + nn * 0.5
    rc = np.asarray(coherent_energy, dtype=np.float64) / (rc + 1e-9)

    nm_sig = mk * (ns > 0).astype(np.float64)
    gn_scaled = mk * gn_arr * nx

    SC = float(np.sum(nm_sig * coherent_energy))
    RC = float(np.sum(nm_sig * rc))
    GN = float(np.sum(nm_sig * gn_scaled))
    NS = float(np.sum(nm_sig))

    nm_sat = mk - nm_sig
    EH = float(np.sum(nm_sat * total_energy))

    return GN, EC, SC - EC, RC / (SC + 0.01), EH / 2.0, ES / 2.0, NC, NS


# ---------------------------------------------------------------------------
# Set packet amplitudes (avx_setAMP_ps equivalent)
# ---------------------------------------------------------------------------

def set_packet_amplitudes(p: np.ndarray,
                          q: np.ndarray,
                          q_norm: np.ndarray,
                          q_sin: np.ndarray,
                          q_cos: np.ndarray,
                          q_a: np.ndarray,
                          q_A: np.ndarray,
                          mask: np.ndarray) -> tuple:
    """Set packet amplitudes for waveform reconstruction.

    Returns (N_eff, new_p, new_q) where N_eff is the effective pixel count.
    """
    p_arr = np.asarray(p)
    q_arr = np.asarray(q)
    qn_arr = np.asarray(q_norm)
    n_ifo = p_arr.shape[0]

    aA = np.asarray(q_a) + np.asarray(q_A)
    mk = 0.5 * (np.asarray(mask) > 0).astype(p_arr.dtype)

    n_arr = aA[:, np.newaxis] * mk[np.newaxis, :] * qn_arr

    new_p = n_arr * (p_arr * np.asarray(q_cos)[:, np.newaxis] -
                     q_arr * np.asarray(q_sin)[:, np.newaxis])
    new_q = n_arr * (q_arr * np.asarray(q_cos)[:, np.newaxis] +
                     p_arr * np.asarray(q_sin)[:, np.newaxis])

    N_eff = float(np.sum(qn_arr.sum(axis=0) * mk)) * 4.0 / n_ifo
    return N_eff, new_p, new_q


# ---------------------------------------------------------------------------
# Null packet (avx_loadNULL_ps equivalent)
# ---------------------------------------------------------------------------

def compute_null_packet(data_00: np.ndarray,
                        data_90: np.ndarray,
                        signal_00: np.ndarray,
                        signal_90: np.ndarray) -> tuple:
    """Compute null = data − signal per detector and pixel.

    Returns (null_00, null_90) each shape (n_ifo, n_pix).
    """
    return (np.asarray(data_00, dtype=np.float32) - np.asarray(signal_00, dtype=np.float32),
            np.asarray(data_90, dtype=np.float32) - np.asarray(signal_90, dtype=np.float32))


# ---------------------------------------------------------------------------
# Polarisation projection (avx_pol_ps equivalent)
# ---------------------------------------------------------------------------

def project_polarisation(p: np.ndarray,
                         q: np.ndarray,
                         mask: np.ndarray,
                         fp: np.ndarray,
                         fx: np.ndarray,
                         f: np.ndarray,
                         F: np.ndarray) -> tuple:
    """Project data onto the network polarisation plane (PnP + DSP).

    Returns (new_p, new_q, (radius_00, angle_00), (radius_90, angle_90)).
    """
    EPS = 1e-5
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    f_arr = np.asarray(f, dtype=np.float64)
    F_arr = np.asarray(F, dtype=np.float64)
    n_ifo = p_arr.shape[0]
    n_pix = p_arr.shape[1]

    mk = (np.asarray(mask) > 0).astype(np.float64)
    mk_p = p_arr * mk[np.newaxis, :]
    mk_q = q_arr * mk[np.newaxis, :]

    # Dot products
    xp = (f_arr * mk_p.T).sum(axis=1)
    XP = (f_arr * mk_q.T).sum(axis=1)
    xx = (F_arr * mk_p.T).sum(axis=1)
    XX = (F_arr * mk_q.T).sum(axis=1)

    fp_arr = np.asarray(fp, dtype=np.float64)
    fx_arr = np.asarray(fx, dtype=np.float64)
    sqrt_fp = np.sqrt(fp_arr) + EPS
    sqrt_fx = np.sqrt(fx_arr) + EPS

    cpol = xp / sqrt_fp
    spol = xx / sqrt_fx
    CPOL = XP / sqrt_fp
    SPOL = XX / sqrt_fx

    rpol = xp * xp / (fp_arr + EPS) + xx * xx / (fx_arr + EPS)
    RPOL = XP * XP / (fp_arr + EPS) + XX * XX / (fx_arr + EPS)

    r = np.sqrt(rpol).astype(np.float32)
    a = np.arctan2(spol, cpol).astype(np.float32)
    R = np.sqrt(RPOL).astype(np.float32)
    A = np.arctan2(SPOL, CPOL).astype(np.float32)

    # PnP and DSP (preserve last-pixel-only behaviour for compatibility)
    new_p = np.empty((n_ifo, n_pix), dtype=np.float32)
    new_q = np.empty((n_ifo, n_pix), dtype=np.float32)

    i = n_pix - 1
    cpol_s = cpol[i] / sqrt_fp[i]
    spol_s = spol[i] / sqrt_fx[i]
    CPOL_s = CPOL[i] / sqrt_fp[i]
    SPOL_s = SPOL[i] / sqrt_fx[i]

    for j in range(n_ifo):
        new_p[j][i] = f_arr[i][j] * cpol_s + F_arr[i][j] * spol_s
        new_q[j][i] = f_arr[i][j] * CPOL_s + F_arr[i][j] * SPOL_s

    Nval = np.sqrt(cpol_s ** 2 + CPOL_s ** 2)
    cpol_s /= (Nval + EPS)
    CPOL_s /= (Nval + EPS)

    for j in range(n_ifo):
        new_p[j][i] = new_p[j][i] * cpol_s + new_q[j][i] * CPOL_s
        new_q[j][i] = new_q[j][i] * cpol_s - new_p[j][i] * CPOL_s

    return new_p, new_q, (r, a), (R, A)


# ---------------------------------------------------------------------------
# XTalk energy sums (xtalk_energy_sum_numpy equivalent)
# ---------------------------------------------------------------------------

def xtalk_energy_sum(p: np.ndarray,
                     q: np.ndarray,
                     xtalks: np.ndarray,
                     xtalks_lookup: np.ndarray,
                     mask: np.ndarray) -> float:
    """Compute raw xtalk-convolved energy sum.

    This is the I<0 branch of C++ _avx_norm_ps — no clamping, no SNR ratio.
    Used for Em (data energy) and Np (null energy).
    """
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    mk_arr = np.asarray(mask)
    n_ifos = p_arr.shape[0]

    g = np.zeros(n_ifos, dtype=np.float64)

    for i in range(p_arr.shape[1]):
        if mk_arr[i] <= 0.0:
            continue
        r0, r1 = xtalks_lookup[i]
        xt = xtalks[r0:r1]
        idx = xt[:, 0].astype(np.int32)
        cc = xt[:, 4:8].T.astype(np.float64)

        p_nbr = p_arr[:, idx]
        q_nbr = q_arr[:, idx]

        x = np.vstack((p_nbr @ cc[0],
                        p_nbr @ cc[1],
                        q_nbr @ cc[2],
                        q_nbr @ cc[3]))

        pi = p_arr[:, i]
        qi = q_arr[:, i]
        t = x[0] * pi + x[1] * qi + x[2] * pi + x[3] * qi

        for j in range(n_ifos):
            if t[j] > 0.0:
                g[j] += t[j]

    return float(np.sum(g))
