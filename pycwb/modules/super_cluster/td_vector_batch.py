"""
JAX-native batch time-delay vector extraction for super_cluster.

Replaces the serial per-pixel loop:
    for pixel in pixels:
        for ifo in range(n_ifo):
            td_vec = wdm.get_td_vec(tf_map, pixel_index=idx, K=K, mode="a")

with a single vmap'd JAX call per (layer, ifo) group:
    batch_get_td_vecs_jax(pixel_indices, padded00, padded90, T0, Tx, M, n_coeffs, K, J, L)

Notes
-----
- L=1 is assumed (set_td_filter called with L=1), so J = M.
- K < J must hold (always true in practice: TDSize << M).
- This replicates core.time_delay.get_pixel_amplitude + get_td_vec(mode="a")
  in JAX for vmap compatibility.
- Call ``prepare_td_inputs`` to extract numpy arrays from the TF map and
  filter bank before invoking the batch function.
"""

import logging
import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure JAX math (no @jax.jit — called inside outer jitted scope)
# ---------------------------------------------------------------------------

def _get_pixel_amplitude_jax(n, m, dT, padded_plane, T0, Tx, M, n_coeffs, J, quad):
    """
    JAX reimplementation of core.time_delay.get_pixel_amplitude.

    Parameters
    ----------
    n : traced int  — time-bin index of the pixel
    m : traced int  — frequency-layer index of the pixel
    dT : traced int — delay index in [-J, J]
    padded_plane : jax.Array, shape (n_time + 2*n_coeffs, M+1)
        Zero-padded TF plane (00-phase or 90-phase).
    T0 : jax.Array, shape (2*J+1, 2*n_coeffs+1)  — same-band TD filter table
    Tx : jax.Array, shape (2*J+1, 2*n_coeffs+1)  — cross-band TD filter table
    M, n_coeffs, J : static ints
    quad : bool (Python)  — False → 00-phase, True → 90-phase result

    Returns
    -------
    scalar float
    """
    is_odd_n = (n & 1).astype(jnp.bool_)
    is_odd_mn = ((m + n) & 1).astype(jnp.bool_)
    cancel_even = jnp.logical_xor(is_odd_n, jnp.asarray(quad, dtype=jnp.bool_))
    is_edge_layer = (m == 0) | (m == M)

    offsets = jnp.arange(2 * n_coeffs + 1)
    even_mask = (offsets % 2) == 0  # (2*n_coeffs+1,)

    dT_idx = dT + J
    t0_row = T0[dT_idx]   # (2*n_coeffs+1,)
    tx_row = Tx[dT_idx]   # (2*n_coeffs+1,)

    # L=1, so dt = dT / 1.0 = dT (as float)
    dt_val = dT.astype(jnp.float64)

    # ---- Same-band term (layer m) ----
    # padded_plane is zero-padded by n_coeffs on each side along axis-0.
    # For original time index n, the padded window starts at index n.
    col_same = padded_plane[:, m]   # shape (n_time + 2*n_coeffs,) — dynamic m
    window_same = jax.lax.dynamic_slice_in_dim(col_same, n, 2 * n_coeffs + 1)

    sum_even_same = jnp.sum(jnp.where(even_mask, window_same * t0_row, 0.0))
    sum_odd_same  = jnp.sum(jnp.where(~even_mask, window_same * t0_row, 0.0))

    # Edge parity: for edge layers, suppress one parity contribution
    sum_even_same = jnp.where(is_edge_layer & cancel_even,  0.0, sum_even_same)
    sum_odd_same  = jnp.where(is_edge_layer & ~cancel_even, 0.0, sum_odd_same)

    same_phase = m.astype(jnp.float64) * jnp.pi * dt_val / M
    result = jnp.where(
        is_odd_n,
        sum_even_same * jnp.cos(same_phase) - sum_odd_same * jnp.sin(same_phase),
        sum_even_same * jnp.cos(same_phase) + sum_odd_same * jnp.sin(same_phase),
    )

    # ---- Cross-band low term (layer m-1), active when m > 0 ----
    col_low = padded_plane[:, jnp.maximum(m - 1, 0)]  # clamp to avoid OOB
    window_low = jax.lax.dynamic_slice_in_dim(col_low, n, 2 * n_coeffs + 1)

    low_even = jnp.sum(jnp.where(even_mask, window_low * tx_row, 0.0))
    low_odd  = jnp.sum(jnp.where(~even_mask, window_low * tx_row, 0.0))

    # m==1: low band is layer 0, which is an edge
    low_even = jnp.where((m == 1) & cancel_even,  0.0, low_even)
    low_odd  = jnp.where((m == 1) & ~cancel_even, 0.0, low_odd)

    low_phase = (2.0 * m.astype(jnp.float64) - 1.0) * dt_val * jnp.pi / (2.0 * M)
    low_term = jnp.where(
        is_odd_n,
        (low_even - low_odd) * jnp.sin(low_phase),
        (low_even + low_odd) * jnp.sin(low_phase),
    )
    # sqrt(2) when low band or current band touches an edge (m==1 or m==M)
    low_term = low_term * jnp.where((m == 1) | (m == M), math.sqrt(2.0), 1.0)
    result = result + jnp.where(m > 0, jnp.where(is_odd_mn, -low_term, low_term), 0.0)

    # ---- Cross-band high term (layer m+1), active when m < M ----
    col_high = padded_plane[:, jnp.minimum(m + 1, M)]  # clamp to avoid OOB
    window_high = jax.lax.dynamic_slice_in_dim(col_high, n, 2 * n_coeffs + 1)

    high_even = jnp.sum(jnp.where(even_mask, window_high * tx_row, 0.0))
    high_odd  = jnp.sum(jnp.where(~even_mask, window_high * tx_row, 0.0))

    # m==M-1: high band is layer M, which is an edge
    high_even = jnp.where((m == M - 1) & cancel_even,  0.0, high_even)
    high_odd  = jnp.where((m == M - 1) & ~cancel_even, 0.0, high_odd)

    high_phase = (2.0 * m.astype(jnp.float64) + 1.0) * dt_val * jnp.pi / (2.0 * M)
    high_term = jnp.where(
        is_odd_n,
        (high_even + high_odd) * jnp.sin(high_phase),
        (high_even - high_odd) * jnp.sin(high_phase),
    )
    # sqrt(2) when high band or current band touches an edge (m==0 or m==M-1)
    high_term = high_term * jnp.where((m == 0) | (m == M - 1), math.sqrt(2.0), 1.0)
    result = result + jnp.where(m < M, jnp.where(is_odd_mn, -high_term, high_term), 0.0)

    # Zero out if edge+cancel condition
    return jnp.where(is_edge_layer & cancel_even, 0.0, result)


@partial(jax.jit, static_argnames=("M", "n_coeffs", "K", "J"))
def _get_td_vec_single_jax(pixel_index, padded00, padded90, T0, Tx, M, n_coeffs, K, J):
    """
    Compute the mode="a" TD vector for one pixel.

    Returns
    -------
    jax.Array, shape (4*K+2,)
        Concatenation of [a00(-K..K), a90(-K..K)].
    """
    M1 = M + 1
    n = pixel_index // M1
    m = pixel_index % M1
    delays = jnp.arange(-K, K + 1, dtype=jnp.int32)  # (2K+1,)

    # a00: 00-phase amplitudes over delays (uses padded00 plane)
    a00 = jax.vmap(
        lambda dT: _get_pixel_amplitude_jax(n, m, dT, padded00, T0, Tx, M, n_coeffs, J, quad=False)
    )(delays)

    # a90: 90-phase amplitudes over delays (uses padded90 plane)
    a90 = jax.vmap(
        lambda dT: _get_pixel_amplitude_jax(n, m, dT, padded90, T0, Tx, M, n_coeffs, J, quad=True)
    )(delays)

    return jnp.concatenate([a00, a90])  # (4K+2,)


@partial(jax.jit, static_argnames=("M", "n_coeffs", "K", "J"))
def batch_get_td_vecs_jax(pixel_indices, padded00, padded90, T0, Tx, M, n_coeffs, K, J):
    """
    Batch TD vector extraction using jax.vmap over pixel_indices.

    Parameters
    ----------
    pixel_indices : jax.Array, shape (n_pixels,)  — linear pixel indices
    padded00 : jax.Array, shape (n_time + 2*n_coeffs, M+1)
    padded90 : jax.Array, shape (n_time + 2*n_coeffs, M+1)
    T0 : jax.Array, shape (2*J+1, 2*n_coeffs+1)
    Tx : jax.Array, shape (2*J+1, 2*n_coeffs+1)
    M, n_coeffs, K, J : static ints

    Returns
    -------
    jax.Array, shape (n_pixels, 4*K+2)
    """
    return jax.vmap(
        lambda idx: _get_td_vec_single_jax(idx, padded00, padded90, T0, Tx, M, n_coeffs, K, J),
        in_axes=(0,),
    )(pixel_indices)


# ---------------------------------------------------------------------------
# Helpers: extract numpy arrays → call batch → write results back to pixels
# ---------------------------------------------------------------------------

def prepare_td_inputs(tf_map, wdm):
    """
    Extract JAX arrays from a TimeFrequencyMap and WDM instance for batch TD extraction.

    Parameters
    ----------
    tf_map : TimeFrequencyMap  (pycwb type, data is complex128 shape (M+1, n_time))
    wdm : WDMWavelet instance with td_filters already set via set_td_filter

    Returns
    -------
    dict with keys:
        padded00 : jnp.Array  shape (n_time + 2*n_coeffs, M+1)
        padded90 : jnp.Array  shape (n_time + 2*n_coeffs, M+1)
        T0       : jnp.Array  shape (2*J+1, 2*n_coeffs+1)
        Tx       : jnp.Array  shape (2*J+1, 2*n_coeffs+1)
        M        : int
        n_coeffs : int
        J        : int
    """
    data = np.asarray(tf_map.data, dtype=np.complex128)  # (M+1, n_time)
    # Transpose to (n_time, M+1) — matching time_delay.py convention
    tf00 = np.ascontiguousarray(data.real.T, dtype=np.float64)  # (n_time, M+1)
    tf90 = np.ascontiguousarray(data.imag.T, dtype=np.float64)

    td_filters = wdm.td_filters
    n_coeffs = int(td_filters.n_coeffs)
    M = int(td_filters.M)
    J = int(td_filters.max_delay)   # = M * L; for L=1, J=M

    pad = [(n_coeffs, n_coeffs), (0, 0)]
    padded00 = jnp.asarray(np.pad(tf00, pad), dtype=jnp.float64)
    padded90 = jnp.asarray(np.pad(tf90, pad), dtype=jnp.float64)

    T0 = jnp.asarray(td_filters.T0, dtype=jnp.float64)
    Tx = jnp.asarray(td_filters.Tx, dtype=jnp.float64)

    return {
        "padded00": padded00,
        "padded90": padded90,
        "T0": T0,
        "Tx": Tx,
        "M": M,
        "n_coeffs": n_coeffs,
        "J": J,
    }


def batch_extract_td_vecs(pixel_indices_np, tf_inputs, K):
    """
    Run the batch JAX TD extraction and return a numpy array.

    Parameters
    ----------
    pixel_indices_np : np.ndarray, shape (n_pixels,), dtype int32
    tf_inputs : dict from prepare_td_inputs
    K : int  — TD filter half-range (corresponds to config.TDSize)

    Returns
    -------
    np.ndarray, shape (n_pixels, 4*K+2), dtype float32
    """
    pixel_indices_jax = jnp.asarray(pixel_indices_np, dtype=jnp.int32)
    result = batch_get_td_vecs_jax(
        pixel_indices_jax,
        tf_inputs["padded00"],
        tf_inputs["padded90"],
        tf_inputs["T0"],
        tf_inputs["Tx"],
        tf_inputs["M"],
        tf_inputs["n_coeffs"],
        K,
        tf_inputs["J"],
    )
    result = jax.block_until_ready(result)
    return np.asarray(result, dtype=np.float32)
