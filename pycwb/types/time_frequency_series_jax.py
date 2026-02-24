"""
JAX-optimized version of time_frequency_series operations.

This module provides JAX implementations for computationally intensive
operations in TimeFrequencyMap, including:
- Element-wise maximum operations (vectorized)
- Gamma2Gauss transformation
- wdm_packet energy calculations

JAX allows for JIT compilation and GPU acceleration when available.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import lru_cache
from typing import Tuple, Optional, Union


@jit
def _time_slide_jax(data: jnp.ndarray, length: int, src_idx: int, dst_idx: int) -> jnp.ndarray:
    """
    JAX-compiled version of time slide copy operation.
    
    Copy a contiguous time slice with zero-padded out-of-range behavior.
    """
    out = jnp.zeros_like(data)
    if length > 0:
        # Compute actual copy length
        actual_length = jnp.minimum(
            length,
            jnp.minimum(data.shape[0] - dst_idx, data.shape[0] - src_idx)
        )
        # Create slice and copy
        indices = jnp.arange(actual_length)
        out = out.at[dst_idx + indices].set(data[src_idx + indices])
    return out


@jit
def _element_wise_max_complex(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Element-wise maximum for complex arrays based on magnitude."""
    mag_a = jnp.abs(a)
    mag_b = jnp.abs(b)
    return jnp.where(mag_a >= mag_b, a, b)


class JAXTimeFrequencyOps:
    """Helper class for JAX-accelerated TF operations."""
    
    @staticmethod
    @jit
    def gamma2gauss_jax(data: jnp.ndarray, edge: Optional[float] = None,
                       wavelet_rate: float = 1.0) -> Tuple[jnp.ndarray, float]:
        """
        JAX-compiled Gamma2Gauss transformation.
        
        Parameters
        ----------
        data : jnp.ndarray
            1D or 2D array of TF energy values
        edge : float, optional
            Edge parameter in seconds
        wavelet_rate : float
            Wavelet time resolution (1/dt)
            
        Returns
        -------
        transformed : jnp.ndarray
            Transformed energy values
        alp : float
            Scaling parameter (or 0.0 on failure)
        """
        original = jnp.asarray(data)
        original_shape = original.shape
        flat = jnp.real(original) if jnp.iscomplexobj(original) else original
        flat = flat.ravel()
        
        # Early returns for failure cases
        if flat.size < 4:
            return jnp.zeros_like(flat).reshape(original_shape), 0.0
        
        # Compute boundaries
        nL = int(float(edge or 0.0) * wavelet_rate)
        nn = int(flat.size)
        nL = max(0, min(nL, nn - 2))
        nR = nn - nL
        
        if nR <= nL + 1:
            return jnp.zeros_like(flat).reshape(original_shape), 0.0
        
        # Extract work region
        work = flat[nL:nR]
        med = float(jnp.median(work))
        
        if med <= 0:
            return jnp.zeros_like(flat).reshape(original_shape), 0.0
        
        # Calculate valid data
        mask = (work > 0.01) & (work < 20 * med)
        valid_data = work[mask]
        
        if valid_data.size == 0:
            return jnp.zeros_like(flat).reshape(original_shape), 0.0
        
        # Compute Gamma shape
        aaa = jnp.sum(valid_data)
        bbb = jnp.sum(jnp.log(valid_data))
        count = float(valid_data.size)
        
        alp = jnp.log(aaa / count) - bbb / count
        if alp <= 0:
            return jnp.zeros_like(flat).reshape(original_shape), 0.0
        
        alp = (3 - alp + jnp.sqrt((alp - 3) * (alp - 3) + 24 * alp)) / (12 * alp)
        alp_val = float(alp)
        
        # Transform
        avr = med * (3 * alp + 0.2) / (3 * alp - 0.8)
        ALP = med * alp / avr
        
        amp = flat * alp / avr
        transformed = jnp.where(amp < ALP, 0.0, amp - ALP * (1.0 + jnp.log(amp / ALP)))
        
        # Calculate RMS
        core = transformed[nL:nR]
        core = core[core > 1.0e-5]
        
        if core.size == 0:
            return jnp.zeros_like(flat).reshape(original_shape), 0.0
        
        q68 = float(jnp.quantile(core, 0.6827))
        if q68 <= 0:
            return jnp.zeros_like(flat).reshape(original_shape), 0.0
        
        rms = 1.0 / q68
        transformed = transformed * rms
        
        # Reshape back
        if len(original_shape) == 2:
            transformed = transformed.reshape(original_shape)
        
        return transformed, alp_val

    @staticmethod
    @jit
    def wdm_packet_energies_jax(
        complex_map: jnp.ndarray,
        pattern: int = 0,
        m_low: int = 0,
        m_high: int = 1000,
        jb: int = 0,
        je: int = -1
    ) -> jnp.ndarray:
        """
        JAX-compiled WDM packet energy calculation.
        
        Vectorized computation of packet energies for a given pattern.
        """
        PATTERNS = {
            0: [],
            1: [(0, 1), (0, -1)],
            2: [(1, 0), (-1, 0)],
            3: [(1, 1), (-1, -1)],
            4: [(1, -1), (-1, 1)],
            5: [(1, 1), (-1, -1), (2, 2), (-2, -2)],
            6: [(1, -1), (-1, 1), (2, -2), (-2, 2)],
            7: [(0, 1), (0, -1), (1, 0), (-1, 0)],
            8: [(1, 1), (1, -1), (-1, 1), (-1, -1)],
            9: [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        }
        
        pattern = abs(int(pattern))
        if pattern not in PATTERNS or pattern == 0:
            # For single pixel, return magnitude squared
            return jnp.abs(complex_map) ** 2
        
        M, T = complex_map.shape
        offsets = PATTERNS[pattern]
        
        # Determine mean and center_weight  
        if pattern in (1, 3, 4):
            mean = 3.0
        elif pattern == 2:
            mean = 3.0
        elif pattern in (5, 6):
            mean = 5.0
        elif pattern in (7, 8):
            mean = 5.0
        elif pattern == 9:
            mean = 9.0
        else:
            mean = 1.0
        
        center_weight = mean - 8.0
        
        # Initialize accumulators
        ee = jnp.zeros((M, T), dtype=jnp.float32)
        EE = jnp.zeros((M, T), dtype=jnp.float32)
        
        # Center contribution
        ee = ee + (jnp.abs(complex_map.real) ** 2) * center_weight
        EE = EE + (jnp.abs(complex_map.imag) ** 2) * center_weight
        
        # Add offset contributions
        for dr, dt in offsets:
            src_m0 = max(0, -dr)
            src_m1 = min(M, M - dr)
            src_t0 = max(0, -dt)
            src_t1 = min(T, T - dt)
            
            dst_m0 = src_m0 + dr
            dst_m1 = src_m1 + dr
            dst_t0 = src_t0 + dt
            dst_t1 = src_t1 + dt
            
            if src_m1 > src_m0 and src_t1 > src_t0:
                block = complex_map[src_m0:src_m1, src_t0:src_t1]
                ee = ee.at[dst_m0:dst_m1, dst_t0:dst_t1].add(jnp.real(block) ** 2)
                EE = EE.at[dst_m0:dst_m1, dst_t0:dst_t1].add(jnp.imag(block) ** 2)
        
        # Compute energy
        em = (ee + EE) / 2.0
        
        # Apply frequency bounds
        em_out = em.at[:m_low, :].set(0.0)
        em_out = em_out.at[m_high+1:, :].set(0.0)
        
        # Apply flattened bounds if provided
        if je > jb > 0:
            m_idx = jnp.arange(M).reshape(M, 1)
            t_idx = jnp.arange(T).reshape(1, T)
            j_flat = m_idx + t_idx * M
            invalid_mask = (j_flat < jb) | (j_flat >= je)
            em_out = jnp.where(invalid_mask, 0.0, em_out)
        
        return em_out


def create_jax_time_delay_max_energy(tf_map_instance):
    """
    Factory function to create a JAX-optimized time_delay_max_energy for a TimeFrequencyMap.
    
    This wraps the existing method but uses JAX kernels for the compute-intensive parts.
    """
    
    @jit
    def _compute_max_delay_jax(base_tf_real, base_tf_imag, shifted_real, shifted_imag):
        """JIT-compiled element-wise max of real/imag parts."""
        new_real = jnp.maximum(base_tf_real, shifted_real)
        new_imag = jnp.maximum(base_tf_imag, shifted_imag)
        return new_real, new_imag
    
    def time_delay_max_energy_jax(dt, downsample=1, pattern=0, hist=None):
        """JAX-optimized time_delay_max_energy."""
        # Fall back to NumPy for now, but future optimization can use JAX arrays internally
        # This serves as a placeholder for full JAX integration
        return tf_map_instance.time_delay_max_energy.__wrapped__(dt, downsample, pattern, hist)
    
    return time_delay_max_energy_jax
