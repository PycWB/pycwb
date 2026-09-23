"""Experimental six-block CUDA prefilter with native CPU JAX FFT arithmetic.

This module is deliberately disconnected from the production job processor.
The reduction order is specific to the validated CPU JAX/YNN implementation:
the CUDA :class:`wdm_prefilter.Prefilter` produces the block contractions and
the CPU JAX FFT in :func:`_finish` turns them into WDM coefficients.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .wdm_prefilter import Prefilter


@partial(jax.jit, static_argnames=("m",))
def _finish(eq23: jax.Array, m: int) -> jax.Array:
    """Turn ``(n_time, 2 * m)`` prefilter rows into ``(m + 1, n_time)`` complex WDM coefficients."""
    sqrt2 = jnp.sqrt(2.0)
    spectrum = jnp.fft.rfft(eq23, n=2 * m, axis=-1)
    real, imag = spectrum.real, spectrum.imag
    endpoint0, endpointm = real[:, 0] / sqrt2, real[:, m] / sqrt2
    real = real.at[:, 0].set(endpoint0).at[:, m].set(endpointm)
    imag = imag.at[:, 0].set(endpoint0).at[:, m].set(endpointm)
    parity = ((jnp.arange(m + 1)[None, :] + (jnp.arange(len(eq23))[:, None] & 1)) & 1) == 1
    primary = jnp.where(parity, sqrt2 * imag, sqrt2 * real)
    quadrature = jnp.where(parity, sqrt2 * real, -sqrt2 * imag)
    return (primary + 1j * quadrature).T


class HybridTransform:
    """Bounded quadrature WDM transform: CUDA prefilter plus CPU JAX FFT.

    Drop-in for ``time_delay_jax._t2w_data_jax`` inside
    :class:`max_energy_hybrid.HybridMaximum`.

    Attributes
    ----------
    prefilter : Prefilter
        CUDA block contraction stage.
    """

    def __init__(self) -> None:
        self.prefilter = Prefilter()

    def __call__(
        self,
        wdm_M: int,
        wdm_m_H: int,
        wavelet_filter: np.ndarray,
        ts_data: np.ndarray,
        mm_mode: int,
        bounded: bool = False,
    ) -> np.ndarray:
        """Transform one time series into WDM coefficients.

        Parameters
        ----------
        wdm_M : int
            Number of layers ``m``; a power of two in ``[8, 1024]``.
        wdm_m_H : int
            Filter half-length; must equal ``12 * m + 1`` (six blocks).
        wavelet_filter : array-like
            Wavelet filter taps; the first ``wdm_m_H`` are used.
        ts_data : array-like
            Float64 time series whose length is a multiple of ``m`` and at
            least ``wdm_m_H``.
        mm_mode : int
            Must be ``-1`` (quadrature output).
        bounded : bool, optional
            Must be ``True``; only the bounded native path is reproduced.

        Returns
        -------
        numpy.ndarray
            Complex128 coefficients of shape ``(m + 1, n_time)`` with
            ``n_time = len(ts_data) // m``, in Fortran (time-major) memory
            order so :class:`packet_energy.PacketEnergy` can upload rows
            without copying.

        Raises
        ------
        ValueError
            If the layer count or filter length is not the native six-block
            configuration, JAX x64 is disabled, the input is not a vector, or
            the bounded/quadrature/alignment requirements are not met.
        """
        m, nh = int(wdm_M), int(wdm_m_H)
        ts = np.asarray(ts_data, dtype=np.float64)
        if m < 8 or m > 1024 or m & (m - 1) or nh != 12 * m + 1:
            raise ValueError("Experimental hybrid requires native six-block LF filters")
        if ts.ndim != 1 or not jax.config.x64_enabled:
            raise ValueError("Experimental hybrid requires a vector and JAX FP64")
        if not bounded or mm_mode != -1 or len(ts) % m or nh > len(ts):
            raise ValueError("Experimental hybrid requires bounded aligned quadrature input")
        nt = len(ts) // m
        tile = max(32, (262144 // m) // 32 * 32)
        tile = min(tile, nt)
        # Match native bounded_core's fallback exactly: an unaligned time-bin
        # count uses one full FFT batch, not padded tiles with another shape.
        if nt % 32:
            tile = nt
        padded_nt = ((nt + tile - 1) // tile) * tile
        extended = np.zeros(padded_nt * m + 2 * nh, np.float64)
        extended[nh : nh + len(ts)] = ts
        extended[:nh] = ts[1 : nh + 1][::-1]
        extended[nh + len(ts) : len(ts) + 2 * nh] = ts[-nh:][::-1]
        eq23 = self.prefilter(extended, np.asarray(wavelet_filter)[:nh], m, m, padded_nt, 6)
        # PacketEnergy consumes time-major rows. Keep the native transposed
        # layout to avoid copying the complete coefficient map each delay.
        result = np.empty((m + 1, padded_nt), np.complex128, order="F")
        with jax.default_device(jax.devices("cpu")[0]):
            for start in range(0, padded_nt, tile):
                result[:, start : start + tile] = np.asarray(_finish(eq23[start : start + tile], m))
        return result[:, :nt]
