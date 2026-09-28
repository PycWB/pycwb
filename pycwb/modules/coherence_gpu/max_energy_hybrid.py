"""Experimental native CPU WDM/FFT plus ordered CUDA packet-energy maximum.

:class:`HybridMaximum` replaces ``time_delay_jax._time_delay_max_energy_pattern_jit``
for quadrature packet patterns: every delay shift is transformed on the CPU
(native JAX WDM, or :class:`wdm_hybrid.HybridTransform`) and its packet energy
is folded into a device-resident maximum by :class:`packet_energy.PacketEnergy`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import jax
import numpy as np

from pycwb.modules.coherence_native import projection, time_delay_jax

from pycwb.utils.function_binding import specialize
from pycwb.modules.coherence_gpu.packet_energy import PacketEnergy

logger = logging.getLogger(__name__)
transform = jax.jit(time_delay_jax._t2w_data_jax, static_argnums=(0, 1, 4, 5))
"""Native WDM transform jitted once; the default ``transform_impl``."""


class HybridMaximum:
    """Max-energy core with CPU transforms and a CUDA running maximum.

    Parameters
    ----------
    transform_impl : callable, optional
        ``(wdm_M, wdm_m_H, wavelet_filter, ts_data, mm_mode, bounded) -> coefficients``
        returning complex128 ``(m + 1, n_time)``. Defaults to the jitted
        native :data:`transform`.

    Attributes
    ----------
    packet : PacketEnergy
        Device-resident maximum, reused across calls of the same map shape.
    transform : callable
        The transform in use.
    """

    def __init__(self, transform_impl: Callable[..., Any] = transform) -> None:
        self.packet = PacketEnergy()
        self.transform = transform_impl

    def __call__(
        self,
        ts_data: np.ndarray,
        sample_rate: float,
        t0: float,
        downsample: int,
        max_delay: int,
        wavelet_filter: np.ndarray,
        mm_mode: int,
        pattern: int,
        edge: float,
        wavelet_rate: float,
        f_low: float,
        f_high: float,
        df: float,
        coeff_shape: tuple[int, int],
        wdm_M: int,
        wdm_m_H: int,
        bounded: bool = False,
    ) -> jax.Array:
        """Return the maximum packet energy over all delay shifts.

        The signature mirrors the native ``_time_delay_max_energy_pattern_jit``
        hook; ``sample_rate`` and ``t0`` are accepted for that reason only.

        Parameters
        ----------
        ts_data : array-like
            Float64 whitened time series.
        sample_rate, t0 : float
            Unused; part of the native hook signature.
        downsample : int
            Delay step in samples, at least 1.
        max_delay : int
            Largest delay in samples, clipped to ``len(ts_data) - 1``.
        wavelet_filter, mm_mode, wdm_M, wdm_m_H, bounded
            Forwarded to the transform; ``mm_mode`` must be ``-1``.
        pattern : int
            Native packet pattern, non-zero.
        edge, wavelet_rate, f_low, f_high, df
            Forwarded to :class:`packet_energy.PacketEnergy`.
        coeff_shape : tuple[int, int]
            Expected ``(n_frequency, n_time)`` of every transform result.

        Returns
        -------
        jax.Array
            Float64 ``(n_frequency, n_time)`` maximum map on the CPU device,
            with the first row (and second for patterns 5, 6, 9) zeroed like
            the native core.

        Raises
        ------
        ValueError
            If JAX x64 is disabled, ``mm_mode`` is not ``-1``, ``pattern`` is
            0, ``downsample`` is not positive, or a transform result does not
            have ``coeff_shape``.
        """
        if not jax.config.x64_enabled or mm_mode != -1 or pattern == 0:
            raise ValueError("Hybrid maximum requires FP64 quadrature packet mode")
        cpu = jax.devices("cpu")[0]
        ts = np.asarray(ts_data, dtype=np.float64)
        xx = ts.copy()
        step, limit = int(downsample), min(int(max_delay), len(ts) - 1)
        if step < 1:
            raise ValueError("Delay step must be positive")
        start = time.perf_counter()
        transform_s = 0.0
        calls = 0

        def consume(data: np.ndarray) -> None:
            nonlocal transform_s, calls
            t = time.perf_counter()
            with jax.default_device(cpu):
                coefficients = np.asarray(
                    self.transform(
                        wdm_M, wdm_m_H, wavelet_filter, data, mm_mode, bounded
                    )
                )
            transform_s += time.perf_counter() - t
            if coefficients.shape != coeff_shape:
                raise ValueError("Hybrid transform shape differs from native input")
            self.packet(
                coefficients,
                pattern,
                edge,
                wavelet_rate,
                f_low,
                f_high,
                df,
                accumulate=calls > 0,
            )
            calls += 1

        consume(ts)
        for k in range(step, limit + 1, step):
            # Preserve native cpf's untouched boundary from the prior shift.
            xx[:-k] = ts[k:]
            consume(xx)
            xx[k:] = ts[:-k]
            consume(xx)
        result = self.packet.result()
        result[0, :] = 0.0
        if pattern in (5, 6, 9) and result.shape[0] > 2:
            result[1, :] = 0.0
        logger.info(
            "GPU hybrid maximum: calls=%d transforms=%.6f total=%.6f",
            calls,
            transform_s,
            time.perf_counter() - start,
        )
        return jax.device_put(result, cpu)


def make_projection(cuda_prefilter: bool = False) -> Callable[..., Any]:
    """Return a ``projection.max_energy`` clone whose pattern core is :class:`HybridMaximum`.

    Parameters
    ----------
    cuda_prefilter : bool, optional
        Use :class:`wdm_hybrid.HybridTransform` instead of the native jitted
        transform.

    Returns
    -------
    callable
        Specialized ``max_energy`` bound to a specialized
        ``time_delay_max_energy``; the production modules are untouched.
    """
    if cuda_prefilter:
        from pycwb.modules.coherence_gpu.wdm_hybrid import HybridTransform

        maximum = HybridMaximum(HybridTransform())
    else:
        maximum = HybridMaximum()
    td = specialize(
        time_delay_jax.time_delay_max_energy, _time_delay_max_energy_pattern_jit=maximum
    )
    return specialize(projection.max_energy, time_delay_max_energy=td)
