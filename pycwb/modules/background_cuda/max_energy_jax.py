"""Exploratory FP64 GPU max-energy core, keeping native CPU inverse and scaling.

GPU FFT arithmetic may differ from CPU. This stage is not an accepted backend
until downstream membership, thresholds and persisted-output gates establish it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import jax
import numpy as np

from pycwb.modules.coherence_native import projection, time_delay_jax

from . import flags
from .binding import specialize

logger = logging.getLogger(__name__)


def make_projection() -> Callable[..., Any]:
    """Return a ``projection.max_energy`` clone whose jitted cores run on the first GPU.

    Both ``_time_delay_max_energy_pattern_jit`` and
    ``_time_delay_max_energy_complex_jit`` are wrapped so their array
    arguments are placed on the GPU and the call runs under that default
    device. With ``PYCWB_GPU_COMPARE_MAX_ENERGY=1`` (read once here) every
    call is repeated on the CPU and the number of differing 64-bit words and
    the largest absolute difference are logged.

    Returns
    -------
    callable
        Specialized ``max_energy``; the production modules are untouched.

    Raises
    ------
    RuntimeError
        If JAX x64 is not enabled.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError("GPU max-energy exploration requires FP64 JAX")
    device = jax.devices("gpu")[0]
    compare = flags.enabled("COMPARE_MAX_ENERGY")

    def on_gpu(kernel: Callable[..., Any]) -> Callable[..., Any]:
        def run(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            with jax.default_device(device):
                gpu_args = tuple(
                    jax.device_put(x, device) if isinstance(x, (jax.Array, np.ndarray)) else x for x in args
                )
                result = kernel(*gpu_args, **kwargs)
                result.block_until_ready()
            gpu_s = time.perf_counter() - start
            if compare:
                start = time.perf_counter()
                with jax.default_device(jax.devices("cpu")[0]):
                    expected = np.asarray(kernel(*args, **kwargs))
                cpu_s = time.perf_counter() - start
                actual = np.asarray(result)
                different = int(np.count_nonzero(expected.view("u8") != actual.view("u8")))
                max_abs = float(np.max(np.abs(expected - actual)))
                logger.info(
                    "GPU max-energy comparison: shape=%s gpu=%.6fs cpu=%.6fs differing_words=%d max_abs=%.17g",
                    actual.shape,
                    gpu_s,
                    cpu_s,
                    different,
                    max_abs,
                )
            else:
                logger.info("GPU max-energy core: shape=%s gpu=%.6fs", result.shape, gpu_s)
            return result

        return run

    td = specialize(
        time_delay_jax.time_delay_max_energy,
        _time_delay_max_energy_pattern_jit=on_gpu(time_delay_jax._time_delay_max_energy_pattern_jit),
        _time_delay_max_energy_complex_jit=on_gpu(time_delay_jax._time_delay_max_energy_complex_jit),
    )
    return specialize(projection.max_energy, time_delay_max_energy=td)
