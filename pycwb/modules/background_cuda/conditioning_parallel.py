"""Parallel native detector conditioning for the opt-in GPU job.

Runs in the job (parent) process before lag workers exist. Each detector is
conditioned by the unchanged native ``condition_strain`` on the CPU
JAX device; only the per-detector scheduling changes.
"""

from __future__ import annotations

import copy
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import jax

from pycwb.modules.data_conditioning.data_conditioning import (
    condition_strains as serial,
)
from pycwb.modules.data_conditioning.data_conditioning import (
    condition_strain,
)

from pycwb.constants.gpu_options import gpu_options

if TYPE_CHECKING:
    from pycwb.config import Config

logger = logging.getLogger(__name__)

MAX_CONDITION_WORKERS = 3
"""Largest ``gpu.condition_workers``; one thread per detector of a three-detector network."""


def condition_strains(config: Config, strains: list[Any]) -> tuple[Any, ...]:
    """Condition every detector strain, in parallel threads when requested.

    Switches, read at call time:

    * ``gpu.condition_workers`` (default 1, at most
      ``MAX_CONDITION_WORKERS``): with 1 the native serial function is called
      unchanged; otherwise one thread per detector (bounded by the count).
    * ``gpu.validate_conditioning=true``: also run the native serial path
      on a deep copy of the inputs and require bitwise identical output.

    Parameters
    ----------
    config : Config
        Search configuration.
    strains : list
        One input strain per detector, as accepted by the native functions.

    Returns
    -------
    tuple
        Same layout as the native ``data_conditioning`` return, i.e. one tuple
        per product with a per-detector entry each.

    Raises
    ------
    ValueError
        If ``gpu.condition_workers`` is not an integer in
        ``[1, MAX_CONDITION_WORKERS]``.
    AssertionError
        If validation is enabled and the parallel result differs from serial.
    """
    workers = gpu_options(config).condition_workers
    if workers == 1:
        return serial(config, strains)
    validate = gpu_options(config).validate_conditioning
    reference_inputs = copy.deepcopy(strains) if validate else None
    device = jax.devices("cpu")[0]

    def condition(strain: Any) -> Any:
        with jax.default_device(device):
            return condition_strain(config, strain)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(workers, len(strains))) as pool:
        results = list(pool.map(condition, strains))
    actual = tuple(zip(*results))
    logger.info(
        "GPU job parallel conditioning: detectors=%d seconds=%.6f",
        len(strains),
        time.perf_counter() - start,
    )
    if validate:
        from .validation import leaves

        expected = serial(config, reference_inputs)
        if leaves(expected) != leaves(actual):
            raise AssertionError(
                "Conditioned strains/noise maps differ from native serial"
            )
        logger.info(
            "GPU conditioning parity: all detector data and coordinates bitwise exact"
        )
    return actual
