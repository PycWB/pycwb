"""Parallel native detector conditioning for the opt-in GPU job.

Runs in the job (parent) process before lag workers exist. Each detector is
conditioned by the unchanged native ``data_conditioning_single`` on the CPU
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
    data_conditioning as serial,
)
from pycwb.modules.data_conditioning.data_conditioning import (
    data_conditioning_single,
)

from . import flags

if TYPE_CHECKING:
    from pycwb.config import Config

logger = logging.getLogger(__name__)

MAX_CONDITION_WORKERS = 3
"""Largest ``PYCWB_GPU_CONDITION_WORKERS``; one thread per detector of a three-detector network."""


def data_conditioning(config: Config, strains: list[Any], nproc: int = 1) -> tuple[Any, ...]:
    """Condition every detector strain, in parallel threads when requested.

    Switches, read at call time:

    * ``PYCWB_GPU_CONDITION_WORKERS`` (default 1, at most
      ``MAX_CONDITION_WORKERS``): with 1 the native serial function is called
      unchanged; otherwise one thread per detector (bounded by the count).
    * ``PYCWB_GPU_VALIDATE_CONDITIONING=1``: also run the native serial path
      on a deep copy of the inputs and require bitwise identical output.

    Parameters
    ----------
    config : Config
        Search configuration.
    strains : list
        One input strain per detector, as accepted by the native functions.
    nproc : int, optional
        Forwarded to the native serial function only.

    Returns
    -------
    tuple
        Same layout as the native ``data_conditioning`` return, i.e. one tuple
        per product with a per-detector entry each.

    Raises
    ------
    ValueError
        If ``PYCWB_GPU_CONDITION_WORKERS`` is not an integer in
        ``[1, MAX_CONDITION_WORKERS]``.
    AssertionError
        If validation is enabled and the parallel result differs from serial.
    """
    workers = flags.worker_count("CONDITION_WORKERS", maximum=MAX_CONDITION_WORKERS)
    if workers == 1:
        return serial(config, strains, nproc=nproc)
    validate = flags.enabled("VALIDATE_CONDITIONING")
    reference_inputs = copy.deepcopy(strains) if validate else None
    device = jax.devices("cpu")[0]

    def condition(strain: Any) -> Any:
        with jax.default_device(device):
            return data_conditioning_single(config, strain)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(workers, len(strains))) as pool:
        results = list(pool.map(condition, strains))
    actual = tuple(zip(*results))
    logger.info("GPU job parallel conditioning: detectors=%d seconds=%.6f", len(strains), time.perf_counter() - start)
    if validate:
        from .validation import leaves

        expected = serial(config, reference_inputs, nproc=nproc)
        if leaves(expected) != leaves(actual):
            raise AssertionError("Conditioned strains/noise maps differ from native serial")
        logger.info("GPU conditioning parity: all detector data and coordinates bitwise exact")
    return actual
