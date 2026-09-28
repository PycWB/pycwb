"""Parallel native detector conditioning with explicit scheduling options.

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


if TYPE_CHECKING:
    from pycwb.config import Config

logger = logging.getLogger(__name__)


def condition_strains(
    config: Config, strains: list[Any], *, workers: int = 1, validate: bool = False
) -> tuple[Any, ...]:
    """Condition every detector strain, in parallel threads when requested.

    Parameters
    ----------
    config : Config
        Search configuration.
    strains : list
        One input strain per detector, as accepted by the native functions.

    workers : int
        Maximum concurrent reads or detector tasks; positive, defaults to one.
    validate : bool
        Also run the serial reference and compare exact outputs.

    Returns
    -------
    tuple
        Same layout as the native ``data_conditioning`` return, i.e. one tuple
        per product with a per-detector entry each.

    Raises
    ------
    ValueError
        If workers is not a positive integer.
    AssertionError
        If validation is enabled and the parallel result differs from serial.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if workers == 1:
        return serial(config, strains)
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
        "Parallel conditioning: detectors=%d seconds=%.6f",
        len(strains),
        time.perf_counter() - start,
    )
    if validate:
        from pycwb.utils.fingerprint import leaves

        expected = serial(config, reference_inputs)
        if leaves(expected) != leaves(actual):
            raise AssertionError("Conditioned strains/noise maps differ from native serial")
        logger.info("Conditioning parity: all detector data and coordinates bitwise exact")
    return actual
