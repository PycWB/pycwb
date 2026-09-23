"""Bounded native TD-cache preparation by independent resolution level.

Runs in the job (parent) process once per trial. Each level is built by the
unchanged native ``_build_td_inputs_single_level`` on the CPU JAX device; the
result layout (each level's values stored under both ``layers`` and
``layers + 1``) is the one produced by the native ``build_td_inputs_cache``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import jax

from pycwb.types.time_series import TimeSeries
from pycwb.utils.td_vector_batch import _build_td_inputs_single_level
from pycwb.utils.td_vector_batch import build_td_inputs_cache as native_build

from . import flags

if TYPE_CHECKING:
    from pycwb.config import Config

logger = logging.getLogger(__name__)

MAX_TD_SETUP_WORKERS = 3
"""Largest ``PYCWB_GPU_TD_SETUP_WORKERS``; each level's WDM context is memory-heavy on the reference host."""


def build_td_inputs_cache(config: Config, strains: list[Any]) -> dict[int, Any]:
    """Build the TD-inputs cache for all WDM levels with bounded threads.

    Drop-in for the native ``build_td_inputs_cache``. Switches, read at call
    time:

    * ``PYCWB_GPU_TD_SETUP_WORKERS`` (default 1, at most
      ``MAX_TD_SETUP_WORKERS``): concurrent level builds.
    * ``PYCWB_GPU_VALIDATE_TD_SETUP=1``: also run the native function and
      require bitwise identical planes, filters and metadata.

    Parameters
    ----------
    config : Config
        Search configuration providing ``WDM_level`` and optionally ``upTDF``.
    strains : list
        Conditioned strains accepted by ``TimeSeries.from_input``.

    Returns
    -------
    dict of int to object
        Native cache layout keyed by layer count.

    Raises
    ------
    ValueError
        If ``PYCWB_GPU_TD_SETUP_WORKERS`` is not an integer in
        ``[1, MAX_TD_SETUP_WORKERS]``.
    AssertionError
        If validation is enabled and the result differs from native.
    """
    workers = flags.worker_count("TD_SETUP_WORKERS", maximum=MAX_TD_SETUP_WORKERS)
    normalized = [TimeSeries.from_input(value) for value in strains]
    up = int(getattr(config, "upTDF", 1))

    def prepare(level: Any) -> tuple[Any, Any]:
        with jax.default_device(jax.devices("cpu")[0]):
            return _build_td_inputs_single_level(level, config, normalized, up)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        groups = list(pool.map(prepare, config.WDM_level))
    result: dict[int, Any] = {}
    for layers, values in groups:
        result[int(layers)] = values
        result[int(layers) + 1] = values
    if flags.enabled("VALIDATE_TD_SETUP"):
        from .validation import leaves

        expected = native_build(config, strains)
        if leaves(expected) != leaves(result):
            raise AssertionError("Parallel TD-cache planes, filters or metadata differ")
        logger.info("GPU TD setup parity: levels=%d all planes filters metadata exact=1", len(groups))
    return result
