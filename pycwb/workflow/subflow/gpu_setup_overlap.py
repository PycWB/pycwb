"""Overlap independent preparation stages without retaining background threads.

The native processor calls ``setup_coherence`` and then ``build_td_inputs_cache``
on the same trial inputs. :class:`OverlappedSetup` computes the TD cache on one
helper thread while the coherence setup runs on the calling thread, joins the
helper before returning, and hands the cached result back on the subsequent
``build_td_inputs_cache`` call. No executor thread survives into lag-worker
creation.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import jax

logger = logging.getLogger(__name__)


class OverlappedSetup:
    """Own one trial's TD result until the native processor consumes it.

    The two methods must be called strictly alternately for the same
    ``(config, strains)`` objects: :meth:`setup_coherence` stores the TD cache
    under ``pending`` and :meth:`build_td_inputs_cache` releases it. Identity
    (``is``) of ``config`` and ``strains`` is checked, not equality, because
    the native processor passes the very same objects to both calls.

    Parameters
    ----------
    coherence : callable
        ``setup_coherence(config, strains, **kwargs)`` implementation.
    td : callable
        ``build_td_inputs_cache(config, strains)`` implementation, run on the
        helper thread inside a CPU JAX default-device scope.

    Attributes
    ----------
    pending : tuple or None
        ``(config, strains, cache)`` of the trial whose TD cache has been
        computed but not yet consumed.
    """

    def __init__(self, coherence: Callable[..., Any], td: Callable[[Any, Any], Any]) -> None:
        self.coherence = coherence
        self.td = td
        self.pending: tuple[Any, Any, Any] | None = None

    def setup_coherence(self, config: Any, strains: Any, **kwargs: Any) -> Any:
        """Run the coherence setup while the TD cache builds on one helper thread.

        Parameters
        ----------
        config
            Trial configuration passed to both preparation functions.
        strains
            Conditioned strains passed to both preparation functions.
        **kwargs
            Forwarded to ``coherence`` only.

        Returns
        -------
        object
            The ``coherence`` result.

        Raises
        ------
        RuntimeError
            If the previous trial's TD cache was never consumed.
        """
        if self.pending is not None:
            raise RuntimeError("Previous trial's TD cache was not consumed")
        device = jax.devices("cpu")[0]

        def prepare_td() -> Any:
            with jax.default_device(device):
                return self.td(config, strains)

        start = time.perf_counter()
        # Join before returning: no executor threads survive into lag-worker
        # creation, including when either preparation function raises.
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(prepare_td)
            result = self.coherence(config, strains, **kwargs)
            cache = future.result()
        self.pending = (config, strains, cache)
        logger.info("GPU overlapped coherence/TD preparation: seconds=%.6f", time.perf_counter() - start)
        return result

    def build_td_inputs_cache(self, config: Any, strains: Any) -> Any:
        """Return the TD cache computed during :meth:`setup_coherence` and clear it.

        Parameters
        ----------
        config
            Must be the same object passed to :meth:`setup_coherence`.
        strains
            Must be the same object passed to :meth:`setup_coherence`.

        Returns
        -------
        object
            The ``td`` result for this trial.

        Raises
        ------
        RuntimeError
            If :meth:`setup_coherence` has not run since the last consumption.
        ValueError
            If ``config`` or ``strains`` is not the pending trial's object.
        """
        if self.pending is None:
            raise RuntimeError("Coherence preparation must precede TD consumption")
        owner_config, owner_strains, cache = self.pending
        if config is not owner_config or strains is not owner_strains:
            raise ValueError("TD cache requested for different trial inputs")
        self.pending = None
        return cache
