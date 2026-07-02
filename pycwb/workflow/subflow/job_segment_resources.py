"""Resource and thread helpers for the native job-segment workflow."""

import logging
from contextlib import contextmanager

from pycwb.config import Config

logger = logging.getLogger(__name__)


def _free_jax_buffers():
    """Release JAX device arrays still held in memory.

    JAX allocates GPU/CPU device buffers that Python's GC does not always
    reclaim promptly. Call this between lags to prevent accumulation. Safe to
    call even when JAX is not installed or not in use.
    """
    try:
        import jax

        for _device in jax.devices():
            try:
                # Clear the live arrays tracked by the JAX memory allocator.
                jax.clear_backends()
                break  # clear_backends() clears all backends at once
            except Exception:
                pass
    except Exception:
        pass  # JAX not installed; nothing to free


@contextmanager
def _temporary_numba_threads(n_threads: int | None):
    """Temporarily override the Numba thread count, restoring on exit."""
    if n_threads is None:
        yield
        return

    # Best-effort: if Numba is not available, just run the block unchanged.
    try:
        import numba
    except Exception:
        yield
        return

    old_threads = numba.get_num_threads()
    max_allowed = getattr(numba.config, "NUMBA_NUM_THREADS", old_threads)
    target = max(1, int(n_threads))
    if max_allowed > 0:
        target = min(target, int(max_allowed))

    if target != old_threads:
        numba.set_num_threads(target)
        logger.debug("Numba threads: %d -> %d (max %d)", old_threads, target, max_allowed)

    try:
        yield
    finally:
        if target != old_threads:
            numba.set_num_threads(old_threads)


def _parallel_inner_threads(config: Config, lag_workers: int) -> int:
    configured = getattr(config, "parallel_lag_inner_threads", None)
    if configured is not None:
        return max(1, int(configured))
    nproc = int(getattr(config, "nproc", 0) or 0)
    if nproc <= 0:
        return 1
    return max(1, nproc // max(1, int(lag_workers)))
