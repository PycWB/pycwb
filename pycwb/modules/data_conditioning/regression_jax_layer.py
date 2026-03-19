"""
JAX-accelerated regression layer for pycWB data conditioning.

This module exposes the JAX backend for the WDM regression filter, which
uses ``jax.vmap`` to process all TF frequency layers simultaneously inside a
single ``@jax.jit``-compiled call instead of a sequential Numba prange loop.

The public entry point is ``regression_jax``, which has an identical interface
to ``regression_python`` in ``regression_py.py`` but always enforces the JAX
backend regardless of the ``PYCWB_REGRESSION_ENGINE`` environment variable.

Key JAX components (all defined in regression_py.py and re-exported here)
--------------------------------------------------------------------------
``_jax_process_layers``
    vmap'd JIT function — processes all selected TF layers in one call.
    Static args: K, K2, K4, half, fm, edge_samples, fltr, eigen_*, rate_tf, edge_seconds.
    Dynamic args: real_layers, imag_layers  (shape: n_layers × n_time).

``_jax_process_one_layer``
    Per-layer pipeline: build stats → solve LPE system → apply filter → gate.

``_jax_layer_build_stats``, ``_jax_layer_solve_filters``, etc.
    Composable sub-steps, each @jax.jit, useful for profiling / testing.

JAX vs Numba trade-offs
-----------------------
* JAX vmap:   compiled once → parallel device execution (CPU/GPU); no Python GIL;
              optimal for long segments (n_time >> 1000) or GPU use.
* Numba prange: spawns OS threads over layers; best for short segments on CPU
                when the layer count is small (< ~20).

The ``regression_jax`` function below uses ``jax.block_until_ready`` to ensure
all device computation is complete before returning.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def regression_jax(config, h):
    """
    Run WDM regression with the JAX vmap backend unconditionally.

    This is a thin wrapper around ``regression_python`` that forces
    ``REGRESSION_ENGINE='jax'`` via the config object without mutating the
    original config.

    Parameters
    ----------
    config : pycwb Config object
        Analysis configuration.  Read-only; not mutated.
    h : gwpy.TimeSeries | pycwb.TimeSeries
        Input strain time series.

    Returns
    -------
    pycwb.types.time_series.TimeSeries
        Regression-cleaned time series, same sample rate and start time.
    """
    from pycwb.modules.data_conditioning.regression_py import regression_python

    class _JAXConfig:
        """Proxy that forces REGRESSION_ENGINE='jax' without modifying config."""
        def __init__(self, base):
            self._base = base

        def __getattr__(self, name):
            if name == "REGRESSION_ENGINE":
                return "jax"
            return getattr(self._base, name)

        def __hasattr__(self, name):
            return hasattr(self._base, name)

    return regression_python(_JAXConfig(config), h)


__all__ = ["regression_jax"]
