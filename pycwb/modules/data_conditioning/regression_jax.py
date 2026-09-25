"""
JAX-accelerated regression layer for pycWB data conditioning.

This module exposes the JAX backend for the WDM regression filter, which
uses ``jax.vmap`` to process all TF frequency layers simultaneously inside a
single ``@jax.jit``-compiled call instead of a sequential Numba prange loop.

    The public entry point is ``regression_jax``, which has an identical interface
    to ``regression_python`` in ``regression.py`` but always enforces the JAX
backend through an explicit copy of the execution profile.

Key JAX components (all defined in regression.py and re-exported here)
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
from copy import copy
from dataclasses import replace
from pycwb.constants.execution_profile import execution_profile

logger = logging.getLogger(__name__)


def regression_jax(config, h):
    """
    Run WDM regression with the JAX vmap backend unconditionally.

    This is a thin wrapper around ``regression_python`` that forces
    ``execution_profile.regression_engine='jax'`` without mutating the
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
    from pycwb.modules.data_conditioning.regression import regression_python

    selected = copy(config)
    selected.execution_profile = replace(execution_profile(selected), regression_engine="jax")
    return regression_python(selected, h)


__all__ = ["regression_jax"]
