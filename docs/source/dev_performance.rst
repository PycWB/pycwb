.. _dev_performance:

Performance Guide
=================

How to write and optimize high-performance code for pycWB's computational
hot paths.

.. contents:: Table of Contents
   :depth: 2
   :local:


Performance Strategy
--------------------

pycWB's goal: **best single CPU-core throughput now; GPU acceleration via
JAX in the future.** All hot-path code must target at least one of Numba or
JAX — never use pure NumPy for inner loops.


Numba Patterns
--------------

Use Numba ``@njit`` with ``prange`` for CPU-bound loops over time-delay
batches:

.. code-block:: python

   from numba import njit, prange
   import numpy as np

   @njit(parallel=True)
   def process_time_delays(data, delays, output):
       """Process time-delay batched data in parallel."""
       for i in prange(len(delays)):
           t = delays[i]
           output[i] = np.sum(data[t : t + window] ** 2)
       return output

**Key files**: :py:mod:`pycwb.utils.td_vector_batch`

**Tips**:
- Use ``prange`` instead of ``range`` for CPU parallelism.
- Keep Numba functions small and focused — large functions have longer
  compilation times.
- Avoid Python objects inside ``@njit`` functions — use NumPy arrays and
  scalars only.
- Profile with ``@njit`` first, add ``parallel=True`` only when the loop
  is large enough to benefit.


JAX Patterns
------------

Use JAX ``jit`` + ``vmap`` for batched coherence and likelihood computations:

.. code-block:: python

   import jax
   import jax.numpy as jnp

   @jax.jit
   def coherent_energy(data, antenna_patterns):
       """Compute coherent energy for all sky directions."""
       return jnp.sum((data @ antenna_patterns) ** 2, axis=-1)

   # Vectorize over sky directions
   batch_coherent = jax.vmap(coherent_energy, in_axes=(None, 0))
   result = batch_coherent(data, all_sky_patterns)

**Key files**: :py:mod:`pycwb.modules.coherence.coherence`

**Tips**:
- Write device-agnostic code — same code runs on CPU and GPU.
- JAX compilation cache: ``~/.cache/pycwb/jax_compilation_cache/``.
- First call compiles (slow), subsequent calls are fast.
- Use ``jax.block_until_ready()`` for accurate timing benchmarks.


Memory Management (Critical)
----------------------------

**JAX device buffers must be explicitly freed after each lag.** This is a
known pitfall that causes memory leaks in long-running analyses:

.. code-block:: python

   for lag_idx in range(n_lags):
       result = jax_computation(data, lag_idx)
       # ... use result ...

       # CRITICAL: free JAX device buffers
       for buf in result:
           if hasattr(buf, 'delete'):
               buf.delete()
       jax.clear_caches()  # optional, for very long runs

Without explicit cleanup, each lag accumulates GPU/TPU memory until the
process runs out.


Struct-of-Arrays (SoA) Layout
-----------------------------

Pixel data uses a **struct-of-arrays** layout
(:py:class:`~pycwb.types.pixel_arrays.PixelArrays`) instead of
array-of-structs for better cache locality and vectorization:

.. code-block:: python

   # SoA: each field is a contiguous array
   pixels = PixelArrays(
       time=np.array([...]),       # N elements
       frequency=np.array([...]),  # N elements
       rate=np.array([...]),       # N elements
       layers=np.array([...]),     # N elements
       pixel_index=[...],          # per-IFO indices
   )

   # Fast: vectorized operations on contiguous arrays
   central_time = pixels.time / (pixels.rate * pixels.layers)


Profiling
---------

**Line profiling**:

.. code-block:: bash

   pip install line_profiler
   # Add @profile decorator to suspect function
   kernprof -l -v script.py

**JAX profiling**:

.. code-block:: python

   with jax.profiler.trace("/tmp/jax-trace"):
       result = jax_computation(data)

**Numba profiling**:

.. code-block:: python

   from numba import njit
   # Check compilation time
   %timeit njit(my_func)(data)  # first call
   %timeit njit(my_func)(data)  # subsequent calls


Performance Benchmarks
----------------------

Pre-written benchmarks live in:

- ``_test_njit.py`` — Numba warm-up and throughput
- ``_test_mra_njit.py`` — Multi-Resolution Analysis benchmarks
- ``benchmark/`` — Additional benchmarks (I/O, likelihood, supercluster)

Run before and after performance changes to verify no regressions.


Avoiding Common Pitfalls
------------------------

- **NumPy in hot paths**: Pure NumPy is 10–100× slower than Numba/JAX for
  inner loops. Always use Numba or JAX for per-pixel or per-lag operations.
- **Python objects in loops**: Never iterate over Python lists inside
  performance-critical code. Use NumPy/JAX arrays.
- **JAX buffer leaks**: Always free JAX device buffers after each lag.
- **ROOT overhead**: Avoid ROOT I/O in hot paths. Use Parquet via pyarrow.
- **Large JIT compilation**: Split large functions into smaller JIT-
  compilable units to reduce first-call latency.
