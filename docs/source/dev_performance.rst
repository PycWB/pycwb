.. _dev_performance:

Performance Guide
=================

Configure native execution, understand its memory and throughput tradeoffs,
and optimize pycWB's computational hot paths. Scientific calculation choices
are explained in :ref:`native_calculation_conventions`; detector inputs are
explained in :ref:`detector_support`.

For experimental job scheduling, shared raw-input caching, and worker memory
budgets, see :ref:`workflow_execution`. These use the separate ``execution``
configuration block; ``execution_profile`` below controls native processing
options within each job.

For experimental resource-aware scheduling, shared raw-input caching, and
worker memory budgets, see :ref:`workflow_execution`.

.. contents:: Table of Contents
   :depth: 2
   :local:


.. _execution_profile:

Configure a native execution profile
------------------------------------

Put execution settings in ``user_parameters.yaml`` under ``execution_profile``.
The schema fills omitted defaults and rejects unknown keys, quoted booleans and
invalid values. Configuration loading resolves the profile once; each prepared
job uses that immutable snapshot. To change settings, create a new configuration
and setup. Changing environment variables inside a running process has no effect
on these choices.

For example, this fragment retains default numerical choices while enabling
compact, staged time-delay inputs and less frequent full garbage collection:

.. code-block:: yaml

   max_energy_backend: jax
   coherence_timing: false
   execution_profile:
     sky_delay_reuse: true
     compact_td_cache: true
     staged_td: true
     gc_full_interval: 16

This is a fragment to merge into an analysis configuration, not a complete run.
The settings below apply to the native paths that consume the profile; they do
not switch a legacy ROOT processor to native execution. The native segment
processor can be selected with:

.. code-block:: yaml

   segment_processer: pycwb.workflow.subflow.process_job_segment_native.process_job_segment

The full profile, including defaults, is saved in catalog metadata as
``config.execution_profile`` and restored by batch workers. Reusing a catalog
with a different profile is rejected. Catalogs created before profiles were
recorded require a new run; historical environment switches cannot be recovered
reliably from those catalogs.

The old ``PYCWB_*`` execution and ``WDM_*`` transform environment switches are no
longer read. Installation paths such as ``HOME_WAT_FILTERS`` and external library
thread counts/device visibility remain outside this profile. The scheduler or
launcher still controls CPU/GPU allocation.

.. _execution_profile_options:

Execution and memory options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every option in the tables below belongs inside ``execution_profile``. Defaults
are shown explicitly. Enabling an option selects an implementation; speed and
peak memory depend on cluster size, sky grid, transform shape and hardware.
Keep scientific settings and input geometry fixed when comparing performance.

Sky likelihood
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 11 60

   * - Setting
     - Default
     - Meaning and use
   * - ``sky_delay_reuse``
     - ``true``
     - Group sky directions with identical delay tuples, loading delayed data once per group. False uses singleton groups in the same compiled scan. Both retain original sky-order tie-breaking. Grouping may reduce parallelism when very few groups remain; scratch reuse is internal, with no separate switch.
   * - ``scalar_dpf``
     - ``false``
     - Select the scalar dominant-polarization-frame regulator kernel to reduce temporary allocations. This selects regulator implementation, not sky grouping; validate numerical outputs for the analysis being compared.

Coherence and time-delay storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 11 60

   * - Setting
     - Default
     - Meaning and use
   * - ``coherence_early_cuts``
     - ``false``
     - Apply coherence cuts before materializing clusters, avoiding allocation for rejected candidates.
   * - ``preindex_shifts``
     - ``false``
     - Precompute shifted time indices used by pixel selection instead of repeatedly constructing them.
   * - ``cluster_runs``
     - ``false``
     - Use runs of adjacent pixels for connectivity calculation while retaining the original pixels for likelihood.
   * - ``compact_coherence``
     - ``false``
     - Share immutable coherence energy storage to reduce copies.
   * - ``direct_max_energy_input``
     - ``false``
     - Experimental: feed conditioned strain directly into max-energy calculation. Keep disabled for the documented bounded CPU recipe: a pilot changed an intermediate LF cluster.
   * - ``tiled_wdm``
     - ``false``
     - Use tiled forward WDM with bounded workspace for supported CPU shapes. Unsupported shapes retain the existing transform path; memory savings therefore vary with segment geometry.
   * - ``compact_td_cache``
     - ``false``
     - Store time-delay inputs in a compact representation to reduce retained memory.
   * - ``band_td_cache``
     - ``false``
     - Experimental: restrict compact time-delay storage to selected frequency bands. Requires compact_td_cache: true; it is not enabled in the documented CPU recipe.
   * - ``staged_td``
     - ``false``
     - Populate coarse time-delay vectors for subnet selection before computing fine likelihood vectors for surviving clusters.

Max energy and regression backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 11 60

   * - Setting
     - Default
     - Meaning and use
   * - ``bounded_jax_max_energy``
     - ``false``
     - Use bounded JAX transforms inside max-energy calculation. Applies to the JAX max-energy path; this is separate from WDM instance options below.
   * - ``numba_max_energy_mode``
     - ``parallel``
     - Select Numba max-energy traversal: parallel, time-major or streaming. Applies when Numba is selected, including a Numba fallback. Alternate traversal modes are experimental; measure complete-job behavior.
   * - ``regression_engine``
     - ``numba``
     - Select the regression numerical backend: numba or jax. This is independent of the top-level max_energy_backend selection.

WDM transforms
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 11 60

   * - Setting
     - Default
     - Meaning and use
   * - ``wdm_bounded_jax_forward``
     - ``false``
     - Use bounded workspace for forward JAX WDM transforms where supported.
   * - ``wdm_bounded_jax_inverse``
     - ``false``
     - Use bounded workspace for inverse JAX WDM transforms where supported.
   * - ``wdm_deterministic_jax_inverse``
     - ``false``
     - Request ordered deterministic inverse JAX accumulation. This controls accumulation behavior, not device placement; it does not promise whole-pipeline bitwise equality across hardware.
   * - ``wdm_bounded_numba``
     - ``false``
     - Use bounded Numba forward WDM when the transform uses the Numba backend.
   * - ``wdm_compact_complex``
     - ``false``
     - Assemble complex WDM maps without separate quadrature temporaries.

Cleanup and diagnostics
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 29 11 60

   * - Setting
     - Default
     - Meaning and use
   * - ``gc_full_interval``
     - ``1``
     - Positive integer: number of lag cleanup calls between full garbage collections. Larger values reduce full-collection overhead; intermediate calls collect younger objects, with a memory-growth safeguard. Automatic garbage collection remains enabled. Monitor memory when increasing it.
   * - ``perf_diagnostics``
     - ``false``
     - Log detailed native performance diagnostics. Useful for investigation; extra instrumentation can affect timings.
   * - ``require_gpu``
     - ``false``
     - Fail the batch GPU check if JAX has no GPU device. This does not select a GPU implementation or request a GPU from the scheduler.

WDM options are passed explicitly through conditioning, injections, coherence,
time-delay preparation and reconstruction. Use a companion ``wdm-wavelet``
installation that supports these explicit constructor options. Enabling a
backend-specific flag does not itself select that backend.

Calculation choices in the same profile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The remaining four fields are ``native_chirp`` (false),
``release_waveform_stats`` (false), ``regression_cap`` (false), and
``regression_percentile_stride`` (1). These affect estimators, summary arithmetic
or statistical sampling. Their behavior and restrictions are documented in
:ref:`native_calculation_conventions`; treat them as analysis choices when
reviewing results, even though they share the execution-profile container.

Top-level backend and timing settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``max_energy_backend`` is outside the profile and defaults to ``jax``. Accepted
YAML values are ``jax``, ``numba`` and ``auto``. Auto uses Numba for middle WDM
resolutions and JAX for endpoint resolutions; the selected backend is recorded
in coherence setup. It is a resolution-based policy, not a live benchmark.
JAX can execute on a CPU, so this setting does not imply GPU use.

``coherence_timing`` is also top-level and defaults to false. It enables
coherence setup timing logs. Use it for setup timing; use
``execution_profile.perf_diagnostics`` for broader native diagnostics.

Choosing and measuring a profile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start from defaults, change settings relevant to the observed bottleneck, and
compare output fields as well as complete-job time and peak memory. Separate
JIT warm-up from steady-state timing. Measure representative segment lengths,
resolutions and sky grids; bounded paths can fall back for unsupported shapes.
Sky grouping speedups depend on how many directions share a delay tuple.

The :download:`bounded CPU recipe <../../examples/performance/bounded_cpu.yaml>`
is an optional parameter fragment. It combines memory/throughput settings with
explicit scientific calculation choices; it is not the default profile or a
performance-only preset. Review those choices before merging it into a run.
CPU campaign measurements found peak-memory reductions of 15–38% and about
2–4% faster execution for six paired runs, while other short workloads had
small slowdowns. These combined-profile measurements do not establish an
isolated speedup for any individual flag or predict GPU performance.

The execution-profile migration was checked with native and companion WDM
regression tests, configuration/catalog round trips and a saved likelihood
replay. It did not include a new full-job throughput or GPU campaign.


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

- **NumPy in hot paths**: Measure representative workloads before choosing
  NumPy, Numba or JAX; compilation and transfer overhead can dominate small tasks.
- **Python objects in loops**: Never iterate over Python lists inside
  performance-critical code. Use NumPy/JAX arrays.
- **JAX buffer leaks**: Always free JAX device buffers after each lag.
- **ROOT overhead**: Avoid ROOT I/O in hot paths. Use Parquet via pyarrow.
- **Large JIT compilation**: Split large functions into smaller JIT-
  compilable units to reduce first-call latency.
