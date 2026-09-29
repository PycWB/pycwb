.. _recipe_injections:

Injection Campaign
==================

**Goal:** Measure detection efficiency by injecting simulated signals and
recovering them.

**Inputs:**

- Base ``user_parameters.yaml`` with noise config
- Injection parameters (waveform, sky distribution, amplitude range)

**Key Config:**

.. code-block:: yaml

   injection:
     seed: 42
     repeat_injection: 1
     parameters:
       - mass1: 35
         mass2: 35
         approximant: IMRPhenomPv2
         f_lower: 20
         delta_t: 0.000244140625
         hrss: 1e-21
     sky_distribution:
       type: UniformAllSky
     time_distribution:
       type: poisson
       mean_interval: 500.0
       max_trail: 10

   parallel_injection_trail: true
   iwindow: 5.0

**Commands:**

.. code-block:: bash

   # Run injection search
   pycwb run user_parameters_injection.yaml

   # Build simulation summary
   pycwb simulation-summary user_parameters_injection.yaml \
       --work-dir . \
       --output catalog/simulations.parquet

**Expected Outputs:**

- ``catalog/catalog.parquet`` — recovered triggers
- ``catalog/simulations.parquet`` — one row per injection (truth table)

**Validation Checks:**

- Every injection has a row in simulations.parquet
- Recovered triggers have matching sim_idx
- Sky positions of recovered injections match distribution
- Efficiency increases with hrss

**Common Failure Modes:**

- ``iwindow`` too small to contain waveform
- GPS times outside segment windows
- ``netRHO`` too high for faint injections
- ``parallel_injection_trail`` not set (trials not parallelized)

See :doc:`analysis_recipes` for the other analysis tasks.
