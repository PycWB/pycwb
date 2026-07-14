.. _analysis_recipes:

Analysis Recipes
================

Recipes are copy-paste workflow templates that solve one practical task.
Each includes the goal, inputs, commands, expected outputs, and validation
checks. For step-by-step learning, see :ref:`tutorials`. For help choosing
a recipe, see :ref:`decision_guides`.

Reproducible workflow templates for common pycWB analyses. Each recipe
includes the goal, required inputs, configuration, commands, expected outputs,
and validation checks.

.. contents:: Table of Contents
   :depth: 2
   :local:


All-Sky Short Burst Search
--------------------------

**Goal:** Run a standard all-sky burst search on real detector data.

**Inputs:**
- GWOSC frame files or frame-file list
- DQ segment files (CAT0/1/2)
- ``user_parameters.yaml``

**Key Config:**

.. code-block:: yaml

   # Network
   ifos: [H1, L1]
   fLow: 64
   fHigh: 2048
   inRate: 4096

   # Segment
   segLen: 600
   segMLS: 300
   segEdge: 8

   # Lags
   lagSize: 100
   lagStep: 1.0
   lagOff: 6

   # Likelihood
   netRHO: 4.0
   netCC: 0.5
   healpix: 7
   Acore: 1.414

**Commands:**

.. code-block:: bash

   # Set up working directory
   pycwb config-setup O4_K02_C00_BurstLF_LH_BKG_standard \
       --config-base-path ./config --machine default --datatype gwosc

   # Submit to cluster
   pycwb batch-setup user_parameters.yaml \
       --cluster condor --submit \
       --accounting-group ligo.dev.o4.burst.cwb

**Expected Outputs:**
- ``catalog/catalog.parquet`` — trigger list with SNR, sky position, time, frequency
- ``catalog/progress.parquet`` — per-job processing metadata

**Validation Checks:**
- Triggers appear in catalog (non-empty)
- Zero-lag triggers present (lag_idx = offset index)
- Progress shows all lags completed
- SNR distribution is reasonable (peak near netRHO, long tail)


Targeted External-Trigger Search
--------------------------------

**Goal:** Search a specific sky region around an external trigger (GRB, neutrino).

**Inputs:**
- External trigger RA, Dec, and error radius
- Same data and DQ as all-sky search

**Key Config (add to ``user_parameters.yaml``):**

.. code-block:: yaml

   sky_mask:
     type: Patch
     coordsys: icrs
     patch:
       center:
         ra: "197.5 deg"
         dec: "-23.4 deg"
       radius: "5 deg"

   healpix: 8           # Higher resolution for smaller patch

**Commands:** Same as all-sky search; ``sky_mask`` restricts the likelihood scan.

**Expected Outputs:** Same format, but triggers clustered around the target.

**Validation Checks:**
- All trigger sky positions within patch radius
- Fewer total triggers than all-sky (restricted region)
- Higher healpix gives finer sky localization

**Common Failure Modes:**
- Patch center in wrong coordinate system (check ``coordsys``)
- Radius too small for trigger localization uncertainty
- ``healpix`` too low for small patch (use ≥ 7)


Injection Campaign
------------------

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


Background-Only Production
--------------------------

**Goal:** Run a production background search (no injections) for FAR estimation.

**Inputs:**
- Real detector data with DQ files
- No ``injection`` block in config

**Key Config:**

.. code-block:: yaml

   # Exclude injection block entirely
   # Use production DQ and frame settings
   lagSize: 200          # More lags for better FAR statistics
   slagSize: 10          # Super lags for multi-detector

**Commands:** Same as all-sky search with ``--job-type BKG``.

**Expected Outputs:**
- Trigger catalog with zero-lag and non-zero-lag events
- Progress file with livetime per lag

**Validation Checks:**
- Zero-lag excluded from FAR calculation
- FAR vs. rho curve is smooth and monotonically decreasing
- Total livetime matches N_jobs × N_lags × segLen

**Common Failure Modes:**
- CAT2 veto windows applied as segments instead of windows
- ``lagOff`` incorrectly set (zero-lag leaks into background)
- Train/FAR split not respecting interval boundaries


Training XGBoost Ranking
------------------------

**Goal:** Train an XGBoost classifier to combine event features into a single
ranking statistic.

**Inputs:**
- Background trigger catalog (training fraction)
- Simulation trigger catalog (matched to truth)
- ``xgb_config.py`` with feature list

**Commands:**

.. code-block:: bash

   # Full postproduction workflow
   pycwb post-process standard_analysis_10pct_workflow.yaml

**Expected Outputs:**
- ``model.ubj`` — trained XGBoost model
- Scored background catalog with ranking statistic
- Feature importance table

**Validation Checks:**
- Train and FAR samples are disjoint (no shared job intervals)
- Feature importances are stable across training chunks
- Ranking statistic separates BKG and SIM distributions
- No pathological background sculpting (FAR curve is smooth)

**Common Failure Modes:**
- Train/FAR leakage through interval boundaries
- Too few simulation events for training (need ≥ hundreds)
- Features not computed correctly (check ``xgb_config.py``)


Efficiency Study
----------------

**Goal:** Compute detection efficiency vs. signal amplitude and produce
hrss50/hrss90 sensitivity metrics.

**Inputs:**
- Scored simulation catalog (from XGBoost inference)
- Simulation truth table (``simulations.parquet``)
- Trained model (``model.ubj``)

**Commands:**

.. code-block:: bash

   # Score simulations with trained model
   pycwb post-process efficiency_workflow.yaml

**Expected Outputs:**
- Efficiency vs. hrss curves (per waveform type)
- hrss50 and hrss90 values
- Sigmoid-fit parameters

**Validation Checks:**
- Efficiency → 100% for loud signals (hrss ≫ hrss50)
- hrss50/hrss90 consistent across waveform families
- Binomial error bars decrease with more injections

**Common Failure Modes:**
- FAR threshold too strict (many real signals missed)
- Insufficient injection statistics at low hrss
- Waveform groups not filtering correctly (check ``waveform_groups``)


Debugging a Failed Production
-----------------------------

**Goal:** Diagnose and fix a failed or suspicious production run.

**Checklist:**

1. **Check logs** — ``log/`` directory, look for ERROR or traceback
2. **Check progress** — ``pycwb progress --work-dir .`` shows failed lags/jobs
3. **Check catalog** — is ``catalog.parquet`` non-empty? Reasonable row count?
4. **Check DQ** — do CAT0 segments cover your GPS range? Are CAT2 windows reasonable?
5. **Check frames** — do ``.gwf`` files exist for all detectors and times?
6. **Check zero-lag** — is it excluded from background?
7. **Check SNR distribution** — does it peak near ``netRHO``? Long tail?
8. **Check livetime** — does computed livetime match expected N_jobs × N_lags × segLen?
9. **Check memory** — did any job hit OOM? Check ``job_memory`` setting.
10. **Rescue failed lags** — use ``skip_lags`` to restart from the point of failure

**Common Root Causes:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Symptom
     - Likely Cause
   * - Zero triggers
     - ``netRHO`` too high, wrong frequency range, bad frame data
   * - All triggers at same time
     - Injection GPS times overlap with glitch
   * - FAR curve flat
     - Zero-lag leaked into background
   * - Efficiency near 0%
     - Injections not recovered (check SNR, GPS times, waveform params)
   * - Job OOM
     - ``segLen`` too long, ``healpix`` too high, ``job_memory`` too low
   * - Run time too long
     - ``healpix`` too high, ``lagSize`` too large, no parallelization
