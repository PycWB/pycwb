.. _recipe_all_sky:

All-Sky Short Burst Search
==========================

**Goal:** Run a standard all-sky burst search on real detector data.

**Inputs:**

- GWOSC frame files or frame-file list
- DQ segment files (CAT0/1/2)
- ``user_parameters.yaml``

**Key Config:**

.. code-block:: yaml

   # Network
   ifo: [H1, L1]
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

See :doc:`analysis_recipes` for the other analysis tasks.
