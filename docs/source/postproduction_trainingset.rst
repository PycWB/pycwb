.. _postproduction_trainingset:

Training Set Preparation
========================

This guide explains how pycWB selects and prepares training data for the
XGBoost ranking classifier, including background/simulation splitting
strategies and injection matching.

.. contents:: Table of Contents
   :depth: 2
   :local:


Overview
--------

The XGBoost classifier requires labeled training data: **background** triggers
(label = 0) and **simulation** triggers (label = 1). Careful training set
preparation is essential to avoid biases that could invalidate the background
estimate or inflate sensitivity claims.


Data Requirements
-----------------

A complete training setup requires these artifacts from the search jobs:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Artifact
     - Description
   * - Background catalog (``catalog.parquet``)
     - Trigger rows from non-zero-lag analysis of background data
   * - Background progress (``progress.parquet``)
     - Per-job/per-lag processing metadata with livetime
   * - Simulation catalog (``catalog.parquet``)
     - Trigger rows from injection recovery analysis
   * - Simulation progress (``progress.parquet``)
     - Per-job metadata for simulation jobs
   * - Simulation summary (``simulations.parquet``)
     - One row per injected signal, built before postproduction

Build the simulation summary file with:

.. code-block:: bash

   pycwb simulation-summary user_parameters.yaml \
     --work-dir /path/to/sim/run \
     --output /path/to/sim/run/catalog/simulations.parquet


Training Set Splitting
----------------------

Background Split
~~~~~~~~~~~~~~~~

The background catalog must be split into independent **training** and **FAR**
holdout subsets. Using the same background data for both training and FAR
evaluation creates a bias (the classifier learns the noise fluctuations it's
supposed to measure).

Three splitting strategies are available:

1. ``interval_livetime`` (recommended): Splits by time intervals within
   jobs. The same physical data segment cannot appear in both train and FAR
   through different lag or shift combinations. This is the most rigorous
   approach for production analyses.

2. ``job``: Splits by whole job segments. Simpler but requires that jobs
   do not overlap in physical time.

3. ``fraction``: Simple random split of trigger rows. Useful for quick
   testing but may introduce correlations between train and FAR if triggers
   from the same noise transient appear at multiple lags.

Example workflow step:

.. code-block:: yaml

   - id: bkg_split
     name: Split Background Train/FAR
     action: postprocess.selection.trigger_selection
     inputs:
       catalog_file: ${paths.bkg_catalog}
       progress_file: ${paths.bkg_progress}
     args:
       exclude_zero_lag: true
       returns: [jobs, triggers, livetime]
       split:
         by: interval_livetime
         seed: 42
         fractions:
           train: 0.1
           far: 0.9
     outputs:
       train:
         triggers_file: tmp://bkg_train.parquet
       far:
         triggers_file: tmp://bkg_far.parquet


Simulation Selection
~~~~~~~~~~~~~~~~~~~~

Simulation triggers must be:

1. **Matched** to injection truth via
   :py:func:`pycwb.modules.postprocess.matching.match_simulations`.
2. **Filtered** to keep only recovered, non-vetoed signals.
3. **Selected** to a training fraction consistent with the background split.

Matching uses interval overlap between trigger time and injection time:

.. code-block:: yaml

   - id: sim_match
     name: Match Simulations
     action: postprocess.matching.match_simulations
     inputs:
       catalog_file: ${paths.sim_catalog}
       simulation_file: ${paths.simulations}
     args:
       how: outer              # outer = include all injections (recovered + missed)
       window_buffer: 0.0
     outputs:
       output_file: tmp://sim_matched.parquet

The ``how`` parameter:

- ``outer`` (training): Include all injection rows, even unrecovered ones.
  Unrecovered signals are labeled as background for classifier training,
  teaching the model what *wasn't* detected.
- ``inner`` (efficiency): Include only matched trigger-injection pairs.
  Used for detection efficiency computation.
- ``right``: Include all simulation rows with matching info added. Useful
  for standalone matching outside a workflow.

Filtering real (non-zero-lag, non-vetoed) simulation triggers:

.. code-block:: yaml

   - id: sim_real
     name: Filter Real SIM
     action: postprocess.selection.filter_real_simulation
     inputs:
       matched_file: "@sim_match.matched_file"
       sim_catalog: ${paths.sim_catalog}
     args:
       require_recovered: true
       exclude_vetoed: true
       output_schema: matched
     outputs:
       output_file: tmp://sim_real.parquet


Training Fraction Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

After filtering, select the training fraction:

.. code-block:: yaml

   - id: sim_train_select
     name: Select SIM Training Fraction
     action: postprocess.selection.trigger_selection
     inputs:
       catalog_file: "@sim_real.triggers_file"
       progress_file: ${paths.sim_progress}
     args:
       exclude_zero_lag: false    # Simulations are at zero-lag
       returns: [jobs, triggers, livetime]
       selection:
         fraction: 0.1            # Match background train fraction
         seed: 43
     outputs:
       triggers_file: tmp://sim_train.parquet


Multi-Chunk Training
--------------------

For production analyses, the training set typically combines data from
multiple observing chunks to increase statistical power:

.. code-block:: yaml

   - id: model
     name: Train XGBoost
     action: postprocess.train_xgboost.train_xgboost
     inputs:
       bkg_catalogs:
         - /path/to/O4_K20/bkg_train.parquet
         - /path/to/O4_K22/bkg_train.parquet
         - "@bkg_split.train.triggers_file"      # Target chunk train
       sim_catalogs:
         - /path/to/O4_K20/sim_train.parquet
         - /path/to/O4_K22/sim_train.parquet
         - "@sim_train_select.triggers_file"     # Target chunk train

The ``@step.field`` syntax references outputs from earlier workflow steps,
enabling a single YAML to orchestrate the entire train-FAR-efficiency pipeline.


Best Practices
--------------

- **Keep train/FAR seeds fixed** for reproducibility across analysis versions.
- **Use the same train fraction** (e.g., 10%) for both background and
  simulation to maintain class balance.
- **Train on multiple chunks** but evaluate FAR on each target chunk
  independently to avoid overfitting.
- **Verify train/FAR independence** by checking that no physical time interval
  appears in both subsets (visualize interval overlap with the progress
  Parquet files).
- **Build simulations.parquet** before starting the postproduction workflow.
  Use ``pycwb simulation-summary`` to generate it from the simulation run.


----

**See also:** :doc:`postproduction_xgboost` · :doc:`postproduction_background` · :doc:`injection_infrastructure`

**Next:** :doc:`postproduction_xgboost` — training the XGBoost classifier
