.. _postproduction_study:

Run Ranking and Efficiency on a Study
=====================================

Use this guide with completed background (BKG) and simulation (SIM) catalogs
from your own study. It explains how to adapt the maintained workflow template,
select its inputs and inspect the resulting model, FAR mapping and efficiency.
For a small worked example of truth matching or background accounting, see
:doc:`tutorial_population` and :doc:`tutorial_background`.

.. _postproduction_study_inputs:

Prepare the study inputs
------------------------

Install the XGBoost extra for your current checkout:

.. code-block:: bash

   python -m pip install '.[xgboost]'

Use ``examples/postproduction/standard_analysis_10pct_workflow.yaml`` as a
complete workflow template. Copy it into your study directory. Its ``vars``
section names the training background, target background/progress, training
simulations, independent evaluation simulations, model configuration and output
paths. Replace every ``/path/to`` entry and resolve the chunk/run names to
actual completed catalogs. Preserve catalog manifests alongside those files.

.. list-table:: Input checklist
   :header-rows: 1

   * - Input
     - Purpose
   * - BKG catalogs and progress
     - Training events and a disjoint FAR evaluation sample with exposure.
   * - Training SIM catalogs and truth
     - Recovered eligible sources used to train the classifier.
   * - Evaluation SIM catalogs, progress and truth
     - All eligible trials used to measure recovery, including missed sources.
   * - ``xgb_config.py``
     - The feature and model settings for the chosen search family.

Generate ``simulations.parquet`` for each simulation run with
``pycwb simulation-summary --work-dir RUN_DIRECTORY``. Inspect truth and matching
as in :doc:`tutorial_population`. The standard workflow includes optional MDC
and public-alert report sections; either provide those inputs or remove the
corresponding steps and report references together.

.. _postproduction_study_workflow:

Inspect the workflow graph
--------------------------

After saving your edited workflow as ``study.yaml``:

.. code-block:: bash

   pycwb post-process study.yaml --diagram-only
   pycwb post-process study.yaml --no-diagram

The graph should show selection/splitting before training, FAR evaluation on
the holdout, simulation matching, scoring and efficiency reporting. The template
uses an interval-based background split. Inspect the selected intervals and
exposure; distinct filenames alone do not prove independence.

.. _postproduction_study_products:

Inspect four products
---------------------

1. **The model and training record.** Check the selected features, source
   populations and training samples. Keep the model together with its settings.
2. **The scored FAR holdout.** Plot the new ranking and derive its FAR using
   that sample's selected exposure. A classifier score is not itself a FAR.
3. **The scored evaluation simulations.** Retain the full eligible truth
   population in the denominator; inspect duplicate matches and missed sources.
4. **Efficiency curves.** At a stated FAR threshold, plot recovered fraction
   against source amplitude for each waveform family, with uncertainty. Only
   report hrss50/hrss90 if the sampled population supports the crossing.

For CBC populations, distance or another explicitly defined population
coordinate may be more useful than source hrss. Fixed-amplitude and target-SNR
populations answer different questions. Do not reuse evaluation samples to
tune the model and then interpret them as an independent sensitivity test.

**Result to keep:** the model, held-out FAR mapping, eligible/recovered counts,
and sensitivity curves with their threshold, units and population definition.
Detailed action options are in :doc:`postproduction_actions`,
:doc:`postproduction_xgboost` and :doc:`postproduction_efficiency`.
