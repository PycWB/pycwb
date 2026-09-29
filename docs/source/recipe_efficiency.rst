.. _recipe_efficiency:

Efficiency Study
================

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
- hrss50/hrss90 agree with the reference for each waveform family; different families can have different sensitivities
- Binomial error bars decrease with more injections

**Common Failure Modes:**

- FAR threshold too strict (many real signals missed)
- Insufficient injection statistics at low hrss
- Waveform groups not filtering correctly (check ``waveform_groups``)

See :doc:`analysis_recipes` for the other analysis tasks.
