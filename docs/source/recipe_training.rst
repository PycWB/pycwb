.. _recipe_training:

Training XGBoost Ranking
========================

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

See :doc:`analysis_recipes` for the other analysis tasks.
