.. _dev_postproduction:

Postproduction architecture
===========================

The postproduction system is built on a **YAML-driven workflow engine**
(:py:mod:`pycwb.post_production.workflow`) that chains actions as a directed
acyclic graph (DAG). Actions are Python functions registered with the
:py:func:`~pycwb.post_production.action_spec.action_spec` decorator
(:py:mod:`pycwb.post_production.action_spec`), declaring their inputs, outputs,
and arguments.

Key modules:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Purpose
   * - :py:mod:`pycwb.modules.postprocess.far`
     - FAR vs. ranking statistic computation
   * - :py:mod:`pycwb.modules.postprocess.train_xgboost`
     - XGBoost classifier training
   * - :py:mod:`pycwb.modules.postprocess.evaluate`
     - Model scoring, FAR evaluation, efficiency
   * - :py:mod:`pycwb.modules.postprocess.selection`
     - Trigger/job selection and train/FAR splitting
   * - :py:mod:`pycwb.modules.postprocess.matching`
     - Trigger-to-injection matching
   * - :py:mod:`pycwb.modules.postprocess.zero_lag`
     - Zero-lag significance analysis
   * - :py:mod:`pycwb.modules.postprocess.report_builder`
     - Multi-tab HTML report generation

For a user-facing workflow, see :doc:`postproduction_workflow`. Exact action
interfaces are in :doc:`postproduction_actions` and persistence rules are in
:doc:`catalog_format`.
