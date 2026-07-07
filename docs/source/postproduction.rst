.. _postproduction:

Postproduction
==============

The postproduction pipeline takes the trigger catalogs produced by pycWB
search jobs and produces final analysis products: background estimates, ranked
candidate lists, detection efficiency curves, and HTML summary reports.

.. mermaid::

   flowchart TD
     A[Background triggers] --> B{Split background}
     B --> C[Background training set]
     B --> D[Background test set]

     E[Simulation triggers<br/>training injections] --> F[XGBoost training]
     C --> F

     F --> G[Trained XGBoost model]

     H[Zero-lag triggers] --> I[Apply model]
     D --> J[Apply model]

     G --> I
     G --> J

     I --> K[Reweighted zero-lag triggers]
     J --> L[Reweighted background triggers]

     L --> M[Background statistics<br/>FAR / IFAR]
     K --> N[Candidate significance]
     M --> N

This section covers the complete postproduction workflow and each analysis
component.

.. toctree::
   :maxdepth: 2

   postproduction_workflow
   postproduction_background
   postproduction_xgboost
   postproduction_trainingset
   postproduction_efficiency


Quick Start
-----------

A complete postproduction run is driven by a workflow YAML:

.. code-block:: bash

   pycwb post-process path/to/postprocess_workflow.yaml

To inspect the dependency graph without running:

.. code-block:: bash

   pycwb post-process path/to/postprocess_workflow.yaml --diagram-only

A reference template is available at
``examples/postproduction/standard_analysis_10pct_workflow.yaml``.


Postproduction Architecture
---------------------------

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


Typical Workflow Steps
----------------------

A complete postproduction analysis follows this sequence:

1. **Split background** into training and FAR-holdout subsets
   (:ref:`postproduction_trainingset`).
2. **Match and filter** simulation triggers to injection truth.
3. **Train XGBoost** ranking model on BKG + SIM features
   (:ref:`postproduction_xgboost`).
4. **Score** background holdout with the trained model.
5. **Build FAR** lookup table from scored background
   (:ref:`postproduction_background`).
6. **Score** simulations and compute **detection efficiency**
   (:ref:`postproduction_efficiency`).
7. **Analyze zero-lag** candidates and compute Poisson significance.
8. **Generate HTML report** with all results.

See :ref:`postproduction_workflow` for detailed YAML examples of each step.
