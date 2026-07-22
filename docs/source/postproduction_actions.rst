.. _postproduction_actions:

Post-Production Action Reference
================================

This page is the canonical catalog of actions that can be used in the
``steps`` section of a post-production workflow.  It lists every function in
``pycwb.modules.postprocess`` registered with ``@action_spec``.  For a
complete pipeline assembled from these actions, start with
:ref:`postproduction_workflow`.


Choosing The Main Actions
-------------------------

Most production workflows need only the following path through the catalog:

.. list-table::
   :header-rows: 1
   :widths: 22 34 44

   * - Pipeline stage
     - Recommended action
     - Important choice
   * - Split background
     - ``postprocess.selection.trigger_selection``
     - Use ``split.by: interval_livetime`` when jobs contain multiple lag or
       shift intervals.  Exclude zero lag from background.
   * - Prepare simulations
     - ``postprocess.matching.match_simulations``
     - Use ``how: outer`` before training cleanup and ``how: right`` when
       missed injections must remain in the efficiency denominator.
   * - Clean simulation training data
     - ``postprocess.selection.filter_real_simulation``
     - Require recovered events and decide explicitly whether vetoed events
       are excluded.
   * - Train a ranking model
     - ``postprocess.train_xgboost.train_xgboost``
     - Keep the model and its XGBoost configuration together; scoring must use
       the same features and ranking definitions.
   * - Score the FAR holdout
     - ``postprocess.evaluate.evaluate_far_rho``
     - Pass only disjoint holdout background and its matching live time.
   * - Score evaluation simulations
     - ``postprocess.evaluate.evaluate_efficiency``
     - Do not use simulation events that trained the model.
   * - Build efficiency products
     - ``postprocess.plot_efficiency.compute_efficiency_vs_hrss_by_waveform``
     - Use a right-matched simulation table so non-recoveries are counted.
   * - Report background and candidates
     - ``postprocess.report.standard_background_report``
     - Give zero-lag and fake-open-box inputs explicitly when enabling those
       report sections.
   * - Assemble the final report
     - ``postprocess.report_builder.postproduction_report``
     - Point its nested sections to artifacts already produced by earlier
       steps.


How To Use An Action
--------------------

The YAML action name is the module and function below
``pycwb.modules``.  Use the short form shown throughout this page:

.. code-block:: yaml

   steps:
     - id: far
       name: Score Background Holdout
       action: postprocess.evaluate.evaluate_far_rho
       inputs:
         catalog_file: "@split.far.triggers_file"
         model_file: ${paths.model_file}
         config_file: ${paths.config_file}
       args:
         livetime: "@split.far.livetime.seconds"
         ranking_par: rhor
         exclude_zero_lag: true
       outputs:
         output_file: ${paths.far_rho_file}
         scored_catalog: ${paths.bkg_far_scored}

``inputs``
   Data artifacts consumed by the action.  Put upstream catalogs, models,
   progress files, and manifests here.

``args``
   Behavior and scientific choices such as fractions, seeds, ranking
   parameters, thresholds, and matching modes.

``outputs``
   Destinations for artifacts written by the action, when the function
   signature accepts those destination names.  A flat mapping is passed both
   as named keyword arguments and as ``outputs``.  Nested output mappings are
   passed only as ``outputs``; this form is used by ``trigger_selection`` for
   named split partitions.  Some actions instead create derived or fixed-name
   products and expose their paths only in the returned dictionary.

The runner combines those three mappings into keyword arguments, so the
function signatures below are the authoritative spelling and type reference.
Most actions accept ``**kwargs`` for workflow context; consequently, a typo
in an argument may not fail immediately.  Copy parameter names exactly and
check the generated diagram before a long run.


Data Flow And Paths
-------------------

Give every reusable step a unique ``id``.  An action's returned dictionary is
stored below that ID, and a later step can read it with ``@step.path``:

.. code-block:: yaml

   inputs:
     catalog_file: "@far.scored_catalog"

The reference is to a **returned key**, not merely to the name written in the
earlier step's ``outputs`` block.  The steps still execute in YAML order; the
DAG diagram visualizes dependencies but does not reorder them.

Use ``${name}`` or ``${nested.name}`` for values under ``vars``.  A whole-value
reference preserves lists and mappings, while an embedded reference becomes
text.  Use ``tmp://name.parquet`` for intermediates under ``runtime.tmp_dir``.
Ordinary relative paths are interpreted by the individual action, normally
relative to ``work_dir``.  Keep ``cleanup_tmp: never`` until the workflow is
validated, then choose ``on_success`` for routine production.


Scientific And Reproducibility Checks
-------------------------------------

Before running a production pipeline, verify all of the following:

* Background used for FAR is disjoint from model training data, and its live
  time describes exactly the selected rows or intervals.
* Zero lag is excluded from background training and FAR estimation, but is
  selected deliberately for candidate evaluation.
* Simulation training and evaluation sets are disjoint.  Missed injections
  remain in efficiency denominators.
* Model scoring uses the same ``config_file``, feature definitions, detector
  count, and search type as training.
* Random selections have recorded seeds, fractions, job lists, and interval
  tables.  Keep the workflow YAML beside the final report.
* Every ``@step.path`` points backward to a key actually returned by that
  action.  Run ``pycwb post-process workflow.yaml --diagram-only`` before the
  full workflow.


Complete Action Catalog
-----------------------

Selection, filtering, and matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Action
     - Use and crucial information
   * - ``postprocess.selection.trigger_selection``
     - Preferred high-level selector.  Select or split by jobs or by
       interval live time and optionally materialize jobs, progress,
       intervals, triggers, and live-time summaries.
   * - ``postprocess.selection.filter_real_simulation``
     - Filter a matched table to recovered simulation triggers for training;
       control veto removal and whether output uses matched or catalog schema.
   * - ``postprocess.matching.match_simulations``
     - Match recovered triggers with ``simulations.parquet``.  The ``how``
       join mode changes whether missed injections survive.
   * - ``postprocess.job_selector.select_jobs_by_livetime``
     - Lower-level legacy building block that writes a randomized job-ID list
       approximating a requested live-time fraction.
   * - ``postprocess.job_selector.filter_catalog_by_jobs``
     - Lower-level building block that filters a parquet catalog by a job-ID
       list.
   * - ``postprocess.job_selector.compute_livetime``
     - Lower-level building block that sums progress live time for selected
       jobs and returns the result as data rather than a file.
   * - ``postprocess.random_filter.random_filter_parquet``
     - Simple row subsampling and optional zero-lag removal.  Prefer
       ``trigger_selection`` when whole-job or interval-level independence is
       scientifically required.
   * - ``postprocess.multi_run.read_catalog_runs``
     - Combine an array of catalogs using a generic ``name`` for each run, add
       stable run identity, reindex events, and optionally extract scheduled
       injections and a run manifest. Extra run fields remain arbitrary
       metadata rather than prescribed study factors.

Training, scoring, and FAR
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Action
     - Use and crucial information
   * - ``postprocess.train_xgboost.train_xgboost``
     - Train one XGBoost model from one or more pre-filtered background and
       simulation catalogs; save the model and optional training-review
       artifacts.
   * - ``postprocess.evaluate.score_catalog``
     - Apply a trained model and configured ranking expressions to a catalog.
       ``lag_selection`` can restrict scoring to zero lag or non-zero lag.
   * - ``postprocess.evaluate.evaluate_far_rho``
     - Score background and build the FAR-versus-ranking lookup.  The supplied
       live time and background selection define the statistical exposure.
   * - ``postprocess.evaluate.evaluate_efficiency``
     - Score a simulation catalog and compute threshold efficiency.  Use the
       waveform actions below for IFAR-dependent injection efficiency.
   * - ``postprocess.evaluate.score_mdc_catalog``
     - Score a blind MDC catalog against a scored background and write events
       above an IFAR threshold.
   * - ``postprocess.far.far_rho_plot``
     - Compute and plot FAR directly from an already-ranked catalog.  Use
       ``evaluate_far_rho`` when XGBoost scoring is also required.

Efficiency products
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Action
     - Use and crucial information
   * - ``postprocess.plot_efficiency.compute_hrss50``
     - Compute the amplitude at 50 percent efficiency for one IFAR threshold.
   * - ``postprocess.plot_efficiency.plot_efficiency_vs_hrss``
     - Plot the aggregate efficiency-versus-amplitude curve for one IFAR.
   * - ``postprocess.plot_efficiency.compute_efficiency_by_waveform``
     - Compute detection efficiency separately for each waveform using a
       right-matched simulation table.
   * - ``postprocess.plot_efficiency.compute_efficiency_vs_hrss_by_waveform``
     - Plot waveform and Q-factor efficiency curves and optionally write fit
       parameters.  This is the usual complete-workflow choice.
   * - ``postprocess.plot_efficiency.compute_hrss50_by_waveform_csv``
     - Write per-waveform ``hrss50`` values for several comma-separated IFAR
       thresholds.

Reports and specialized studies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Action
     - Use and crucial information
   * - ``postprocess.zero_lag.zero_lag_report``
     - Attach FAR and Poisson significance to zero-lag triggers and create
       candidate plots.
   * - ``postprocess.fake_openbox.fake_openbox_report``
     - Sample background intervals reproducibly and present them with
       open-box-style significance for validation.
   * - ``postprocess.report.standard_background_report``
     - Composite background action combining FAR products with optional zero
       lag and fake-open-box sections.
   * - ``postprocess.report_builder.postproduction_report``
     - Build the standard multi-tab HTML and JSON report from background,
       training, and simulation artifacts.
   * - ``postprocess.generic_report.generic_web_report``
     - Package arbitrary interactive HTML plots from upstream actions into a
       portable single-page report.
   * - ``postprocess.angle_comparison.plot_angle_error_comparison``
     - Compare injection truth with sky recovery across multiple labeled runs
       and write interactive map, histogram, summary, and event data. Clicking
       a shared injection on the map highlights its matching recoveries and
       great-circle paths while dimming unrelated events and paths.
   * - ``postprocess.waveform_report.generate_waveform_report``
     - Generate aggregate waveform-reconstruction quality products for one or
       more interferometers from a waveform file.


Action Signatures
-----------------

The signatures and docstrings below come directly from the implementation.
Parameters without defaults are required unless an earlier workflow context
value supplies them.

Selection, filtering, and matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pycwb.modules.postprocess.selection.trigger_selection

.. autofunction:: pycwb.modules.postprocess.selection.filter_real_simulation

.. autofunction:: pycwb.modules.postprocess.matching.match_simulations

.. autofunction:: pycwb.modules.postprocess.job_selector.select_jobs_by_livetime

.. autofunction:: pycwb.modules.postprocess.job_selector.filter_catalog_by_jobs

.. autofunction:: pycwb.modules.postprocess.job_selector.compute_livetime

.. autofunction:: pycwb.modules.postprocess.random_filter.random_filter_parquet

.. autofunction:: pycwb.modules.postprocess.multi_run.read_catalog_runs

Training, scoring, and FAR
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pycwb.modules.postprocess.train_xgboost.train_xgboost

.. autofunction:: pycwb.modules.postprocess.evaluate.score_catalog

.. autofunction:: pycwb.modules.postprocess.evaluate.evaluate_far_rho

.. autofunction:: pycwb.modules.postprocess.evaluate.evaluate_efficiency

.. autofunction:: pycwb.modules.postprocess.evaluate.score_mdc_catalog

.. autofunction:: pycwb.modules.postprocess.far.far_rho_plot

Efficiency products
~~~~~~~~~~~~~~~~~~~

.. autofunction:: pycwb.modules.postprocess.plot_efficiency.compute_hrss50

.. autofunction:: pycwb.modules.postprocess.plot_efficiency.plot_efficiency_vs_hrss

.. autofunction:: pycwb.modules.postprocess.plot_efficiency.compute_efficiency_by_waveform

.. autofunction:: pycwb.modules.postprocess.plot_efficiency.compute_efficiency_vs_hrss_by_waveform

.. autofunction:: pycwb.modules.postprocess.plot_efficiency.compute_hrss50_by_waveform_csv

Reports and specialized studies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pycwb.modules.postprocess.zero_lag.zero_lag_report

.. autofunction:: pycwb.modules.postprocess.fake_openbox.fake_openbox_report

.. autofunction:: pycwb.modules.postprocess.report.standard_background_report

.. autofunction:: pycwb.modules.postprocess.report_builder.postproduction_report

.. autofunction:: pycwb.modules.postprocess.generic_report.generic_web_report

.. autofunction:: pycwb.modules.postprocess.angle_comparison.plot_angle_error_comparison

.. autofunction:: pycwb.modules.postprocess.waveform_report.generate_waveform_report
