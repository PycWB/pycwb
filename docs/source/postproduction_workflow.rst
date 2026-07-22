.. _postproduction_workflow:

Post-Production Workflow Guide
==============================

This guide explains how to write and run a pycWB post-production workflow.
Post-production starts after search jobs have produced catalog and progress
Parquet files.  A workflow YAML then selects analysis subsets, matches
simulation triggers to injection truth, trains or applies ranking models,
builds FAR products, computes simulation efficiency, and writes report
artifacts.

The main command is:

.. code-block:: bash

   pycwb post-process path/to/postprocess_workflow.yaml

To inspect the dependency graph without running any action:

.. code-block:: bash

   pycwb post-process path/to/postprocess_workflow.yaml --diagram-only

The user-facing template
``examples/postproduction/standard_analysis_10pct_workflow.yaml`` is a good
starting point for a complete background, simulation, FAR, efficiency, and
HTML-report workflow.

For a compact multi-run study, see
``examples/postproduction/angle_error_comparison_workflow.yaml``. It shows a
generic three-action pattern: combine and reindex an array of catalogs, pass
the combined data to interactive plot actions, then assemble every returned
HTML plot in one generic report page. Each input needs only a generic ``name``
and ``catalog_file``. The reader adds ``run_index`` and ``run_name`` columns
and also reads scheduled injections from catalog job metadata, so an injection
missed by the search is not lost from the denominator. Optional extra fields
are retained as metadata, without prescribing which factor the study varies.


What The Workflow Consumes
--------------------------

A complete post-production workflow usually consumes these files:

.. list-table::
   :header-rows: 1

   * - Artifact
     - Typical path
     - Purpose
   * - Background catalog
     - ``BKG/.../catalog/catalog.parquet``
     - Trigger rows used for training, FAR holdout, and zero-lag scoring.
   * - Background progress
     - ``BKG/.../catalog/progress.parquet``
     - Job/livetime rows used for train/FAR splits and reports.
   * - Simulation trigger catalog
     - ``SIM/.../catalog/catalog.parquet``
     - Recovered simulation triggers.
   * - Simulation progress
     - ``SIM/.../catalog/progress.parquet``
     - Job metadata for selecting a simulation training subset.
   * - Simulation summary
     - ``SIM/.../catalog/simulations.parquet``
     - One row per injected signal, generated before post-production.
   * - XGBoost config
     - ``XGB/.../xgb_config.py``
     - Feature and ranking-statistic configuration.
   * - Public alerts
     - ``public_alerts.txt``
     - Optional known-event candidates for the background report.

For simulation studies, build ``simulations.parquet`` before running the
workflow:

.. code-block:: bash

   pycwb simulation-summary user_parameters.yaml \
     --work-dir /path/to/sim/run \
     --output /path/to/sim/run/catalog/simulations.parquet

You can also run matching as a standalone command, but complete workflows
should prefer the explicit ``postprocess.matching.match_simulations`` action:

.. code-block:: bash

   pycwb match-simulations catalog.parquet simulations.parquet \
     --how right --output matched_right.parquet


YAML Structure
--------------

A workflow has three top-level sections:

``vars``
   User-editable values and reusable paths.  Put run names, chunks, split
   fractions, output locations, and input file paths here.

``runtime``
   Execution settings owned by the workflow runner, especially the temporary
   directory and cleanup policy.

``steps``
   Ordered actions.  Each step declares an ``id``, human-readable ``name``,
   action path, ``inputs``, optional ``args``, and ``outputs``.

Minimal shape:

.. code-block:: yaml

   vars:
     work_dir: .
     paths:
       bkg_catalog: BKG/run/catalog/catalog.parquet

   runtime:
     tmp_dir: ${work_dir}/tmp/postprod
     cleanup_tmp: never

   steps:
     - id: example_step
       name: Example Step
       action: postprocess.selection.trigger_selection
       inputs:
         catalog_file: ${paths.bkg_catalog}
         progress_file: BKG/run/catalog/progress.parquet
       args:
         returns: [jobs, triggers, livetime]
       outputs:
         triggers_file: tmp://selected.parquet


Variables, References, And Temporary Files
------------------------------------------

Use ``${...}`` to expand values from ``vars``.  Dotted paths are supported:

.. code-block:: yaml

   vars:
     work_dir: .
     chunks:
       train_bkg_1: K20
       train_bkg_2: K22
       target: K21
       tag: k20_k22_k21
     train_fraction_label: 10pct
     run:
       output_name: O4_${chunks.target}_run1
       analysis_slug: bkg_${chunks.tag}_${train_fraction_label}
     paths:
       output_dir: public/${run.output_name}
       model_file: ${paths.output_dir}/models/${run.analysis_slug}_xgb_model_blf.ubj

When an entire value is one variable reference, its YAML type is preserved.
Use this to make structured action inputs explicit and reviewable rather than
relying on values inherited implicitly from the workflow context:

.. code-block:: yaml

   vars:
     runs:
       - name: reference
         catalog_file: first/catalog.parquet
       - name: comparison
         catalog_file: second/catalog.parquet

   steps:
     - id: read_runs
       action: postprocess.multi_run.read_catalog_runs
       inputs:
         runs: ${runs}  # remains a list, not a string

Use ``@step.path`` to read values returned by earlier steps:

.. code-block:: yaml

   catalog_file: "@k21_bkg_split.far.triggers_file"
   livetime: "@k21_bkg_split.far.livetime.seconds"

Use ``tmp://`` for intermediate files.  The workflow runner resolves those
paths under ``runtime.tmp_dir``:

.. code-block:: yaml

   runtime:
     tmp_dir: ${work_dir}/tmp/${run.analysis_slug}

   outputs:
     output_file: tmp://bkg_far_scored.parquet

Cleanup policies are:

``never``
   Keep temporary files.  This is best while developing or debugging.

``on_success``
   Remove ``runtime.tmp_dir`` after a successful workflow.

``always``
   Remove ``runtime.tmp_dir`` even if a later step fails.


Recommended Complete Workflow
-----------------------------

The standard 10 percent workflow has six phases.  The names below match the
template and test workflows, but the same pattern works for other chunks or
fractions.

1. Split Background
~~~~~~~~~~~~~~~~~~~

Split the target background into a training subset and a FAR holdout.  Use
``interval_livetime`` when the same physical jobs can appear in different lag
or shift intervals; this keeps train and FAR livetime disjoint by interval.

.. code-block:: yaml

   - id: k21_bkg_split
     name: Split Target BKG Train/FAR
     action: postprocess.selection.trigger_selection
     inputs:
       catalog_file: ${paths.target_bkg_catalog}
       progress_file: ${paths.target_bkg_progress}
     args:
       exclude_zero_lag: true
       returns: [jobs, triggers, livetime]
       split:
         by: interval_livetime
         seed: 42
         fractions:
           train: ${k21_train_fraction}
           far: ${k21_far_fraction}
     outputs:
       train:
         jobs_file: tmp://k21_bkg_train_jobs.txt
         progress_file: tmp://k21_bkg_train_progress.parquet
         intervals_file: tmp://k21_bkg_train_intervals.parquet
         intervals_csv_file: tmp://k21_bkg_train_intervals.csv
         triggers_file: tmp://k21_bkg_train.parquet
       far:
         jobs_file: tmp://k21_bkg_far_jobs.txt
         progress_file: tmp://k21_bkg_far_progress.parquet
         intervals_file: tmp://k21_bkg_far_intervals.parquet
         intervals_csv_file: tmp://k21_bkg_far_intervals.csv
         triggers_file: tmp://k21_bkg_far.parquet

2. Match And Filter Simulation Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Match the target simulation training trigger catalog to ``simulations.parquet``.
Use ``how: outer`` for training cleanup so the matched table includes recovered
and unrecovered simulation truth rows.  Then keep recovered, non-vetoed, real
simulation triggers for classifier training.

.. code-block:: yaml

   - id: k21_sim_train_match
     name: Match Target SIM Training
     action: postprocess.matching.match_simulations
     inputs:
       catalog_file: ${paths.target_sim_train_catalog}
       simulation_file: ${paths.target_sim_train_simulations}
     args:
       how: outer
       window_buffer: 0.0
     outputs:
       output_file: tmp://k21_sim_train_matched_outer.parquet

   - id: k21_sim_train_real
     name: Filter Target Real SIM Training
     action: postprocess.selection.filter_real_simulation
     inputs:
       matched_file: "@k21_sim_train_match.matched_file"
       sim_catalog: ${paths.target_sim_train_catalog}
     args:
       require_recovered: true
       exclude_vetoed: true
       output_schema: matched
     outputs:
       output_file: tmp://k21_sim_training_real.parquet

3. Select Simulation Training Fraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Select the target simulation training fraction after filtering.  Keep
``exclude_zero_lag: false`` for simulation catalogs.

.. code-block:: yaml

   - id: k21_sim_train_select
     name: Select Target SIM Training Fraction
     action: postprocess.selection.trigger_selection
     inputs:
       catalog_file: "@k21_sim_train_real.triggers_file"
       progress_file: ${paths.target_sim_train_progress}
     args:
       exclude_zero_lag: false
       returns: [jobs, triggers, livetime]
       selection:
         fraction: ${k21_train_fraction}
         seed: 43
     outputs:
       jobs_file: tmp://k21_sim_train_jobs.txt
       triggers_file: tmp://k21_sim_train.parquet

4. Train And Build FAR
~~~~~~~~~~~~~~~~~~~~~~

Train the ranking model with the full training chunks plus the selected target
training subset.  Then score the target FAR holdout and build the FAR lookup.

.. code-block:: yaml

   - id: model
     name: Train XGBoost
     action: postprocess.train_xgboost.train_xgboost
     inputs:
       bkg_catalogs:
         - ${paths.train_bkg_1_catalog}
         - ${paths.train_bkg_2_catalog}
         - "@k21_bkg_split.train.triggers_file"
       sim_catalogs:
         - ${paths.train_sim_1_catalog}
         - ${paths.train_sim_2_catalog}
         - "@k21_sim_train_select.triggers_file"
       config_file: ${paths.config_file}
     args:
       model_file: ${paths.model_file}
       dump: false
       dump_training_review: true
       verbose: false
     outputs:
       training_settings_file: ${paths.xgb_training_settings}
       training_output_file: ${paths.xgb_training_output}

   - id: far_rho
     name: Score Target Holdout BKG And Build FAR
     action: postprocess.evaluate.evaluate_far_rho
     inputs:
       catalog_file: "@k21_bkg_split.far.triggers_file"
       model_file: ${paths.model_file}
       config_file: ${paths.config_file}
     args:
       livetime: "@k21_bkg_split.far.livetime.seconds"
       ranking_par: rhor
       bin_size: 0.0001
       vmin: 0.0
       vmax: 10.0
     outputs:
       output_file: ${paths.far_rho_file}
       scored_catalog: ${paths.bkg_far_scored}

5. Score Evaluation Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Select and score the simulation evaluation catalog.  Match it with ``how:
right`` when computing efficiency so the matched table has one row per
simulation, including missed injections.

.. code-block:: yaml

   - id: sim_eval_selection
     name: Select SIM Evaluation
     action: postprocess.selection.trigger_selection
     inputs:
       catalog_file: ${paths.sim_eval_catalog}
       progress_file: ${paths.sim_eval_progress}
     args:
       exclude_zero_lag: false
       returns: [jobs, triggers, livetime]
       selection:
         fraction: 1.0
         seed: 44
     outputs:
       jobs_file: tmp://sim_eval_jobs.txt
       triggers_file: tmp://sim_eval.parquet

   - id: sim_eval_match
     name: Match SIM Evaluation
     action: postprocess.matching.match_simulations
     inputs:
       catalog_file: ${paths.sim_eval_catalog}
       simulation_file: ${paths.sim_eval_simulations}
     args:
       how: right
       window_buffer: 0.0
     outputs:
       output_file: tmp://sim_eval_matched_right.parquet

   - id: sim_efficiency_score
     name: Score SIM Evaluation
     action: postprocess.evaluate.evaluate_efficiency
     inputs:
       catalog_file: "@sim_eval_selection.triggers_file"
       model_file: ${paths.model_file}
       config_file: ${paths.config_file}
     args:
       threshold: 0.5
     outputs:
       output_file: ${paths.sim_eval_scored}

   - id: waveform_hrss_curves_100yr
     name: Waveform HRSS Curves At 100 Years
     action: postprocess.plot_efficiency.compute_efficiency_vs_hrss_by_waveform
     inputs:
       sim_catalog: ${paths.sim_eval_catalog}
       matched_file: "@sim_eval_match.matched_file"
       bkg_catalog: ${paths.bkg_far_scored}
       model_file: ${paths.model_file}
       config_file: ${paths.config_file}
     args:
       livetime: "@k21_bkg_split.far.livetime.seconds"
       ifar: 100yr
       use_unique_sim: true
       exclude_vetoed: false
     outputs:
       output_file: ${paths.output_dir}/simulations/${run.analysis_slug}_efficiency_vs_hrss_by_waveform_100yr.png
       fit_parameters_file: ${paths.output_dir}/simulations/${run.analysis_slug}_fit_parameters_by_waveform_100yr.csv

6. Build Reports
~~~~~~~~~~~~~~~~

Score zero lag if needed, produce the standard background report, optionally
score MDC detections at an IFAR threshold, and aggregate everything into the
post-production HTML report.

.. code-block:: yaml

   - id: zero_lag_score
     name: Score Target Zero Lag BKG
     action: postprocess.evaluate.score_catalog
     inputs:
       catalog_file: ${paths.target_bkg_catalog}
       model_file: ${paths.model_file}
       config_file: ${paths.config_file}
     args:
       lag_selection: zero_lag
     outputs:
       output_file: ${paths.bkg_zero_lag_scored}

   - id: background_report
     name: Standard Background Report
     action: postprocess.report.standard_background_report
     inputs:
       catalog_file: ${paths.bkg_far_scored}
       zero_lag_catalog_file: ${paths.bkg_zero_lag_scored}
       fake_openbox_intervals_file: "@k21_bkg_split.far.intervals_csv_file"
       progress_file: ${paths.target_bkg_progress}
       job_ids_file: "@k21_bkg_split.far.jobs_file"
     args:
       livetime: "@k21_bkg_split.far.livetime.seconds"
       ranking_par: rhor
       exclude_zero_lag: true
       far_rho_file: ${paths.far_rho_file}
       output_dir: ${paths.output_dir}
       include_zero_lag: true
       include_fake_openbox: true

   - id: postproduction_report
     name: Build Postproduction Report
     action: postprocess.report_builder.postproduction_report
     inputs:
       workflow_file: ${paths.workflow_filename}
       production_catalog_file: ${paths.target_bkg_catalog}
     args:
       title: O4 ${chunks.train_bkg_1}+${chunks.train_bkg_2} full plus ${chunks.target} ${train_fraction_label} BKG training postproduction report
       output_file: ${paths.output_dir}/index.html
       data_file: ${paths.output_dir}/report_data.json
       bkg:
         scored_catalog: ${paths.bkg_far_scored}
         far_json: ${paths.far_rho_file}
         progress_file: "@k21_bkg_split.far.progress_file"
         intervals_file: "@k21_bkg_split.far.intervals_file"
         zero_lag_progress_file: ${paths.target_bkg_progress}
         zero_lag_catalog_file: ${paths.bkg_zero_lag_scored}
         livetime: "@k21_bkg_split.far.livetime.seconds"
         ranking_par: rhor
       training:
         bkg_catalog: "@k21_bkg_split.train.triggers_file"
         sim_catalog: "@k21_sim_train_select.triggers_file"
         model_file: ${paths.model_file}
         config_file: ${paths.config_file}
       simulation_runs:
         - label: STDINJs Set1
           scored_catalog: ${paths.sim_eval_scored}
           matched_file: "@sim_eval_match.matched_file"
           plots:
             - ${paths.output_dir}/simulations/${run.analysis_slug}_efficiency_vs_hrss_by_waveform_100yr.png


Changing Chunks Or Fractions
----------------------------

For a chunk-swap workflow, keep the paths derived from a small set of values:

.. code-block:: yaml

   train_fraction_label: 10pct

   chunks:
     train_bkg_1: K20
     train_bkg_2: K22
     target: K21
     tag: k20_k22_k21

   run:
     output_name: O4_${chunks.target}_run1
     analysis_slug: bkg_${chunks.tag}_${train_fraction_label}

To use K23 instead of K22, update:

.. code-block:: yaml

   chunks:
     train_bkg_1: K20
     train_bkg_2: K23
     target: K21
     tag: k20_k23_k21

The derived catalog paths, temporary directory, model name, FAR JSON, plots,
and report paths will update together.

If you change the training fraction, update all three values together:

.. code-block:: yaml

   k21_train_fraction: 0.3
   k21_far_fraction: 0.7
   train_fraction_label: 30pct


Common Actions
--------------

The actions below are the main building blocks used by the standard workflow.
See :ref:`postproduction_actions` for the complete registered-action catalog,
exact signatures, and guidance on composing actions safely.

``postprocess.selection.trigger_selection``
   Selects or splits trigger catalogs by jobs or intervals.  Use it for BKG
   train/FAR splits, SIM training fractions, and full SIM evaluation
   selections.

``postprocess.matching.match_simulations``
   Matches a trigger catalog to ``simulations.parquet``.  Use ``how: outer``
   before SIM training filters and ``how: right`` for efficiency curves.

``postprocess.selection.filter_real_simulation``
   Keeps recovered simulation triggers and optionally removes vetoed or
   across-segment injections.

``postprocess.train_xgboost.train_xgboost``
   Trains the XGBoost ranking model from BKG and SIM catalog lists.

``postprocess.evaluate.evaluate_far_rho``
   Scores the FAR holdout and writes the FAR-vs-ranking lookup.

``postprocess.evaluate.score_catalog``
   Scores a catalog, commonly the target zero-lag background.

``postprocess.evaluate.evaluate_efficiency``
   Scores simulation evaluation triggers and writes a scored catalog.

``postprocess.plot_efficiency.compute_efficiency_vs_hrss_by_waveform``
   Builds waveform-level efficiency curves using a matched simulation table.

``postprocess.report.standard_background_report``
   Writes the standard background plots, fake-open-box outputs, and zero-lag
   report products.

``postprocess.evaluate.score_mdc_catalog``
   Scores MDC detections against a background IFAR threshold.

``postprocess.report_builder.postproduction_report``
   Aggregates workflow artifacts into the final HTML and JSON report.


Validation And Debugging
------------------------

Start with a diagram-only dry run:

.. code-block:: bash

   pycwb post-process postprocess_workflow.yaml --diagram-only

Useful checks before a full run:

* All ``${...}`` variables resolve to the intended files.
* Every ``@step.path`` reference points to an earlier step.
* ``runtime.tmp_dir`` is unique for this analysis.
* ``how: outer`` is used for SIM training matching.
* ``how: right`` is used for SIM evaluation matching.
* BKG selections use ``exclude_zero_lag: true``.
* SIM selections use ``exclude_zero_lag: false``.
* Report paths point to the same files produced by earlier steps.

Common failure modes:

Missing ``simulations.parquet``
   Run ``pycwb simulation-summary`` for the simulation production, then rerun
   post-production.

``duckdb`` import error
   Install the catalog matching dependency in the active environment.

Reference resolution error
   Check that the producing step has an ``id`` and that the referenced key is
   returned by that action.

Unexpected train/FAR overlap
   Use ``split.by: interval_livetime`` and make sure the input catalog carries
   job shift metadata.

Temporary file missing
   Keep ``cleanup_tmp: never`` while debugging so intermediate parquet files
   remain inspectable.


Outputs To Expect
-----------------

A complete workflow typically produces:

* trained model under ``${paths.output_dir}/models/``;
* XGBoost training settings and training output logs;
* FAR JSON lookup table;
* scored FAR and zero-lag background catalogs;
* scored simulation evaluation catalog;
* waveform efficiency plot and fit-parameter CSV;
* background report plots;
* optional MDC detection CSV;
* final ``index.html`` and ``report_data.json``.

Keep the workflow YAML with the report output.  It records the exact inputs,
fractions, seeds, matching modes, and artifact paths used for the analysis.
