.. _postproduction_report:

Postproduction Reports
======================

.. stage-nav:: postproduction
   :current: report

This guide explains how pycWB turns FAR tables, candidates, training
diagnostics and efficiency tables into report files, and how the final
multi-tab HTML report is assembled, rendered and checked.

.. contents:: Table of Contents
   :depth: 2
   :local:


Why this matters
----------------

The report places livetimes, FAR calibration, candidates and sensitivity next
to the workflow YAML and production configuration behind them, so a reviewer
can check each number against the intended selection. Report actions only
read scored catalogs and earlier products.


Report Types
------------

Background FAR report
~~~~~~~~~~~~~~~~~~~~~

``postprocess.report.standard_background_report`` loads the binned FAR table
from ``far_rho_file`` (keys ``bins``, ``far``, ``cum_events``, ``livetime``,
``ranking_par``, as written by ``evaluate_far_rho``) or else builds it from
``catalog_file`` with ``postprocess.far.far_rho_plot`` (``bin_size`` default
0.1; ``livetime``, or that of the ``job_ids_file`` jobs in ``progress_file``),
then runs the zero-lag and fake open-box reports with it. ``output_dir``
(default ``public``, relative to ``work_dir``) receives ``far_rho.json`` (the
table used), ``far_rho.png``, ``far_rho_n_events.png`` and, on the
``far_rho_plot`` path, ``loudest_background_triggers.csv`` (10 loudest).

Defaults: ``include_zero_lag: true``, ``include_fake_openbox: false`` (true
requires ``fake_openbox_intervals_file``), ``exclude_zero_lag: true`` (no zero
lag in the FAR table or fake open box). Without ``zero_lag_catalog_file`` or
``zero_lag_job_ids_file``, zero lag is read from ``catalog_file`` and the
``job_ids_file`` jobs, so pass the zero-lag catalog explicitly.

Zero-lag report
~~~~~~~~~~~~~~~

``postprocess.zero_lag.zero_lag_report`` selects the zero-lag rows of
``catalog_file``, sums the zero-lag livetime of ``progress_file`` (both
optionally limited to ``job_ids_file``) and attaches FAR, IFAR, p-value and
significance (*Zero-Lag Significance* in :doc:`postproduction_background`).
It writes ``zero_lag_triggers.csv``, ``zero_lag_report.png`` (FAR vs. ranking
statistic over the background curve, significance histogram) and
``zero_lag_poisson.png`` (cumulative events vs. IFAR with Poisson bands); no
plots are made without triggers. With ``public_alerts_file`` (name and GPS
time per line), the trigger nearest each alert within
``public_alert_time_window`` s (default 1.0) is labelled a known candidate.
The FAR table is ``far_rho_data``, else a ``far_rho`` context value (e.g. the
result of step ``far_rho``), else ``<output_dir>/far_rho.json``.

Fake open-box report
~~~~~~~~~~~~~~~~~~~~

``postprocess.fake_openbox.fake_openbox_report`` draws ``fake_openbox_n``
(default 3, at most the number available) rows of ``intervals_file`` (CSV or
Parquet with ``shift_key``, ``lag_idx``, ``livetime``) with
``fake_openbox_seed`` (default 150914), collects the non-zero-lag triggers of
``catalog_file`` in each (superlag shift, lag) interval (``lag_idx`` and
``segment_lag_<IFO>`` columns are required) and reports each interval as if it
were zero lag, with its own livetime. It writes ``fake_openbox_intervals.csv``,
``fake_openbox_NN_triggers.csv``/``_report.png``/``_poisson.png`` (``NN`` =
01, 02, …) and the combined ``fake_openbox_triggers.csv``.

Diagnostics and standalone pages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``postprocess.training_report.training_diagnostics`` writes
  ``training.json`` (best iteration, evaluation history, feature gain),
  ``learning.png`` and ``importance.png`` (20 largest gains) for
  ``model_file``; with ``reference_model_file`` it records exact-equality
  checks and raises if one fails.
- ``postprocess.simulation_report.simulation_efficiency`` applies an explicit
  cut (``ranking_par`` default ``rho_alt``, ``comparison`` ``>``/``>=``,
  ``threshold`` default 0.0) to a right-matched ``matched_file``: it writes
  ``efficiency.csv`` (counts, 95% Wilson intervals), ``decisions.parquet``,
  ``summary.json`` (interpolated hrss50 or a bound), ``efficiency.png``; for
  fitted curves see :doc:`postproduction_efficiency`.
- ``postprocess.cwb_report.collect_comparisons`` merges the ``checks`` of
  comparison JSON files into one file (:doc:`postproduction_root`).
- ``postprocess.generic_report.generic_web_report`` copies HTML ``plots``
  (``title``, ``html_file``) into ``assets/`` beside ``output_file`` and embeds
  them as iframes (:doc:`tutorial_custom_postproduction`).
- ``postprocess.waveform_report.generate_waveform_report`` writes
  ``reports/plots/*.png`` and ``reports/results/*.npz`` next to ``wave.h5``.

Zero lag vs. fake open box
~~~~~~~~~~~~~~~~~~~~~~~~~~

The zero-lag report *is* the box opening. While blind, set
``include_zero_lag: false`` and ``include_fake_openbox: true`` and leave
``zero_lag_csv`` and ``zero_lag_*.png`` out of the final report (its zero-lag
livetime comes from progress rows, not triggers). Intervals drawn from the FAR
partition (``@<split>.far.intervals_csv_file``; *Blind Analysis (Fake Open
Box)* in :doc:`postproduction_background`) already count in the FAR table.


Assembling the Final Report
---------------------------

``postprocess.report_builder.postproduction_report`` requires
``workflow_file`` and ``production_catalog_file``; ``output_file`` defaults to
``public/postproduction_report/index.html`` and ``data_file`` to
``report_data.json`` beside it. Nested ``bkg``, ``training`` and
``simulation_runs`` point at earlier artifacts:

.. code-block:: yaml

   - id: postproduction_report
     action: postprocess.report_builder.postproduction_report
     inputs:
       workflow_file: ${paths.workflow_filename}
       production_catalog_file: ${paths.target_bkg_catalog}
     args:
       title: O4 K21 postproduction report
       output_file: ${paths.output_dir}/index.html   # report_data.json beside it
       bkg:
         scored_catalog: ${paths.bkg_far_scored}
         far_json: ${paths.far_rho_file}
         progress_file: "@k21_bkg_split.far.progress_file"
         intervals_file: "@k21_bkg_split.far.intervals_file"
         zero_lag_progress_file: ${paths.target_bkg_progress}
         zero_lag_csv: ${paths.output_dir}/zero_lag_triggers.csv   # omit while blind
         livetime: "@k21_bkg_split.far.livetime.seconds"
         ranking_par: rhor
         plots: [ "${paths.output_dir}/far_rho.png" ]
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
             - ${paths.output_dir}/simulations/efficiency_vs_hrss_by_waveform_100yr.png

Other keys: ``bkg.binned_far_json`` (read before ``far_json``),
``bkg.zero_lag_catalog_file``; ``training.plots``, ``training_settings_file``,
``training_output_file``, ``bkg_progress_file``, ``bkg_intervals_file``;
``simulation_runs[].fit_parameter_files``, ``efficiency_summary_file`` and
``efficiency_file`` (``summary.json``, ``efficiency.csv`` above). A ``plots``
entry is a path or a ``{path, label}`` mapping; ``.html`` files become
iframes, others images. Tabs: Summary, BKG, Training, Consistency checks (only
with ``validation_file`` from ``collect_comparisons``), one per
``simulation_runs`` entry, and Workflow / YAML (diagram and full YAML):

- **Summary**: catalog run parameters, postproduction selection, livetimes,
  review links, pycWB versions, and an Artifact Health table.
- **BKG**: declared ``plots``; interactive FAR vs. ranking statistic,
  cumulative count vs. IFAR, ranking statistic and FAR vs. GPS time and
  frequency; tables of loudest background events, zero-lag triggers and
  livetime by lag and interval.
- **Training**: min/median/max of ``rho``, ``xgb_prob``, ``net_cc``,
  ``likelihood``, ``coherent_energy``, ``rho``/``xgb_prob`` histograms,
  declared ``plots``, training workflow steps and the XGBoost config.
- **Simulation runs**: catalog statistics, matched counts, efficiency
  summary, ``plots``, counts by amplitude, fit tables.

The BKG livetime is ``bkg.livetime`` unless it differs by more than 5% from
the first available measurement (completed non-zero-lag ``progress_file``
rows, then ``intervals_file``, then the FAR JSON), which then replaces it with
a warning. Zero-lag livetime sums the completed zero-lag rows of
``zero_lag_progress_file``.

Everything is collected into one dictionary, written to ``data_file``,
rendered with the Jinja2 template ``postproduction_report.html.j2`` and
embedded in the page as JSON. Catalogs are reduced to a few columns and
``max_bins`` (80) bins, FAR curves to ``max_plot_points`` (2000) points and
tables to ``table_limit`` (50) rows. Missing inputs (except
``validation_file``) are listed, not fatal; the action returns
``output_file``, ``data_file``, ``n_tabs`` and ``missing_artifacts``.


Output Layout
-------------

With the reference workflow (``paths.output_dir: public/O4_K21_run1``):

.. code-block:: text

   <work_dir>/public/O4_K21_run1/
   ├── index.html  report_data.json    # final report and its data
   ├── <workflow>.yaml  workflow_diagram.html/.png   # copied inputs
   ├── far_rho.json  far_rho.png  far_rho_n_events.png
   ├── zero_lag_triggers.csv  zero_lag_report.png  zero_lag_poisson.png
   ├── fake_openbox_intervals.csv  fake_openbox_01_report.png  ...
   ├── models/                         # model, FAR table, XGB settings
   └── simulations/                    # scored SIM catalog, efficiency

Open ``index.html`` in a browser; the tab is kept in the URL fragment
(``#bkg``, ``#training``, ``#sim-stdinjs_set1`` for label ``STDINJs Set1``).
Interactive panels load Plotly from ``cdn.plot.ly``; offline, placeholders
replace them and static images remain. Artifacts are linked by relative path,
not copied (except the workflow YAML and diagram): moving the report directory
alone breaks links to files outside it, and ``cleanup_tmp: on_success`` or
``always`` deletes the ``tmp://`` products it links to.


Implementation
--------------

- :py:func:`pycwb.modules.postprocess.report.standard_background_report`
  chains :py:func:`pycwb.modules.postprocess.far.far_rho_plot`,
  :py:func:`pycwb.modules.postprocess.zero_lag.zero_lag_report` and
  :py:func:`pycwb.modules.postprocess.fake_openbox.fake_openbox_report`, using
  :py:func:`pycwb.modules.postprocess.far.attach_far_and_significance`,
  :py:func:`pycwb.modules.postprocess.lag_filters.zero_lag_mask` and the
  Matplotlib figures of :py:mod:`pycwb.modules.postprocess.report_plots`.
- :py:func:`pycwb.modules.postprocess.report_builder.postproduction_report`
  uses :py:class:`pycwb.modules.postprocess.report_context.ReportContext`
  (paths, artifacts), :py:mod:`~pycwb.modules.postprocess.report_summaries`
  (readers, figure data), :py:mod:`~pycwb.modules.postprocess.report_sections`
  (tabs) and :py:mod:`~pycwb.modules.postprocess.report_render` (template
  ``templates/postproduction_report.html.j2``).
- Standalone products: :py:mod:`pycwb.modules.postprocess.training_report`,
  :py:mod:`~pycwb.modules.postprocess.simulation_report`,
  :py:mod:`~pycwb.modules.postprocess.cwb_report`,
  :py:mod:`~pycwb.modules.postprocess.generic_report`,
  :py:mod:`~pycwb.modules.postprocess.waveform_report`.


.. raw:: html

   <span id="validation-checks"></span>

Inspect the report
------------------

After building the report, verify:

- **Livetimes agree**: with ``bkg.livetime`` set, the BKG tab reads "BKG live
  time source: explicit" with no warning, and the ``livetime`` in
  ``far_rho.json`` equals ``@<split>.far.livetime.seconds``.
- **No data warnings** in ``report_data.json``: ``bkg.far_curve``,
  ``bkg.scored_catalog`` and ``bkg.zero_lag_livetime`` (missing progress, or
  shifted jobs possibly counted as zero lag) carry no ``warning``.
- **One ranking statistic**: ``bkg.ranking_par`` (default ``rho``) matches
  the FAR, background-report and efficiency steps.
- **Blinding is respected**: while blind, the Summary shows "Zero lag:
  disabled" and the zero-lag table is empty. Summary selection entries are
  read from steps with ids ``bkg_split``, ``far_rho``, ``background_report``
  and ``model``; other ids leave them blank.
- **Nothing is missing**: the Missing Artifacts list is empty and
  ``fake_openbox_intervals.csv`` has ``fake_openbox_n`` rows.


----

**See also:** :doc:`postproduction_workflow` · :doc:`postproduction_actions` · :doc:`tutorial_custom_postproduction`
