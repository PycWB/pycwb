.. _postproduction_root:

cWB ROOT input and same-trigger cross-checks
============================================

The ROOT adapter makes cWB background results usable by native postproduction
without rerunning production. It reads ``waveburst`` and ``liveTime`` through
``uproot`` (install with ``pip install uproot``); PyROOT is needed only for the
independent reference check. Supply detector names in **cWB network order**.

In-memory processing
--------------------

.. code-block:: python

   from pycwb.modules.postprocess.root_adapter import read_cwb_root
   from pycwb.modules.postprocess.background import process_background

   inputs = read_cwb_root("wave.root", ["L1", "H1"])
   result = process_background(
       inputs.triggers, inputs.progress,
       ranking_par="rho_alt",  # cWB rho[1]; rho is rho[0]
       trigger_query="net_cc >= 0.7",
       thresholds=[5., 6., 7., 8.],
       comparison=">",         # cWB report's strict threshold convention
   )
   print(result["livetime"], result["curve"])  # seconds and FAR in Hz

``process_background`` accepts Arrow tables, pandas DataFrames or Parquet
paths. Native pycWB and ROOT-adapted tables use the same implementation.
It uses triggers and exposure as given, including zero lag, and logs a warning
when triggers have no time or segment shift. Select background first, as in
the next section, to match a cWB background report.
``comparison=">="`` remains the native cumulative-rate default. This does not
change the existing histogram-based ``far_rho_plot`` action; compare explicit
threshold grids using ``process_background`` when testing cWB boundaries.
Event-quality cuts do not subtract exposure. Apply time vetoes consistently
to both input events and livetime before this step.

Parquet and workflow integration
--------------------------------

.. code-block:: python

   from pycwb.modules.postprocess.selection import trigger_selection

   paths = inputs.write("converted")
   # catalog.parquet and progress.parquet contain matching job metadata, so
   # selection can identify zero lag and write the matching exposure.
   trigger_selection(
       ".", paths["catalog_file"], paths["progress_file"], exclude_zero_lag=True,
       outputs={"triggers_file": "converted/background.parquet",
                "progress_file": "converted/background_progress.parquet"},
   )
   result = process_background(
       "converted/background.parquet", "converted/background_progress.parquet",
       thresholds=[5., 6., 7., 8.], comparison=">",
   )

The resulting catalog uses the native flat Trigger schema, with extra source
and plugin columns. Existing ``postprocess.selection.trigger_selection`` and
catalog scoring actions can consume it. Score it with the **same frozen model,
feature configuration and cuts** as the reference; conversion itself does not
validate model parity. Missing optional Q-veto values are null, not measured
zeros. Verify that all fields required by the chosen model exist in the input.

The workflow action ``postprocess.root_adapter.import_cwb_root`` accepts
``work_dir``, ``wave_files``, ``ifo_list``, optional ``live_files``, and optional
``output_dir``. It returns ``triggers`` and ``progress`` DataFrames plus ``jobs``;
when an output directory is given it also returns ``catalog_file`` and
``progress_file``. Files are resolved relative to ``work_dir``.

Data contracts
--------------

* ``lag[nIFO]`` and ``slag[nIFO]`` are network indices. The preceding entries
  are detector offsets; trailing padding in liveTime is ignored.
* One native job is assigned per (cWB run, superlag). Original run and superlag
  indices remain in ``root_run`` and ``root_slag_idx``. Job metadata carries
  detector shifts so existing zero-lag filters work with superlags.
* Exposure comes from every liveTime row, including lags without events.
  Duplicate exposure keys, events without exposure, inconsistent offsets,
  nonfinite/negative exposure and wrong detector counts are errors.
* Use one campaign per call. Independent campaigns with overlapping run IDs
  must be converted separately. Conversion currently materializes the tables
  in memory, although ROOT reading is batched.
* ``root_file`` and ``root_entry`` identify the source event. IDs are stable
  for the same absolute source path and entry, and differ when a file moves.
* This is a background adapter. It does not construct injection truth tables,
  veto intervals, interval-splitting metadata, or search configuration from
  ROOT. Do not infer simulation efficiency from converted triggers alone.

Independent reference
---------------------

``examples/cwb_results_conversion/check_background_consistency.py`` invokes
``root_background_reference.py`` in a separate ROOT Python environment. The
reference imports no pycWB code. It evaluates event membership, exposure and
strict threshold counts directly with ROOT, following the all-background
lag exclusion and ``rho > Xcut`` convention in ``cwb_report_prod_2.C``.
It is a reference calculation, **not a run of the full cWB report macro**.

.. code-block:: bash

   python examples/cwb_results_conversion/check_background_consistency.py \
       --wave /data/wave.root --ifo L1 H1 \
       --root-python /path/to/root/environment/bin/python \
       --output /data/postproduction-check \
       --rho-index 1 \
       --root-cut 'netcc[0]>=0.7 && rho[1]>=8' \
       --query 'net_cc >= 0.7 and rho_alt >= 8'

Use equivalent ROOT and native cut expressions. The check compares source
entry membership, completed livetime, counts and FAR on the same grid,
including thresholds exactly equal to observed rho values. It also checks
in-memory versus Parquet equality and the existing native selection action.
It saves input hashes, ROOT version, reference output, selected entries,
rate curves and a summary. Assertion failures terminate the check.

These comparisons use the same input triggers, so differences identify
postproduction behavior rather than differences between searches.

Simulations, training and standard-command validation
-----------------------------------------------------

``postprocess.root_simulation.import_cwb_simulation`` reads the merged ``waveburst``
and ``mdc`` trees. It associates triggers with their stored injection time, type
and factor, chooses the loudest reconstruction before the recovery-time cut,
and retains every MDC injection in ``matched.parquet``. ``unique.parquet`` can
be used for cWB-compatible training; ``catalog.parquet`` contains time-matched
recoveries. Campaigns with ambiguous truth keys or multiple superlag recoveries
per injection must be split rather than silently double-counted.

``postprocess.simulation_report.simulation_efficiency`` uses that matched table
as the denominator. An optional scored catalog supplies the ranking by event
ID; absent predictions remain misses. Choose ``comparison`` explicitly: cWB's
rho cut is strict ``>`` and its IFAR cut is inclusive ``>=``. Set
``amplitude_rtol: 0.1`` only when reproducing cWB ``simulation=1`` grouping.
The action writes counts, a decision table, 95% Wilson intervals, plots, and
measured log-amplitude hrss50 crossings or bounds. These are not cWB sigmoid-fit
parameters; sparse or unbracketed curves must not be reported as fitted points.

Standard cWB validation proceeds through ``cwb_merge``,
``cwb_setcuts M2 '--unique true'``, ``cwb_xgboost`` training/prediction,
``cwb_report LABEL create`` and ``cwb_setifar``. Use disjoint training and
held-out noise blocks, identical ranking configurations and the same injection
truth. The following example assumes the ROOT imports and scores already exist::

    - id: efficiency
      action: postprocess.simulation_report.simulation_efficiency
      args:
        matched_file: sim/matched.parquet
        scored_file: sim/scored.parquet
        ranking_par: rhor
        threshold: 0
        comparison: '>'
        amplitude_rtol: 0.1
        output_dir: efficiency
    - id: comparison
      action: postprocess.cwb_report.compare_cwb_report
      args:
        catalog_file: bkg/scored.parquet             # scored selected background
        progress_file: bkg/background_progress.parquet  # trigger_selection progress_file
        ranking_par: rhor
        reference_dir: /path/to/cwb/background/report/data
        efficiency_file: efficiency/efficiency.csv
        simulation_reference_dir: /path/to/cwb/simulation/report/data
        output_file: comparison.json

``compare_cwb_report`` compares the actual cWB text-table counts, with allowances
only for their printed numerical precision. Its background files are used as
given, so pass selected background triggers and the ``progress_file`` written
by the same ``trigger_selection`` step. It records disagreements and raises
on failure. ``compare_cwb_scores`` additionally checks event membership,
classification probabilities, ROOT-precision rankings, optional IFAR values,
and scores from an independently trained native model.

``attach_cwb_ifar`` is an explicit compatibility action for cWB's saved FAR step
graph, including small transition ramps and finite-tail clamping. It is distinct
from native empirical-tail IFAR calibration. IFAR values are seconds; cWB report
``T_ifar`` is years of 365 days. Do not compare these two calibration conventions
without matching them deliberately.

Some cWB versions write pickle models despite requiring a ``.json`` filename.
``import_cwb_model`` converts these to portable JSON/UBJ only with the explicit
``trusted_pickle: true`` option; use it for trusted, locally generated models.
The input file is preserved. Pin XGBoost and numerical dependency versions when
comparing independent training.

``training_diagnostics`` produces learning curves and feature-importance plots
from the saved native model. Declare them in the report's ``training.plots``.
For each ``simulation_runs`` entry, declare ``efficiency_file``,
``efficiency_summary_file`` and its plot in ``plots``. Supply the JSON written by
``collect_comparisons`` as ``validation_file`` to add a Consistency checks tab.
