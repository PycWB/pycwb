.. _understanding_results:
.. _understanding-your-results:

Reading and Analyzing Results
==============================

Use this page after :doc:`start_here` to open a completed run in Python,
connect triggers to their job segments, inspect progress and timing profiles,
and match recovered events to simulated signals. The examples use
``my_first_search``; replace that path with your own run directory.

Output directory
----------------

The run contains several related tables. For batch runs, first merge the
fragments with ``pycwb merge --work-dir RUN_DIRECTORY``
(see :doc:`run_on_clusters`), then open the master catalog.

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Path
     - Contents
   * - ``config/user_parameters.yaml``
     - Configuration copied during run setup.
   * - ``catalog/catalog.parquet``
     - One row per trigger, plus run configuration and version metadata.
   * - ``catalog/jobs.parquet``
     - Job descriptions: detector network, time windows, shifts and scheduled injections.
   * - ``catalog/progress.parquet``
     - Job/trial/lag completion, trigger counts and analyzed livetime.
   * - ``catalog/simulations.parquet``
     - One row per scheduled simulation, generated with ``simulation-summary``.
   * - ``trigger/`` and ``output/``
     - Event products enabled by the YAML, including waveforms and sky maps.
   * - ``gpu_profiles/*.pstats``
     - Optional function-call profiles from GPU lag profiling.

A **job segment** is a time interval prepared for the selected detectors. A
**trial** is an injection realization within that job; a **lag** selects a
relative detector time shift. A **trigger** is a reconstructed candidate that
passed the search cuts. One job/trial/lag can produce zero, one or many triggers.

Use ``job_id`` to connect a trigger to a job's ``index``. The combination
``(job_id, trial_idx, lag_idx)`` connects it to progress. Simulation matches add
columns prefixed with ``sim_`` while preserving the trigger columns.

Keep the catalog directory together when moving results: ``catalog.jobs``
follows the master catalog's reference to ``jobs.parquet``. See
:doc:`catalog_format` for the storage format and manifest rules.

Inspect a catalog
-----------------

.. code-block:: python

   from pathlib import Path
   import pandas as pd
   from pycwb.modules.catalog import Catalog

   run = Path("my_first_search")
   catalog = Catalog.open(str(run / "catalog/catalog.parquet"))
   table = catalog.triggers()        # PyArrow table
   events = table.to_pandas()        # pandas DataFrame
   jobs = catalog.jobs              # List of job dictionaries

   print("Detectors:", catalog.ifo_list)
   print("Triggers:", len(events), "Jobs:", len(jobs))
   columns = ["id", "job_id", "trial_idx", "lag_idx", "rho", "net_cc"]
   columns += [f"time_{ifo}" for ifo in catalog.ifo_list]
   print(events[columns].head().to_string(index=False))

``catalog.triggers()`` removes repeated rows with the same event/job/trial/lag
identity by default. ``catalog.config`` contains the saved configuration and
``catalog.version`` identifies the producing software. Direct
``pandas.read_parquet`` is also useful, but reads the stored rows without the
catalog reader's duplicate handling.

Current detector-specific columns use the detector name as a suffix. Inspect
``events.columns`` when opening unfamiliar data; older catalogs may contain
list-valued fields.

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
     - Unit / interpretation
   * - ``job_id``, ``trial_idx``, ``lag_idx``
     - Analysis job, injection trial and lag identifier
     - Integer identifiers; resolve lag shifts through job metadata
   * - ``time_H1``, ``time_L1``
     - Reconstructed event time in each detector
     - GPS seconds; arrival times may differ between detectors
   * - ``central_freq_H1``, ``central_freq_L1``
     - Central frequency. As in cWB, the first detector's column holds the
       network waveform centroid; the other detectors hold a pixel-based
       network estimate, not a per-detector measurement
     - Hz; ``bandwidth_*`` and ``duration_*`` follow the same convention
       (:ref:`event_output`)
   * - ``rho``
     - Coherent ranking statistic for the configured search mode
     - Dimensionless; not a probability or FAR
   * - ``net_cc``
     - Network correlation statistic
     - Dimensionless; use with the documented likelihood definition
   * - ``ra``, ``dec``
     - Reconstructed celestial position
     - Degrees in trigger output; see :ref:`units_conventions`
   * - ``injection``
     - Associated simulation parameters when available
     - Nullable struct; missing association is not proof of a missed injection

The complete current field definitions and serialization rules are in
:py:class:`pycwb.types.trigger.Trigger`. Do not substitute zero for missing
values: zero is a valid value for some statistics, while null means unavailable.

Select, summarize and plot triggers
------------------------------------

For a quick view, sort by ranking statistic or group by job and trial:

.. code-block:: python

   loudest = events.nlargest(10, "rho")
   print(loudest[columns].to_string(index=False))
   counts = events.groupby(["job_id", "trial_idx", "lag_idx"]).size()
   print(counts.rename("n_triggers"))

   # Catalog.filter returns another Arrow table, leaving the source unchanged.
   first_job_id = jobs[0]["index"]
   selected = catalog.filter(f"job_id == {first_job_id}").to_pandas()

These counts describe recorded candidates. Jobs with no triggers are absent
from this grouping; use the progress table below to find them.

For a distribution plot:

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, ax = plt.subplots()
   ax.hist(events["rho"].dropna(), bins=20)
   ax.set_xlabel("Coherent ranking statistic, rho")
   ax.set_ylabel("Trigger count")
   fig.tight_layout()
   plt.show()

The first-search demo has one trigger. A larger catalog is needed to see a
distribution. For detector waveforms and sky products, continue with
:doc:`tutorial_event_inspection`.

``Catalog.query`` exposes the table as ``triggers`` to DuckDB SQL. It is useful
for nested injection fields and waveform-specific JSON parameters:

.. code-block:: python

   injected = catalog.query("""
       SELECT id, rho, injection.approximant AS waveform,
              json_extract(injection.parameters, '$.frequency')::DOUBLE AS frequency
       FROM triggers
       WHERE injection IS NOT NULL
       ORDER BY rho DESC
   """).to_pandas()
   print(injected)

The nested ``injection`` field describes an association attached to a trigger;
it is not the complete list of scheduled or missed sources. Use the simulation
summary and explicit matching below for that list.

Connect triggers to job segments
--------------------------------

Read the job descriptions through the catalog, then select by job ID:

.. code-block:: python

   job_table = pd.DataFrame(jobs)
   job_columns = ["index", "ifos", "analyze_start", "analyze_end", "seg_edge", "shift"]
   print(job_table[job_columns].to_string(index=False))
   jobs_by_id = {job["index"]: job for job in jobs}

   if not events.empty:
       event = events.nlargest(1, "rho").iloc[0]
       job = jobs_by_id[int(event["job_id"])]
       print("Analysis window:", job["analyze_start"], job["analyze_end"])
       print("Scheduled injections in this job:", len(job.get("injections") or []))

``analyze_start`` and ``analyze_end`` are GPS seconds. ``seg_edge`` is the
padding loaded on each side for signal processing; it does not add analyzed
livetime. ``shift`` records each detector's superlag displacement. Job IDs are
identifiers, so ``jobs[event.job_id]`` is not a reliable lookup.

To use the time-window and lag properties of a job in Python, reconstruct a
:py:class:`pycwb.types.job.WaveSegment` from its dictionary:

.. code-block:: python

   from dacite import from_dict
   from pycwb.types.job import WaveSegment

   segment = from_dict(data_class=WaveSegment, data=jobs[0])
   print("Padded window:", segment.padded_start, segment.padded_end)
   print("Detector analysis starts:", segment.physical_analyze_starts)
   print("Lag shifts in seconds:", segment.lag_shifts)

``lag_idx`` indexes the job's lag-shift matrix; it is not a duration in seconds.
Resolve the actual shifts before separating zero lag and time-slide background.
See :doc:`job_control` for segment construction and lag conventions.

.. _empty-catalogs-and-failed-jobs:

Read progress and analyzed livetime
-----------------------------------

Start with the CLI summary:

.. code-block:: bash

   pycwb progress --work-dir my_first_search --verbose

To inspect the recorded outcomes in Python:

.. code-block:: python

   progress = pd.read_parquet(catalog.progress_file)
   keys = ["job_id", "trial_idx", "lag_idx"]
   progress = progress.sort_values("timestamp").drop_duplicates(keys, keep="last")
   print(progress[keys + ["status", "n_triggers", "livetime"]].to_string(index=False))

   completed = progress.loc[progress["status"] == "completed"]
   exposure = completed.groupby(["job_id", "trial_idx"])["livetime"].sum()
   print(exposure.rename("summed_lag_livetime_s"))

``livetime`` is the analyzed exposure in seconds for that lag after vetoes.
The sum above accumulates the recorded lags; it is not the elapsed running
time or the duration of unique detector data. ``timestamp`` is a Unix timestamp
for the progress record, not an event's GPS time.

A completed row with ``n_triggers == 0`` records a searched interval with no
retained candidates. ``skipped_segTHR`` records an interval skipped by the
minimum-duration rule. A missing completion record does not count as a completed
empty search; inspect the run log and ``progress`` summary. For selected catalogs,
use the corresponding selection's progress and livetime products.

Read saved timing profiles
---------------------------

Ordinary runs print stage timings to the run log. Optional GPU lag profiling
also writes ``.pstats`` files under ``gpu_profiles/`` when
``gpu.profile_lags`` is enabled. Read any saved profiles with Python's
``pstats`` module:

.. code-block:: python

   import pstats

   profile_files = sorted((run / "gpu_profiles").glob("*.pstats"))
   if profile_files:
       stats = pstats.Stats(*(str(path) for path in profile_files))
       stats.strip_dirs().sort_stats("cumulative").print_stats(20)
   else:
       print("No saved function profiles; use the run log for stage timings.")

``ncalls`` is the number of calls, ``tottime`` is time inside the function,
and ``cumtime`` includes its callees. Each file accumulates sampled lags for
one stage and process; combining files adds those measurements. Parallel workers
and nested calls mean these values cannot be added up as end-to-end wall time.
Profiling itself also adds overhead. See :doc:`dev_performance` for profiling
and performance settings.

``pycwb progress --timeit`` times the progress-report command itself. Saved
``execution_profile`` settings in ``catalog.config`` describe processing choices;
they are separate from measured timings.

.. _reading_simulation_matches:

Match triggers to simulations
------------------------------

For an injection run, create the simulation summary from its saved configuration
and match it to the trigger catalog:

.. code-block:: bash

   pycwb simulation-summary --work-dir my_first_search
   pycwb match-simulations my_first_search/catalog/catalog.parquet \
     my_first_search/catalog/simulations.parquet \
     --how right --output my_first_search/matched_simulations.parquet

The summary includes scheduled sources without recovered triggers. Building it
also evaluates waveform timing and requires the original generator and its inputs.
The match uses waveform/trigger time-window overlap within a trial, with
job/shift information when available. It chooses a unique association using
``--ranking-par`` (default ``rho``) and timing to resolve competing candidates.

.. list-table:: Choose the output rows
   :header-rows: 1
   :widths: 20 80

   * - ``--how``
     - Rows retained
   * - ``right``
     - All simulations; unmatched sources have null trigger fields.
   * - ``left``
     - All triggers; unmatched triggers have null simulation fields.
   * - ``inner``
     - Matched pairs only.
   * - ``outer``
     - Matched pairs and unmatched rows from both inputs.

.. code-block:: python

   truth = pd.read_parquet(run / "catalog/simulations.parquet")
   matched = pd.read_parquet(run / "matched_simulations.parquet")
   print(truth[["sim_idx", "job_id", "trial_idx", "gps_time", "hrss"]])
   print(matched[["sim_sim_idx", "sim_job_id", "sim_trial_idx", "id", "rho"]])

   recovered = matched.loc[matched["id"].notna()]
   missed = matched.loc[matched["id"].isna()]
   print("Associated sources:", len(recovered))
   print("Sources without an associated trigger:", missed["sim_sim_idx"].tolist())

Simulation columns receive a ``sim_`` prefix, so ``sim_idx`` becomes
``sim_sim_idx``. Keep unmatched rows when studying recovery. Before computing
an efficiency, inspect ``error``, ``vetoed_cat0``, ``vetoed_cat1``,
``vetoed_cat2`` and ``across_segments`` in the summary and define which sources
belong in the denominator. A null veto flag means that check was not available.
For matched tables these fields also have the ``sim_`` prefix.

Use :doc:`tutorial_population` for a worked recovered/missed-source example,
or :doc:`postproduction_efficiency` for selection and efficiency calculations.

FAR, IFAR and efficiency
------------------------

FAR is estimated from an appropriate background and its selected analyzed
exposure. IFAR is its reciprocal when FAR is positive. State the time units,
selection, ranking model, and finite-background limitations when reporting it.
Neither a high ``rho`` nor a single successful injection establishes significance.
Use :ref:`postproduction_background` for background estimation and
:ref:`postproduction_efficiency` for injection-population recovery fractions.

Keep training samples separate from the background used to evaluate FAR.
Compute exposure from completed jobs and the actual interval/lag selection,
including vetoes and overlap handling. ``N_jobs × N_lags × segLen`` is only an
idealized check when every job has that usable duration and no selection losses.

Where to go next
----------------

* :doc:`tutorial_event_inspection`: inspect one event's waveforms and sky products.
* :doc:`tutorial_population`: run an injection population and identify missed sources.
* :doc:`postproduction_study`: analyze your own completed catalogs and produce reports.
* :doc:`catalog_format`: look up storage, job manifests and selection provenance.
