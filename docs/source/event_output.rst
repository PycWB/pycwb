.. _event_output:

Event Output
============

.. stage-nav:: search
   :current: events

This guide explains what the search writes for each accepted trigger and each
processed lag: the Parquet trigger catalog, per-trigger folders, the waveform
file and the progress file, plus batch merging and what postproduction reads.

.. contents:: Table of Contents
   :depth: 2
   :local:


Why this matters
----------------

Postproduction reads these files, not the in-memory events. Knowing which
columns are placeholders and how progress rows relate to catalog rows tells an
empty lag from a failed one, and a measured zero from a value never computed.


What Is Written and When
------------------------

.. code-block:: text

   <workdir>/
   ├── catalog/
   │   ├── catalog.parquet       # one row per accepted trigger
   │   ├── jobs.parquet          # immutable job manifest of the master catalog
   │   ├── progress.parquet      # one row per completed or skipped lag
   │   └── fragment/             # batch: catalog_<id>.parquet, progress_<id>.parquet
   ├── trigger/
   │   └── trigger_<job_id>_<trial_idx>_<stop>_<hash>/
   │       ├── cluster.json            # save_cluster
   │       ├── skymap_statistics.json  # save_sky_map
   │       └── *.png                   # plot_* flags
   └── output/
       └── wave.h5               # save_waveform / save_injection (batch: wave_<id>.h5)

Run setup creates an empty ``catalog.parquet`` (resolved configuration and
pycWB version in its schema metadata) and ``jobs.parquet``. Then, for every lag
of every trial, the native workflow
(:py:mod:`pycwb.workflow.subflow.process_job_segment_native`) writes the
trigger folders and reconstructs the waveforms. Reconstruction always runs,
because ``q_veto`` and ``q_factor`` (minima over detectors, from the whitened
reconstructed data and signal) need it; simulations also reconstruct the
injection. Each event is appended as a :py:class:`~pycwb.types.trigger.Trigger`,
and one progress row records the lag's trigger count and post-veto livetime.

In ``pycwb run`` and batch runs one writer process handles all catalog,
progress and waveform writes and flushes buffered triggers before each
progress row, so a progress row means its lag's triggers are in the catalog.
When a batch job restarts, lags with a progress row are skipped and triggers
of (trial, lag) pairs without one are removed first. Progress columns and
``pycwb progress`` are under *Progress Tracking* in :doc:`job_control`.

Trigger folders
~~~~~~~~~~~~~~~

The folder name reuses the stop time (first detector's ``event_stop``) and
hash that end the catalog ``id``, ``<job_id>_<trial_idx>_<lag_idx>_<stop>_<hash>``,
so every catalog row leads to its folder.

- ``cluster.json`` (``save_cluster``): the
  :py:class:`~pycwb.types.network_cluster.Cluster` with pixels and metadata.
- ``skymap_statistics.json`` (``save_sky_map``):
  :py:class:`~pycwb.modules.likelihoodWP.results.SkyMapStatistics`, the
  per-direction sky arrays (``nLikelihood``, ``nProbability``, …) and ``l_max``.
- PNGs: ``likelihood_map`` and ``null_map`` (``plot_trigger``), one per sky
  array (``plot_sky_map``), ``<ifo>_wf_REC/DAT/NUL[_whiten]``
  (``plot_waveform``) and ``<ifo>_wf_INJ[_whiten]`` (``plot_injection``).

With the defaults a folder holds only ``cluster.json``; with
``save_cluster: false`` and nothing else enabled, no folders are created.

Waveform file
~~~~~~~~~~~~~

``save_waveform`` and ``save_injection`` write one HDF5 file in ``outputDir``
(``wave.h5`` locally, ``wave_<id>.h5`` per batch fragment). Each trigger is a
group named by its hash, the last part of ``id``. Datasets are
``<ifo>_wf_REC`` (reconstructed signal), ``<ifo>_wf_DAT`` (signal plus noise)
and ``<ifo>_wf_NUL`` (DAT − REC), each also as ``_whiten``, plus
``<ifo>_wf_INJ[_whiten]`` with ``save_injection``. Datasets carry
``sample_rate`` and ``start_time`` attributes.

Batch fragments and provenance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Batch workers write ``catalog/fragment/catalog_<id>.parquet`` and
``progress_<id>.parquet``, with the job list inline because fragments travel
without the run manifest. ``pycwb merge --work-dir <run>`` concatenates them
into ``catalog.parquet`` and ``progress.parquet`` (``--mlabel <label>`` writes
``catalog.<label>.parquet`` and ``progress.<label>.parquet``). It keeps the
master catalog's provenance, rejects fragments whose job descriptions conflict
with it, and asks before replacing rows of a non-empty master catalog.
``pycwb merge --wave`` separately combines ``output/wave_*.h5`` into
``output/wave.h5``. Provenance and transfer rules are in :doc:`catalog_format`.


Catalog Columns
---------------

The row schema is :py:meth:`pycwb.types.trigger.Trigger.arrow_schema` for the
configured ``ifo`` list. Per-detector quantities are flat ``<field>_<ifo>``
columns (``time_H1``, ``hrss_L1``, …) in ``ifo`` order: 53 network columns,
23 per detector and one ``injection`` struct, i.e. 100 columns for H1–L1. GPS
times are float64, most statistics float32. Definitions are in
:doc:`likelihood_guide`; :py:class:`pycwb.types.trigger.Trigger` documents each
field and its cWB name.

.. list-table::
   :header-rows: 1
   :widths: 16 46 38

   * - Group
     - Columns
     - Notes
   * - Identification
     - ``id``, ``job_id``, ``trial_idx``, ``lag_idx``, ``cluster_id``,
       ``event_index``, ``n_detectors``, ``ifo_list``, ``hybrid``
     - ``hybrid`` is false for native output
   * - Time, shifts
     - ``gps_time``, ``time_<ifo>``, ``event_start_<ifo>``,
       ``event_stop_<ifo>``, ``segment_start_<ifo>``, ``left_edge_<ifo>``,
       ``right_edge_<ifo>``, ``time_lag_<ifo>``, ``segment_lag_<ifo>``
     - GPS s; offsets and shifts in s; ``gps_time`` = first detector's ``time``
   * - Signal
     - ``central_freq_<ifo>``, ``freq_low_<ifo>``, ``freq_high_<ifo>``,
       ``bandwidth_<ifo>``, ``duration_<ifo>``, ``sample_rate_<ifo>``
     - Hz, s; see *First-detector columns*
   * - Ranking
     - ``rho``, ``rho_alt``, ``net_cc``, ``sky_cc``, ``subnet_cc``,
       ``subnet_cc2``, ``penalty``, ``likelihood``
     - cWB ``rho[0..1]``, ``netcc[0..3]``; ``penalty``: :math:`\chi^2`/nDoF
   * - Energies, pixels
     - ``coherent_energy``, ``net_null``, ``net_energy``, …;
       ``data_energy_<ifo>``, ``signal_energy_<ifo>``, …; ``n_pixels_*``, …
     - cWB ``snr``, ``sSNR``, ``null`` per detector; ``data_energy`` ≈ SNR²
   * - Sky
     - ``ra``, ``dec``, ``phi``, ``theta``, ``phi_det``, ``theta_det``,
       ``psi``, ``iota``, ``sky_error_regions``, ``fp_<ifo>``, ``fx_<ifo>``
     - ``ra``/``dec``/``phi``/``theta`` in degrees (``phi``/``theta`` Earth-fixed);
       ``sky_error_regions``: 11 entries (cWB ``erA``), NaN if unavailable
   * - Reconstruction
     - ``hrss_<ifo>``, ``strain``, ``noise_rms_<ifo>``, ``q_veto``,
       ``q_factor``
     - ``strain`` is the network hrss; ``noise_rms`` is in strain/√Hz
   * - Chirp, FAR
     - ``mchirp``, ``mchirp_err``, ``chirp_ellip``, ``chirp_pfrac``,
       ``chirp_efrac``; ``ifar``
     - 0 unless estimated (below); ``ifar`` stays 0 in search output
   * - Simulation
     - ``injection`` (struct)
     - Null for background; no other column is null in search output

**First-detector columns.** As in cWB, the first detector's slot holds a
network value: ``central_freq`` is the reconstructed waveform's centroid, and
``bandwidth`` and ``duration`` are energy-weighted over all resolutions. The
other detectors hold pixel-based values.

**Chirp mass.** Native chirp columns are filled only by the micropixel
estimator (:py:func:`pycwb.modules.likelihoodWP.chirp_micropixel.estimate_chirp`).
It needs ``execution_profile.native_chirp`` and ``xgb_rho_mode`` true (both
default ``false``), ``Search`` in ``CBC``/``BBH``/``IMBHB`` (default ``""``),
``optim`` false and a lower-case ``cfg_search`` (default ``r``); ``chirp_pfrac``
then holds the energy symmetry about the chirp track. Otherwise the columns
stay 0: the default Hough estimator only rescales ``rho_alt`` (2G, ``pattern`` ≠ 0).

**Injections.** An injection is attached when its ``gps_time`` lies within
0.1 s of the first detector's [``event_start``, ``event_stop``]. The
:py:class:`~pycwb.types.simulation.InjectionParams` struct holds typed fields
(``name``, ``hrss``, ``gps_time``, ``ra`` and ``dec`` in radians, …),
per-detector lists (``snr_sq``, ``rec_snr_sq``, …) and ``parameters``, the
injection dictionary as JSON (:ref:`units_conventions`). A null ``injection``
does not prove a miss; postproduction matching decides recovery.


What Postproduction Reads
-------------------------

These are the inputs of :doc:`postproduction_trainingset`;
:doc:`understanding_results` shows how to inspect them.

- ``catalog.parquet`` and its job list: statistics and XGBoost features;
  ``lag_idx``, shift columns and per-job shifts separate zero lag and drive
  train/FAR splits (:py:func:`pycwb.modules.postprocess.selection.trigger_selection`).
- ``progress.parquet``: livetime per (job, trial, lag). Exposure comes from
  these rows, so lags and jobs without triggers still count.
- ``simulations.parquet``: one row per injection, from
  ``pycwb simulation-summary`` (default ``catalog/simulations.parquet``).
- ``wave.h5``: only for the waveform report
  (:py:func:`pycwb.modules.postprocess.waveform_report.generate_waveform_report`),
  which needs ``save_waveform`` and ``save_injection``.


Implementation
--------------

- :py:meth:`pycwb.types.network_event.Event.output_py` fills cWB-style arrays
  from a cluster; :py:meth:`pycwb.types.trigger.Trigger.from_event` names them.
- :py:class:`pycwb.modules.catalog.catalog.Catalog` appends catalog and
  progress rows under a soft file lock with atomic replacement;
  :py:func:`pycwb.modules.workflow_utils.trigger_utils.save_trigger` and
  :py:func:`pycwb.workflow.subflow.postprocess_and_plots.add_wf_to_wave` write
  the folder JSON and HDF5 groups.
- :py:func:`pycwb.workflow.batch.data_collector` and
  :py:class:`pycwb.workflow.execution.writer.OutputWriter` are the single
  writers; :py:mod:`pycwb.workflow.merge` implements ``pycwb merge``.

cWB-2G correspondence
~~~~~~~~~~~~~~~~~~~~~

cWB-2G stored events in the ``waveburst`` tree of its ``wave*.root`` files and
per-lag exposure in the ``liveTime`` tree; pycWB writes ``catalog.parquet`` and
``progress.parquet``, with trigger folders and plots in place of CED pages
(:doc:`cwb_heritage`). Positional arrays become named columns (``rho[0]`` →
``rho``, ``rho[1]`` → ``rho_alt``, ``netcc[0]`` → ``net_cc``, ``phi[2]`` →
``ra``, ``time[ifo]`` → ``time_<ifo>``). ROOT results convert with
:py:func:`pycwb.modules.catalog.convert_root.convert_root_to_catalog` or load
into postproduction with :py:func:`pycwb.modules.postprocess.root_adapter.read_cwb_root`
(:doc:`postproduction_root`).


Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Meaning
   * - ``catalog_dir``, ``trigger_dir``, ``outputDir``
     - ``catalog``, ``trigger``, ``output``
     - Directories for catalogs, trigger folders and the wave file
   * - ``save_cluster``
     - ``true``
     - Write ``cluster.json``
   * - ``save_sky_map``
     - ``false``
     - Write ``skymap_statistics.json``
   * - ``save_waveform``, ``save_injection``
     - ``false``
     - Store REC/DAT/NUL and injected (INJ) waveforms in the wave file
   * - ``plot_trigger``, ``plot_waveform``, ``plot_injection``,
       ``plot_sky_map``
     - ``false``
     - Write the PNGs listed under `Trigger folders`_

``pycwb run --plot`` turns on ``save_waveform``, ``save_sky_map``,
``plot_trigger``, ``plot_waveform`` and ``plot_sky_map`` for that run.


.. raw:: html

   <span id="validation-checks"></span>

Inspect saved products
----------------------

- **Progress first**: run ``pycwb progress --work-dir <run>``. A missing
  (job, trial, lag) row means that lag failed, was interrupted or never ran.
- **Rows match counts**: catalog rows per (``job_id``, ``trial_idx``,
  ``lag_idx``) should equal progress ``n_triggers``; fewer means a failed
  conversion or write (see the job log). ``Catalog.triggers()`` drops
  duplicate rows; ``pandas.read_parquet`` does not.
- **Merged and traceable**: after a batch run, check that ``catalog.parquet``
  holds every fragment's rows and that ``Catalog.open(path).jobs`` loads.
- **Zero lag**: rows with all ``time_lag_<ifo>`` and ``segment_lag_<ifo>``
  zero are physical zero lag and must stay out of the background.


----

**See also:** :doc:`pipeline_lifecycle` · :doc:`catalog_format` · :doc:`understanding_results`

**Next:** :doc:`postproduction_background` — postproduction starts by estimating the background from these catalogs
