.. _tutorial_search:

Search Workflow
================

Intermediate · Prerequisite: a completed :ref:`start_here` demo.
This page explains pipeline internals after the first user-facing run.

The main user-facing entry point is :py:func:`pycwb.workflow.run.search`.
It reads a YAML user-parameter file, prepares the working directory, creates
job segments, runs the configured segment processor, and writes trigger and
progress rows to the catalog.

The command-line interface calls the same function:

.. code-block:: bash

   pycwb run user_parameters.yaml

.. code-block:: python

   from pycwb.workflow.run import search

   search("user_parameters.yaml", working_dir=".", n_proc=4)

Execution Profiles
------------------

Existing configurations use the ``simple`` execution profile. To group jobs by
shared frame files and retain raw input between segment processes, opt into the
``scalable`` profile:

.. code-block:: yaml

   execution:
     profile: scalable
     memory_limit: 32GiB
     worker_memory: 8GiB
     cache_limit: 2GiB
     headroom: 1GiB
     batch_size: 8
     preload: auto

These are example reservations. ``worker_memory`` must cover the entire segment
process tree, including lag workers, pixel selection and numerical scratch.
Different search settings can need different reservations even for the same
frames. The runtime also accounts for full-frame decoding, bounds cache payload,
and monitors memory; use scheduler/cgroup limits for a hard allocation ceiling.

The same profile is used by ``run``, ``batch-setup`` and ``batch-runner``.
``segment_processer`` independently selects the scientific processor. Advanced
users can configure dotted ``execution.planner`` and ``execution.executor``
factories. Cluster scripts select explicit groups using stable batch IDs.

``batch-setup`` stores each planned group in a catalog fragment before
submission. Batch runners read their job selection from that fragment.

Use ``preload: off`` to compare direct reads, or ``preload: batch`` to attempt
bounded loading of a whole group's reusable inputs. Oversized entries fall back
to direct reads. Plans and resource measurements from scalable execution are
saved under the run's ``execution`` directory. Keep scientific configuration and
batch membership unchanged when resuming an existing run.

Job Control
-----------

Job setup is handled by :py:func:`pycwb.workflow.subflow.prepare_job_runs.prepare_job_runs`.
It checks the YAML against any existing catalogs, loads the configuration,
generates job segments, creates output directories, and creates the root
catalog. It does not initialize logging; :py:func:`pycwb.workflow.run.search`
calls :py:func:`~pycwb.modules.logger.logger.logger_init` first, so call it yourself
when using ``prepare_job_runs`` directly.

.. code-block:: python

   from pycwb.modules.logger import logger_init
   from pycwb.workflow.subflow.prepare_job_runs import prepare_job_runs

   logger_init(log_file=None, log_level="INFO")
   job_segments, config, working_dir = prepare_job_runs(
       ".",
       "user_parameters.yaml",
       n_proc=4,
       overwrite=False,
   )

The user-parameter YAML is loaded into :py:class:`pycwb.config.Config`.
Both local runs and batch workers use YAML as the runtime configuration.
The staged ``config/user_parameters.yaml`` is included in Condor file transfers;
external ``pycwb_schema`` definitions are embedded in that copy. Custom detector
definition JSON files are copied alongside it and referenced by a relative path.

Before reusing prepared jobs, the YAML settings are compared with a snapshot
in the existing catalog's Parquet metadata. Defaults are included; comments,
formatting and key order do not matter. CLI overrides are applied separately.
A mismatch stops the run and reports the changed settings, even with
``--force-overwrite`` (``overwrite=True`` in Python). Use a new working
directory, or clean the existing catalog, job manifest, progress and fragment
Parquet files and regenerate the run.
Older catalogs without a YAML snapshot also require regeneration because their
metadata mixes YAML values, derived fields and runtime overrides.
Custom detector definitions are checked by content hash, so moving their files
does not invalidate a run but changing their contents does. Other referenced data
files are not hashed; changed input data requires regenerating the prepared jobs.

.. code-block:: python

   from pycwb.config import Config

   config = Config()
   config.load_from_yaml("user_parameters.yaml")

Job segments are created from data-quality periods, explicit GPS windows,
GWOSC/event settings, or simulation settings. If injections are configured,
they are scheduled onto the relevant segments.

.. code-block:: python

   from pycwb.modules.job_segment import create_job_segment_from_config

   job_segments = create_job_segment_from_config(config)
   job_segment = job_segments[0]

The segment processor is loaded from ``config.segment_processer``. The default
processor is :py:func:`pycwb.workflow.subflow.process_job_segment_native.process_job_segment`.

.. code-block:: python

   from pycwb.utils.module import import_function

   segment_processor = import_function(config.segment_processer)

You normally do not need to call the processor directly. Use ``pycwb run`` or
``search(...)`` so catalog collection and run bookkeeping are configured for
you.

Data Analysis
-------------

The native segment processor analyzes one :py:class:`pycwb.types.job.WaveSegment`
at a time. The high-level stages are:

1. Read frame data and/or generate configured noise.
2. Generate and inject simulated signals when the segment has injections.
3. Resample to the analysis rate, regress (line removal), whiten, and compute
   per-detector noise RMS maps.
4. Build lag-independent coherence, time-delay, supercluster, and likelihood
   setup objects.
5. For each lag, run coherence, supercluster, likelihood, waveform
   reconstruction, optional plots, and catalog writes.

Data loading uses :py:func:`pycwb.modules.read_data.read_from_job_segment`,
:py:func:`pycwb.modules.read_data.simulations.generate_noise_for_job_seg`, and
:py:func:`pycwb.modules.injection.generate_strain_from_injection`.

.. code-block:: python

   from pycwb.modules.injection import generate_strain_from_injection
   from pycwb.modules.read_data import read_from_job_segment
   from pycwb.modules.read_data.simulations import generate_noise_for_job_seg

   data = None
   if job_segment.frames:
       data = read_from_job_segment(config, job_segment)
   if job_segment.noise:
       data = generate_noise_for_job_seg(job_segment, config.inRate, f_low=config.fLow, data=data)
   # Injections (generate_strain_from_injection) are omitted here;
   # see tutorial_injection.

The raw data are first resampled to the analysis rate
(``fResample`` or ``inRate``, divided by 2\ :sup:`levelR`) with
:py:func:`pycwb.modules.read_data.data_check.check_and_resample_py`. Data
conditioning then regresses and whitens each detector and returns conditioned
strains and per-detector nRMS maps.

.. code-block:: python

   from pycwb.modules.data_conditioning import condition_strains
   from pycwb.modules.read_data import check_and_resample_py

   data = [check_and_resample_py(data[i], config, i) for i in range(len(job_segment.ifos))]
   strains, nRMS = condition_strains(config, data)

The production processor then runs any configured post-whitening conditioning
hooks. The current native path builds reusable setup objects once per trial and
then processes each lag.

.. code-block:: python

   from pycwb.modules.coherence_native.coherence import setup_coherence, coherence_single_lag
   from pycwb.modules.likelihoodWP.likelihood import evaluate_cluster_likelihood, prepare_likelihood_inputs
   from pycwb.modules.super_cluster_native.super_cluster import setup_supercluster, supercluster_single_lag
   from pycwb.modules.xtalk.type import XTalk
   from pycwb.utils.td_vector_batch import build_td_inputs_cache

   # One-time, lag-independent setup
   coherence_setup = setup_coherence(config, strains, job_seg=job_segment, nRMS=nRMS)
   td_inputs_cache = build_td_inputs_cache(config, strains)
   xtalk = XTalk.load(config.MRAcatalog)
   supercluster_setup = setup_supercluster(config, gps_time=float(strains[0].start_time))
   likelihood_setup = prepare_likelihood_inputs(
       config,
       strains,
       config.nIFO,
       ml=supercluster_setup.get("ml_likelihood", supercluster_setup["ml"]),
       FP=supercluster_setup.get("FP_likelihood", supercluster_setup["FP"]),
       FX=supercluster_setup.get("FX_likelihood", supercluster_setup["FX"]),
   )

   # CAT2 keep windows for this segment (None when no CAT2 files are configured)
   veto_windows = (
       job_segment.cwb_veto_windows
       if job_segment.cwb_veto_windows is not None
       else job_segment.veto_windows
   )

   accepted = []
   for lag in range(job_segment.n_lag):
       fragment_clusters = coherence_single_lag(coherence_setup, lag_idx=lag, veto_windows=veto_windows)
       selected_clusters = supercluster_single_lag(
           supercluster_setup,
           config,
           fragment_clusters,
           lag_idx=lag,
           xtalk=xtalk,
           td_inputs_cache=td_inputs_cache,
       )
       if selected_clusters is None:
           continue
       for k, cluster in enumerate(selected_clusters.clusters):
           if cluster.cluster_status > 0:
               continue
           cluster.cluster_id = k + 1
           result_cluster, sky_stats = evaluate_cluster_likelihood(
               config.nIFO,
               cluster,
               config,
               cluster_id=k + 1,
               nRMS=nRMS,
               setup=likelihood_setup,
               xtalk=xtalk,
               chirp_seed=job_segment.index,
           )
           if result_cluster is not None and result_cluster.cluster_status == -1:
               accepted.append((lag, result_cluster, sky_stats))

The production processor also skips lags whose post-CAT2 livetime is below
``segTHR``, removes intervals excluded by conditioning hooks from the veto
windows, intersects them with the injection windows when
``analyze_injection_only`` is set, and handles lag bookkeeping, waveform
reconstruction, Q-veto, plots, memory cleanup, and catalog writes. For the full
implementation, see
:py:func:`pycwb.workflow.subflow.process_job_segment_native.process_job_segment`.


----

You have learned
----------------

- ✅ How pycWB structures a search with segments, jobs, and lags
- ✅ How to use the ``search()`` function from Python and ``pycwb run`` from CLI
- ✅ How job segments are created from the config file
- ✅ How the segment processor orchestrates each pipeline stage

**Next:** :doc:`tutorial_injection` — add simulated signals and recover them
