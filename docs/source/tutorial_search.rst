.. _tutorial_search:

Search Workflow
================

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

Job Control
-----------

Job setup is handled by :py:func:`pycwb.workflow.subflow.prepare_job_runs.prepare_job_runs`.
It initializes logging, loads the configuration, creates output directories,
generates job segments, and creates the root catalog.

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

1. Read frame data or generate configured noise.
2. Generate and inject simulated signals when the segment has injections.
3. Resample, whiten, and compute per-detector noise RMS maps.
4. Build lag-independent coherence, time-delay, supercluster, and likelihood
   setup objects.
5. For each lag, run coherence, supercluster, likelihood, waveform
   reconstruction, optional plots, and catalog writes.

Data loading uses :py:func:`pycwb.modules.read_data.read_from_job_segment`,
:py:func:`pycwb.modules.read_data.generate_noise_for_job_seg`, and
:py:func:`pycwb.modules.read_data.generate_strain_from_injection`.

.. code-block:: python

   from pycwb.modules.read_data import (
       generate_noise_for_job_seg,
       generate_strain_from_injection,
       read_from_job_segment,
   )

   data = None
   if job_segment.frames:
       data = read_from_job_segment(config, job_segment)
   if job_segment.noise:
       data = generate_noise_for_job_seg(job_segment, config.inRate, data=data)

Data conditioning returns conditioned strains and per-detector nRMS maps.

.. code-block:: python

   from pycwb.modules.data_conditioning import data_conditioning

   strains, nRMS = data_conditioning(config, data)

The current native path builds reusable setup objects once per trial and then
processes each lag.

.. code-block:: python

   from pycwb.modules.coherence_native.coherence import setup_coherence, coherence_single_lag
   from pycwb.modules.likelihoodWP.likelihood import setup_likelihood, likelihood
   from pycwb.modules.super_cluster_native.super_cluster import setup_supercluster, supercluster_single_lag
   from pycwb.modules.xtalk.type import XTalk
   from pycwb.utils.td_vector_batch import build_td_inputs_cache

   coherence_setup = setup_coherence(config, strains, job_seg=job_segment)
   td_inputs_cache = build_td_inputs_cache(config, strains)
   supercluster_setup = setup_supercluster(config, gps_time=float(strains[0].start_time))
   likelihood_setup = setup_likelihood(config, strains, config.nIFO)
   xtalk = XTalk.load(config.MRAcatalog)

   fragment_clusters = coherence_single_lag(coherence_setup, lag_idx=0)
   selected_clusters = supercluster_single_lag(
       supercluster_setup,
       config,
       fragment_clusters,
       lag_idx=0,
       xtalk=xtalk,
       td_inputs_cache=td_inputs_cache,
   )

The production processor also handles lag bookkeeping, veto windows, waveform
reconstruction, Q-veto, plots, memory cleanup, and catalog writes. For the full
implementation, see
:py:func:`pycwb.workflow.subflow.process_job_segment_native.process_job_segment`.
