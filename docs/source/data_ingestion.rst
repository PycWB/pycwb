.. _data_ingestion:

Data Ingestion
==============

.. stage-nav:: search
   :current: data

This guide explains how pycWB obtains the detector strain for one job segment:
where the frame files come from, how they are cut to the segment's padded GPS
window, the synthetic-noise alternative, and the checks applied before the data
is passed to conditioning.

.. contents:: Table of Contents
   :depth: 2
   :local:


Why this matters
----------------

Every later stage works on the time series produced here. The first WDM time
bin is the start of the padded window, and cluster and event times are measured
from it. Many setup failures appear at this stage: a frame list that does not
cover the padded window, a wrong channel name, or frames whose sample rate
differs from ``inRate``. The checks below stop the job at these errors, so the
data is never silently shifted or truncated.


Overview
--------

Data ingestion is split between job setup and the job itself:

1. **Frame selection** (job setup, once per run). After the job segments are
   built, frame files are collected from ``frFiles`` or ``gwdatafind`` and
   attached to every job whose padded window they overlap
   (:py:func:`~pycwb.modules.job_segment.job_segment.attach_frame_files_to_job_segments`).
   See the *Frame File Selection* section of :doc:`job_control`.
2. **Reading** (once per job, before the trial loop). The frames attached to the
   job are read and merged into one time series per detector.
3. **Synthetic noise** (optional). If the job has a noise specification,
   Gaussian noise is generated for the same padded window. Frame data, if
   present, is added to it.
4. **Injections** are then added for each trial (see below and
   :doc:`injection_infrastructure`).

Padded window
~~~~~~~~~~~~~

A job with analysis window :math:`[t_s, t_e]` is always read over the padded
window

.. math::

   [\,t_s - \text{segEdge},\; t_e + \text{segEdge}\,]

of length :math:`t_e - t_s + 2\,\text{segEdge}` (``WaveSegment.padded_start``,
``padded_end``, ``padded_duration``). For jobs built from DQ files or a GPS
period, job setup has already trimmed ``segEdge`` from good time, so the padding
lies inside good time.

With super lags, detector :math:`k` has a shift :math:`\Delta_k` in seconds
(``WaveSegment.shift``, the super-lag id times ``segLen``). Its data is read from
the *physical* window :math:`[t_s - \Delta_k - \text{segEdge},\; t_e - \Delta_k
+ \text{segEdge}]` (``physical_padded_starts`` / ``physical_padded_ends``), and
the time labels are then shifted onto the common padded window. After reading,
all detectors share one time axis.

Reading a frame
~~~~~~~~~~~~~~~

For every attached frame file, the reader:

1. Clips the detector's physical padded window to the frame's
   ``[start, start + duration]``.
2. Reads the detector's channel (``channelNamesRaw``, in ``ifo`` order) over that
   range with ``gwpy.timeseries.TimeSeries.read``.
3. Shifts the time labels from physical to segment time.
4. Checks that the sample rate equals ``inRate`` (``ValueError`` otherwise).
   Frames are **not** resampled here; that is the first step of
   :doc:`data_conditioning`.

The pieces for each detector are then sorted by start time and joined with
gwpy's ``append(gap='raise')``, so any gap between frames is an error. The
merged series must start exactly at ``padded_start`` and end exactly at
``padded_end``.

Injections
~~~~~~~~~~

The strain is read once per job. If the job has several injection trials, each
trial works on its own copy. Fixed-amplitude injections are added to the strain
at ``inRate``, before resampling. Target-SNR injections with the default
``injection_resampling: cwb`` are resampled separately and added after the
strain is resampled. In both cases the signal is in the data before whitening.


Implementation
--------------

- :py:func:`pycwb.modules.read_data.read_data.read_from_job_segment` — reads all
  frames of a :py:class:`~pycwb.types.job.WaveSegment` and merges them. With
  ``nproc > 1`` the frames are read in a ``multiprocessing`` pool of
  ``min(nproc, number of frames)`` processes.
- :py:func:`~pycwb.modules.read_data.read_data.read_single_frame_from_job_segment`
  (one frame), :py:func:`~pycwb.modules.read_data.read_data.read_from_gwf`
  (GWF read with file and NaN checks) and
  :py:func:`~pycwb.modules.read_data.read_data.merge_frames` (join and window
  check).
- :py:func:`pycwb.modules.read_data.parallel.read_from_job_segment` — bounded
  thread or spawned-process reads followed by the same merge. The CUDA
  processor (see :doc:`backends`) uses it when ``gpu.read_workers > 1`` and no
  input provider is given.
- :py:func:`pycwb.modules.read_data.simulations.generate_noise_for_job_seg` —
  Gaussian noise for the padded window
  (:py:func:`pycwb.modules.noise.gaussian.generate_noise`).

The native processor
:py:func:`pycwb.workflow.subflow.process_job_segment_native.process_job_segment`
calls these functions. With the scalable execution profile
(:doc:`workflow_execution`), the processor receives an input provider. Frames
are then read one at a time through it, from the shared raw-input cache when an
entry matches, or else with ``read_from_gwf``.

**Synthetic noise.** A job has a noise specification when the ``injection``
block contains ``noise`` (keys ``type``, ``psds`` and ``delta_seeds``, each
mapped by detector) or, deprecated, ``injection.segment.noise``. Noise is
generated at ``inRate`` with ``fLow`` as low-frequency cutoff. Each ``psds``
entry is a two-column frequency/ASD text file. A detector's seed is its
``delta_seeds`` value plus the job's physical analysis start time.

**Public open data.** :py:mod:`pycwb.modules.gwosc` downloads GWOSC frames and
writes ordinary frame lists, which are then read through ``frFiles``.
``pycwb gwosc-data <yaml>`` fetches ``[gps_start - segEdge, gps_end + segEdge]``
at ``inRate`` and writes DQ files into ``<work-dir>/input``.
``pycwb gwosc <event>`` fetches the event's 4096 Hz files and writes DQ files,
``cwb_period.txt`` and a template ``user_parameters.yaml``. ``inRate`` must match
the downloaded files. See :doc:`tutorial_open_data`.

**Online search.** ``pycwb online`` does not use ``read_from_job_segment``. It
streams data through the sources in :py:mod:`pycwb.modules.online.data_source`
(NDS2, Kafka, shared-memory frame directory). See :doc:`online_search`.

**cWB-2G correspondence.** ``cwb2G::ReadData``, part of the data-conditioning
stage in the :doc:`pipeline_lifecycle` table, reads the frames with
``channelNamesRaw``, stops on NaN samples or a rate other than ``inRate``,
applies the super-lag shift, then applies ``dcCal``, ``fResample`` and the
``levelR`` resampling. pycWB does the reading, checks and super-lag relabelling
here, and applies ``dcCal`` and resampling in
:py:func:`~pycwb.modules.read_data.data_check.check_and_resample_py` at the start
of :doc:`data_conditioning`. pycWB generates injections in memory; the native
reader does not use the MDC channels ``channelNamesMDC``.


Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - Parameter
     - Default
     - Meaning
   * - ``ifo``
     - required
     - Detector order. All per-detector lists below follow it.
   * - ``frFiles``
     - ``[]``
     - One frame-list file per detector, one ``.gwf`` path per line. It takes
       precedence over ``gwdatafind``.
   * - ``gwdatafind.frametype``
     - required with ``gwdatafind``
     - One frame type per detector.
   * - ``gwdatafind.site``
     - first letter of each ``ifo``
     - One site letter per detector.
   * - ``gwdatafind.host``
     - — (gwdatafind default)
     - Datafind server.
   * - ``gwdatafind.urltype``
     - ``file``
     - URL type. An ``osdf://`` path is read from a file with the same
       basename in the current directory (e.g. transferred by HTCondor).
   * - ``channelNamesRaw``
     - ``[]``
     - One strain channel per detector, for example ``H1:GDS-CALIB_STRAIN``.
       Required whenever frames are read.
   * - ``inRate``
     - 16384
     - Sample rate the frames must already have [Hz]. Checked, not
       converted, at read time.
   * - ``segEdge``
     - 8 s
     - Padding read on each side of the analysis window.
   * - ``nproc``
     - 1
     - More than 1 reads frames in a process pool (native reader).
   * - ``gpu.read_workers``
     - 1 (max 2)
     - Concurrent frame reads in the CUDA processor.
   * - ``gpu.read_processes``
     - ``false``
     - Spawn reader processes instead of threads.
   * - ``gpu.validate_read``
     - ``false``
     - Repeat the reads serially and require bitwise-identical data.
   * - ``injection.noise``
     - — (unset)
     - Generate Gaussian noise for each job (see above).


Output
------

Conditioning receives a list of :py:class:`pycwb.types.time_series.TimeSeries`,
one per detector in ``ifo`` order. Each series:

- covers the padded window ``[padded_start, padded_end]`` in segment time
  (``segEdge`` on both sides, super-lag shift removed) at ``inRate``;
- contains frame strain, synthetic noise, or both, plus the trial's
  fixed-amplitude injections.


.. raw:: html

   <span id="validation-checks"></span>

Troubleshoot input data
-----------------------

- **Frames exist and are named correctly**: every file in a ``frFiles`` list
  must exist and be named ``...-<gps>-<duration>.gwf`` (GPS start ≥ 1104105616,
  i.e. 2015-01-01; duration ≥ 1 s). Lines starting with ``#`` are skipped.
- **Coverage fails early**: ``pycwb run ... --list-jobs`` runs job setup. A
  detector without frames, or frames that do not cover a job's padded window
  without gaps, raises ``ValueError`` there, before any strain is read.
- **Sample rate matches** ``inRate``: the log line
  ``data info: start=..., duration=..., rate=...`` shows each merged detector
  series. A rate mismatch is a ``ValueError`` naming the frame file.
- **Channel is present**: a wrong ``channelNamesRaw`` entry appears as a
  ``RuntimeError`` naming the file, channel and read window.
- **No NaN, no empty files**: NaN samples or 0-byte files raise ``ValueError``.
- **Something to analyse**: a job with no frames, noise or injections stops
  with ``No data to process``.


----

**See also:** :doc:`pipeline_lifecycle` · :doc:`job_control` · :doc:`injection_infrastructure`

**Next:** :doc:`data_conditioning` — resampling, line removal and whitening
