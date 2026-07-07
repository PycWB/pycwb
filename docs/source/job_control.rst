.. _job_control:

Job Control
===========

.. rubric:: Pipeline: :doc:`data <pipeline_lifecycle>` → **[jobs & segments]** ← you are here → :doc:`conditioning <pipeline_lifecycle>` → :doc:`WDM <pipeline_lifecycle>` → :doc:`pixels <clustering_algorithm>` → :doc:`clusters <clustering_algorithm>` → :doc:`likelihood <likelihood_guide>` → :doc:`events <pipeline_lifecycle>` → :doc:`bkg <postproduction_background>` → :doc:`ranking <postproduction_xgboost>` → :doc:`eff <postproduction_efficiency>`

This guide explains how pycWB defines and manages analysis jobs, including the
lag/slag structure, trial indexing, segment construction from data-quality
files, and parallelization strategies.

.. contents:: Table of Contents
   :depth: 2
   :local:


Why this matters
----------------

Job control determines how your search is split across computing resources.
Understanding segments, lags, and trials is essential for debugging failed
jobs, estimating runtime, and optimizing cluster utilization. If jobs are
failing or your run takes too long, this is the page to read.


Job Decomposition
-----------------

.. mermaid::

   flowchart TD
     F[Frame Files] --> S[Segments]
     S --> J[Jobs<br/>segment x lag x trial]
     J --> L[Lags<br/>time-shift hypotheses]
     L --> T[Trials<br/>injection groups]
     T --> O[Output<br/>catalog + progress]
     S -.->|DQ files| DQ[CAT0/1/2 Veto]
     J -.->|cluster| BS[Condor / SLURM]
     L -.->|zero-lag| ZL[Physical Signal]
     L -.->|non-zero| BG[Background]


Overview
--------

A pycWB analysis run is decomposed into **job segments**—independent units of
work that can be distributed across cluster nodes. Each job segment is a GPS
time window containing detector data, processed across multiple **lags**
(time-shift hypotheses) and optionally **trials** (injection groups).

The job structure is built by
:py:func:`pycwb.modules.job_segment.job_segment.create_job_segment_from_config`,
which reads the user parameter YAML and produces a list of
:py:class:`~pycwb.types.job.WaveSegment` objects.


Job Segment Construction
------------------------

pycWB supports five modes for defining job segment time windows:

1. **Pure Simulation** — no real data; segments defined solely by injection
   times. Used for waveform injection studies with synthetic noise.

2. ``gps_start`` / ``gps_end`` — a single explicit time interval:

   .. code-block:: yaml

      gps_start: 1264060000
      gps_end: 1264063600

3. ``gps_center`` + ``time_left``/``time_right`` — a centered window:

   .. code-block:: yaml

      gps_center: 1264060000
      time_left: 500
      time_right: 500

4. ``superevent`` + ``time_left``/``time_right`` — queries GraceDB for the
   GPS time of a superevent (e.g., ``S190521g``) and builds a window around it.

5. **DQ Files** — builds segments from science-quality data flags. This is the
   standard mode for production searches:

   .. code-block:: yaml

      DQ_CAT1: input/H1_cat1.txt     # CAT1 veto segments
      DQ_CAT2: input/H1_cat2.txt     # CAT2 veto segments (applied as windows)
      DQ_CAT0: input/H1_cat0.txt     # Science-mode segments (CAT0)

   The segment-building algorithm:
   
   a. Read CAT1 segments and remove them from science time.
   b. Merge remaining science segments that are separated by less than
      ``segTHR`` seconds.
   c. Keep segments longer than ``segMLS`` seconds.
   d. Apply CAT2 veto **windows** (not segments) around each veto edge.
   e. Split long segments into chunks of ``segLen`` seconds with
      ``segOverlap`` overlap.
   f. Add ``segEdge`` seconds of padding on each side for wavelet boundary
      effects.


Segment Sizing Parameters
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Parameter
     - Default
     - Description
   * - ``segLen``
     - 600 s
     - Nominal segment (job) length
   * - ``segMLS``
     - 300 s
     - Minimum segment length after CAT1 veto
   * - ``segTHR``
     - 30 s
     - Minimum separation after CAT2 veto
   * - ``segEdge``
     - 8 s
     - Wavelet boundary padding on each side
   * - ``segOverlap``
     - 0 s
     - Overlap between consecutive job segments


Lag Structure
-------------

Lags implement the time-shift analysis used to estimate the background
(accidental coincidence rate). For an :math:`N`-detector network, time-shifting
one detector's data relative to the others breaks any real gravitational-wave
coincidence.

Regular Lags
~~~~~~~~~~~~

.. code-block:: yaml

   lagSize: 100       # Number of lags to generate
   lagStep: 1.0       # Time step between lags [s]
   lagOff: 6          # Offset: first N lags are skipped (0 = include zero-lag)
   lagMax: 150        # Maximum time shift [s]

Lags are generated as:

.. math::

   \text{lag}[i] = (\text{lagOff} + i) \times \text{lagStep},
   \quad i = 0, 1, \dots, \text{lagSize} - 1

subject to :math:`\text{lag}[i] \leq \text{lagMax}`.

- **Zero-lag** (:math:`i` such that :math:`\text{lagOff} + i = 0`) represents
  the physical (unshifted) coincidence—where a real GW signal would appear.
- **Non-zero lags** are used for background estimation.

You can also provide an explicit lag array or lag file:

.. code-block:: yaml

   lagMode: r                    # "r" = read from file, "w" = write to file
   lagFile: input/lags.txt       # Path to lag list file
   lagSite: 0                    # Site index for time-shift reference

When ``lagMode`` is ``r``, lags are read from ``lagFile``. When ``w``, lags
are written to ``lagFile`` for inspection or sharing.

Super Lags (Segments)
~~~~~~~~~~~~~~~~~~~~~

Super lags (slang) provide an additional layer of time shifts at the segment
level, used for multi-detector networks:

.. code-block:: yaml

   slagSize: 10       # Number of super lags
   slagMin: -5.0      # Minimum super lag [s]
   slagMax: 5.0       # Maximum super lag [s]
   slagOff: 0         # Super lag offset [s]

Super lags are generated as linearly spaced offsets between ``slagMin`` and
``slagMax`` or as explicit step/offset combinations.

Each super lag produces a new "shifted" version of the job segment, increasing
the total number of analysis units by a factor of ``slagSize``.


Trial Indexing
--------------

For simulation (injection) studies, each job segment can contain multiple
**trials**—groups of injections that share the same noise background:

.. math::

   \text{total\_jobs} = N_{segments} \times N_{slags} \times N_{trials}

- ``trial_idx``: identifies which trial an injection belongs to within a
  job segment.
- ``sim_idx``: unique identifier for each injection across all trials and
  jobs.
- ``job_id``: unique identifier for each analysis job (segment × slag ×
  trial combination).

When ``parallel_injection_trail`` is enabled, job segments are flattened by
trial via
:py:func:`~pycwb.modules.job_segment.job_segment.flatten_job_segments_by_trial`,
so each trial becomes a separate job with a contiguous ``job_id``. This enables
trivial parallelization across trials.


Job Directory Structure
-----------------------

Each job creates this directory layout:

.. code-block:: text

   <workdir>/
   ├── output/           # Waveform and trigger output files
   ├── log/              # Job log files
   ├── config/           # Copy of user_parameters.yaml
   ├── catalog/          # Parquet trigger catalogs
   │   ├── catalog.parquet
   │   └── progress.parquet
   ├── trigger/          # Per-event JSON trigger files
   ├── job_status/       # Job completion status files
   ├── public/           # Public-facing results
   └── input/            # DQ files, frame lists, etc.


Frame File Selection
--------------------

Frame files (containing detector strain data) are selected in two ways:

1. **Explicit file list** via the ``frFiles`` parameter:

   .. code-block:: yaml

      frFiles:
        - /path/to/H-H1_GWOSC-1264060000-4096.gwf
        - /path/to/L-L1_GWOSC-1264060000-4096.gwf

2. **gwdatafind query** via the ``gwdatafind`` config block:

   .. code-block:: yaml

      gwdatafind:
        site: H1
        frametype: H1_GWOSC_O4_C01_4KHZ_R1
        host: datafind.ligo.org

   This automatically queries the LIGO data-find server for frames covering
   each job segment's time window.


Parallelization
---------------

Jobs are parallelized at two levels:

- **Across lags** — multiple lags within a segment can be processed
  concurrently (controlled by ``parallel_lag_workers``, default 1).
- **Across segments** — different job segments are independent and can run on
  different cluster nodes (see :ref:`run_on_clusters`).

For SLURM/HTCondor batch submission, jobs are bundled into workers via
``job_per_worker`` to balance scheduling overhead against parallelism.


Progress Tracking
-----------------

Each job writes a ``progress.parquet`` file containing per-lag processing
status (start time, end time, success/failure, number of triggers). The
``pycwb progress`` CLI command summarizes this information:

.. code-block:: bash

   pycwb progress --work-dir /path/to/run


----

**See also:** :doc:`pipeline_lifecycle` · :doc:`run_on_clusters` · :doc:`injection_infrastructure`

**Next:** :doc:`injection_infrastructure` — how to configure simulated signals
