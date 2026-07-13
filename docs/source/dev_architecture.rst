.. _dev_architecture:

Architecture Overview
=====================

This page maps pycWB's internal structure—entry points, pipeline stages,
module organization, and key design conventions.

.. contents:: Table of Contents
   :depth: 2
   :local:


Project Layout
--------------

.. code-block:: text

   pycwb/
   ├── config/             # Config loading, schema validation
   ├── constants/          # Enums, schema defaults, derived-field logic
   ├── modules/            # Pipeline stages (one sub-package each)
   │   ├── read_data/
   │   ├── data_conditioning/
   │   ├── coherence/
   │   ├── super_cluster_native/
   │   ├── likelihoodWP/
   │   ├── catalog/
   │   ├── injection/
   │   ├── postprocess/
   │   └── ...
   ├── types/              # Data classes: WaveSegment, Cluster, PixelArrays, etc.
   ├── utils/              # Shared utilities: time-delay vectors, ROOT checks
   ├── workflow/           # Orchestration: run.py, batch.py, online.py
   │   └── subflow/        # Per-job pipeline: process_job_segment.py
   └── post_production/    # YAML-driven workflow engine
   cwb-core/               # C++ wavelet/ROOT core (being phased out)
   tests/                  # Integration & numerical parity tests


Entry Points
------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Command
     - Purpose
     - Key File
   * - ``pycwb run <config.yaml>``
     - Offline batch analysis (single machine)
     - ``pycwb/workflow/run.py``
   * - ``pycwb online <config.yaml>``
     - Live streaming analysis
     - ``pycwb/workflow/online.py``
   * - ``pycwb batch-setup``
     - Generate Condor/SLURM scripts
     - ``pycwb/workflow/batch.py``
   * - ``pycwb config-setup``
     - Create project from config repo
     - ``pycwb/cli/config_setup.py``
   * - ``pycwb post-process``
     - Run postproduction workflow
     - ``pycwb/post_production/workflow.py``
   * - ``pycwb server``
     - Results viewer / catalog tools
     - In development


Pipeline Stages (per job)
-------------------------

The core analysis runs for each job (segment × lag × trial):

.. code-block:: text

   1. read_data          → gwpy / NDS2 / frame files / synthetic noise
   2. data_conditioning   → resample → regression → whitening
   3. coherence           → WDM time→wavelet transform, max_energy (JAX)
   4. super_cluster       → merge pixel clusters, XTalk catalog
   5. likelihood          → sky scan, DPF, SNR + χ² statistics
   6. catalog             → Parquet atomic writes, progress tracking

Each stage is a self-contained module in ``pycwb/modules/``.

See :ref:`pipeline_lifecycle` for a high-level user-facing walkthrough.


Module Conventions
------------------

- Each module lives in ``pycwb/modules/<name>/`` with its own ``tests/`` subdirectory.
- Modules communicate through **plain Python objects and NumPy arrays** — avoid importing sideways between modules.
- Module entry functions are called from the pipeline orchestrator in ``pycwb/workflow/subflow/process_job_segment.py``.
- Config parameters flow downward from :py:class:`pycwb.config.Config` — modules read what they need, don't mutate global state.


Key Types
---------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Type
     - Description
   * - :py:class:`~pycwb.types.job.WaveSegment`
     - A GPS time window with data, injections, and lag parameters
   * - :py:class:`~pycwb.types.network_cluster.Cluster`
     - A group of time-frequency pixels at one resolution
   * - :py:class:`~pycwb.types.pixel_arrays.PixelArrays`
     - Struct-of-arrays layout for pixel data (time, frequency, rate, layers, etc.)
   * - :py:class:`~pycwb.types.time_series.TimeSeries`
     - Detector strain data with metadata
   * - :py:class:`~pycwb.config.Config`
     - Validated user parameters with auto-derived fields


C++ / ROOT Status
-----------------

The ``cwb-core/`` C++ code provides wavelet transforms and ROOT I/O. This is
**deprecated** and being replaced by pure-Python equivalents:

- ROOT I/O → Parquet (via ``pycwb/modules/catalog/``)
- C++ wavelet transforms → ``wdm-wavelet`` Python package
- All new code must work without ROOT; existing paths are guarded by
  :py:func:`pycwb.utils.check_ROOT.has_ROOT`.


Design Documents
----------------

Active design plans live in the repo root:

- ``INTRA_SEGMENT_PARALLELIZATION_PLAN.md`` — per-lag parallelism refactor
- ``PER_LAG_PROGRESS_PLAN.md`` — progress tracking per lag
- ``ONLINE_WORKFLOW_PLAN.md`` — streaming/online architecture
- ``DOCS_REDESIGN_PLAN.md`` — documentation structure plan
