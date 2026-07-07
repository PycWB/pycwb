.. _modules_guide:

===============
Module Overview
===============

PycWB is built from ~40 self-contained modules organized by pipeline stage.
Each module lives in ``pycwb/modules/`` with its own ``tests/`` subdirectory
and communicates through plain Python objects and NumPy arrays.

For the full auto-generated API reference, see :doc:`pycwb.modules`.

.. tip::

   Modules marked **(legacy)** depend on ROOT/C++ cWB-core bindings and are
   being phased out in favour of native Python equivalents.
   Modules marked **(experimental)** are under active development and not yet
   production-ready.


Data I/O
========

.. list-table::
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`read_data <pycwb.modules.read_data>`
     - Reads GW strain data from frame files (GWF), online NDS2 servers via gwpy,
       or generates synthetic noise for simulations. Also supports MDC
       (mock data challenge) injection I/O and data quality flag checking.
   * - :doc:`injection <pycwb.modules.injection>`
     - Generates and schedules simulated GW signal injections. Builds injection
       parameter lists from config, handles sky distribution sampling,
       time-of-arrival distributions, and assigns injections to specific job
       segments and trials.
   * - :doc:`gwosc <pycwb.modules.gwosc>`
     - Interfaces with the GW Open Science Center (GWOSC) API. Retrieves public
       event metadata (GPS time, detectors, science segments) and downloads
       frame files for known GW events.
   * - :doc:`cwb_results <pycwb.modules.cwb_results>`
     - Reads and summarizes legacy cWB ``liveTime`` ROOT files. Computes
       live-time statistics (total seconds, losses, min/max, counts per
       threshold). **(legacy — ROOT-backed)**
   * - :doc:`gracedb <pycwb.modules.gracedb>`
     - Interfaces with LIGO/Virgo's GraceDB alert system. Retrieves superevent
       metadata and GPS times, and uploads online search triggers as new
       GraceDB events for rapid follow-up.
   * - :doc:`noise <pycwb.modules.noise>`
     - Generates coloured Gaussian noise from arbitrary PSDs using
       ``lalsimulation.SimNoise`` (same engine as PyCBC). Loads PSDs from text
       files and evaluates analytic noise models (aLIGO, Einstein Telescope, etc.).


Signal Conditioning
===================

.. list-table::
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`data_conditioning <pycwb.modules.data_conditioning>`
     - Pure-Python data conditioning pipeline: regression (line removal) followed
       by whitening (wavelet-based or MESA). Operates per-lag on native NumPy
       time series. The production conditioning engine.
   * - :doc:`data_conditioning_root <pycwb.modules.data_conditioning_root>`
     - ROOT-backed data conditioning using cWB C++ regression and whitening
       routines via ROOT bindings. Supports parallel processing with
       multiprocessing. **(legacy — ROOT-backed)**


Coherence & Clustering
======================

.. list-table::
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`coherence_native <pycwb.modules.coherence_native>`
     - Production coherence engine. JAX-accelerated WDM time→frequency
       transforms, max-energy computation, threshold-based pixel selection,
       veto application, and single-resolution pixel clustering. Builds lag
       plans from config.
   * - :doc:`coherence_root <pycwb.modules.coherence_root>`
     - ROOT-backed coherence wrapping cWB's C++ ``network::getNetworkPixels``
       and ``network::cluster``. Loops over resolution levels and lags using
       ROOT WSeries/WDM objects. **(legacy — ROOT-backed)**
   * - ``clustering``
     - Next-generation pixel clustering algorithms: connected components,
       DBSCAN, HDBSCAN, OPTICS, and MRA weighted-graph clustering.
       **(experimental — not yet implemented)**
   * - :doc:`super_cluster_native <pycwb.modules.super_cluster_native>`
     - Native multi-resolution super-clustering. Merges pixel clusters across
       resolution levels, applies sub-net cuts, and defragments using
       Numba-accelerated link-matrix computation.
   * - :doc:`super_cluster_root <pycwb.modules.super_cluster_root>`
     - ROOT-backed super-clustering wrapping cWB's ``netcluster::supercluster``.
       Handles sparse table creation, sky-map resolution, sub-net cuts, and
       defragmentation. **(legacy — ROOT-backed)**
   * - :doc:`sparse_series <pycwb.modules.sparse_series>`
     - Creates sparse time-frequency representations from fragment clusters.
       Extracts pixel-level data (time, frequency, amplitude, phase) from TF
       maps at each resolution level for efficient downstream processing.
   * - :doc:`multi_resolution_wdm <pycwb.modules.multi_resolution_wdm>`
     - Creates and manages the Wavelet Domain Model (WDM) for all resolution
       levels. Validates filter lengths against segment edges and time-delay
       sizes. Wraps ``pycwb.types.wdm.WDM``.


Likelihood
==========

.. list-table::
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`likelihoodWP <pycwb.modules.likelihoodWP>`
     - Numba-accelerated coherent likelihood on CPU. Computes sky localization
       (full HEALPix scan), coherent SNR, null energy, correlation, and
       per-cluster detection statistics using per-pixel time-delay data and
       antenna patterns. Uses ``@njit`` + ``prange`` for inner loops.
   * - :doc:`likelihoodWPGPU <pycwb.modules.likelihoodWPGPU>`
     - GPU-optimized likelihood, drop-in replacement for ``likelihoodWP``.
       Uses ``jax.vmap`` for batched sky scans, ``jax.jit`` compilation, and
       float32 throughout. Device-agnostic — runs on CPU, GPU, or TPU without
       code changes.
   * - :doc:`likelihood_root <pycwb.modules.likelihood_root>`
     - ROOT-backed likelihood wrapping cWB's C++ ``network::likelihood``.
       Iterates over clusters, converting between native and ROOT types via
       ``cwb_conversions``. **(legacy — ROOT-backed)**


Post-processing
===============

.. list-table::
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`postprocess <pycwb.modules.postprocess>`
     - Comprehensive post-production analysis suite. Trains and evaluates
       XGBoost classifiers, computes efficiency curves (hrss50 via sigmoid
       fit), calculates false-alarm rates, runs fake open-box studies, and
       generates automated reports.
   * - :doc:`reconstruction <pycwb.modules.reconstruction>`
     - Reconstructs GW waveforms from coherent pixel sums using multi-resolution
       analysis (MRA). Computes injected waveform statistics, residuals,
       matched-filter SNR, and amplitude spectral densities (ASD).
   * - :doc:`cwb_xgboost <pycwb.modules.cwb_xgboost>`
     - XGBoost-based ranking and classification of GW triggers. Reads Parquet
       catalogs, builds feature matrices with train/test splitting and
       balanced sampling, trains classifiers, and evaluates with ROC/PR curves.
   * - :doc:`autoencoder <pycwb.modules.autoencoder>`
     - Neural-network glitch classifier. Computes a "glitchness" score — a
       per-cluster metric indicating how glitch-like the reconstructed waveform
       is — to help reject non-astrophysical triggers.
   * - :doc:`qveto <pycwb.modules.qveto>`
     - Data quality veto metrics. Computes Qveto and Qfactor from reconstructed
       waveforms using zero-crossing segment-maxima analysis and time-domain
       energy ratios to identify glitch-like signals.
   * - :doc:`statistics <pycwb.modules.statistics>`
     - Statistical tools for GW detection efficiency. Provides sigmoid fitting
       (via ``iminuit``), efficiency curve computation (hrss percentiles),
       chunk merging for distributed results, and efficiency plot generation.


Infrastructure & Workflow
=========================

.. list-table::
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`catalog <pycwb.modules.catalog>`
     - Primary I/O layer. Arrow/Parquet-based trigger and event catalog with
       schema metadata, atomic writes via ``SoftFileLock``, deduplication on
       merge, and SQL/DuckDB query support. Also provides JSON catalog format
       and trigger-to-simulation matching.
   * - :doc:`config_repo_parser <pycwb.modules.config_repo_parser>`
     - Parses cWB project names into structured components (observation run,
       chunk ID, DQ category, search path, label). Extracts GPS times from
       chunk files and sets up project directories with full configuration.
   * - :doc:`logger <pycwb.modules.logger>`
     - Initializes PycWB's structured logging. Configures log format, output
       destination (file/stdout), and log level. Pins noisy external libraries
       (JAX, Numba, matplotlib) to WARNING to keep logs readable.
   * - :doc:`workflow_utils <pycwb.modules.workflow_utils>`
     - Trigger persistence utilities. Creates organized folder structures by job
       segment, trial, GPS time, and hash ID. Saves event, cluster, and skymap
       data as JSON and registers triggers in the Parquet catalog.
   * - :doc:`skymask <pycwb.modules.skymask>`
     - Creates circular sky masks on HEALPix grids for targeted GW searches.
       Converts cWB sky coordinates (phi/theta) to geographic coordinates and
       fills mask pixels within a specified angular radius.
   * - :doc:`superlag <pycwb.modules.superlag>`
     - Generates super-lag (slag) combinations — multi-detector time-shift
       patterns used for background estimation. Computes shift combinations
       sorted by slag distance with configurable min/max distance and offset.
   * - :doc:`xtalk <pycwb.modules.xtalk>`
     - Cross-talk catalog management. Loads pre-computed crosstalk coefficient
       files (binary or .npz) and provides fast Numba-accelerated lookup of
       crosstalk coefficients for pixel pairs.
   * - :doc:`condor <pycwb.modules.condor>`
     - Generates and submits HTCondor DAG batch jobs for distributed analysis.
       Creates job scripts, merge scripts, and simulation summary scripts with
       configurable resource requests (memory, CPUs, disk).
   * - :doc:`slurm <pycwb.modules.slurm>`
     - Generates and submits Slurm job arrays for distributed analysis. Creates
       job scripts with configurable partitions, constraints, and resource
       allocations for HPC clusters.
   * - :doc:`online <pycwb.modules.online>`
     - Full streaming GW search pipeline. Components: ``DataSource`` (abstract
       data adapter), ``DataAcquisitionManager`` (polling daemon),
       ``RingBuffer`` (thread-safe per-IFO buffer), ``TriggerHandler``
       (deduplication + significance + GraceDB alerts), and ``LatencyMonitor``.
   * - :doc:`plot <pycwb.modules.plot>`
     - Visualization toolkit. Spectrograms, 1D/2D histograms, event overlays,
       detector antenna patterns, globe plots, fragment cluster visualization,
       and data quality diagnostic plots.
   * - :doc:`cwb_conversions <pycwb.modules.cwb_conversions>`
     - Bidirectional type conversion between native PycWB types (NumPy arrays,
       ``TimeSeries``, ``FragmentCluster``, ``PixelArrays``) and legacy cWB
       ROOT types (``wavearray``, ``WSeries``, ``netcluster``, ``sseries``).
   * - :doc:`cwb_interop <pycwb.modules.cwb_interop>`
     - Creates standalone cWB working directories for direct numerical
       comparison between PycWB and original cWB runs. Generates equivalent
       ``user_parameters.C`` configs, frame file lists, and DQ files.
   * - :doc:`external_module_manager <pycwb.modules.external_module_manager>`
     - Manages installation and versioning of external PycWB modules from Git
       repositories. Loads module config from YAML, checks existence, and
       pulls/clones external modules into the PycWB modules directory.
   * - :doc:`job_segment <pycwb.modules.job_segment>`
     - Constructs the analysis job segmentation. Reads DQ segment lists, builds
       job segments with frame file selection, injection scheduling, and
       super-lag generation. Handles flattening by trial index and CAT2 veto
       windows.
