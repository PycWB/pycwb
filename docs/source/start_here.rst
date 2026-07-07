.. _start_here:

Start Here
==========

Welcome to pycWB! This page gets you from zero to a working gravitational-wave
burst search in about 10 minutes.

.. contents:: Table of Contents
   :depth: 2
   :local:


What is a cWB / pycWB Search?
------------------------------

**Coherent Wave Burst (cWB)** is an algorithm that searches for
gravitational-wave bursts—short, unmodeled signals from sources like
supernovae, gamma-ray bursts, or unexpected phenomena. Unlike template-based
searches (which look for specific waveforms), cWB looks for **any excess
coherent power** across a network of detectors.

**pycWB** is the Python implementation. It takes strain data from LIGO, Virgo,
and KAGRA, transforms it into time-frequency maps, finds clusters of excess
power, and evaluates a likelihood that each cluster is a real signal.

The key idea: a real gravitational wave appears **coherently** in all detectors
(with appropriate time delays), while instrumental noise is uncorrelated.
pycWB exploits this to separate signals from noise.


The Five Key Objects
--------------------

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Object
     - Where Defined
     - What It Represents
   * - **Config**
     - ``user_parameters.yaml``
     - All settings: detectors, frequency range, data source, thresholds, injection parameters. One file controls everything.
   * - **Segment**
     - Built from config
     - A GPS time window of detector data. A search is split into many independent segments.
   * - **Job**
     - segment × lag × trial
     - One unit of computation submitted to a cluster.
   * - **Event**
     - Likelihood pipeline output
     - A candidate trigger: time, frequency, sky position, SNR, ranking statistic.
   * - **Postproduction**
     - Workflow YAML + Parquet files
     - After jobs finish: background estimation, ranking, efficiency, final report.


Minimal Installation Check
--------------------------

.. code-block:: bash

   python -c "import pycwb; print(pycwb.__version__)"
   pycwb --help
   python -c "import jax; print('JAX OK:', jax.__version__)"

If any fail, see :ref:`installing_pycwb`.


First Run in 10 Minutes
-----------------------

Uses built-in noise and waveform generators—no real data needed.

.. code-block:: bash

   # 1. Copy the injection example
   cp -r examples/injection my_first_search
   cd my_first_search

   # 2. Run the search
   pycwb run user_parameters_injection.yaml

   # 3. Inspect results
   ls catalog/        # catalog.parquet — trigger list
   ls trigger/        # Per-event JSON files
   ls log/            # Run log

The example injects simulated signals into Gaussian noise and recovers them
in under a minute on a modern laptop.

**What just happened:**

1. pycWB read ``user_parameters_injection.yaml``.
2. It generated synthetic noise and injected simulated signals.
3. It ran the full pipeline: wavelet transform → pixel clustering → likelihood → triggers.
4. Results written to ``catalog/`` as Parquet files.


Understanding the Output
------------------------

``catalog/catalog.parquet``
   Main trigger catalog—each row is a candidate event. Open with:

   .. code-block:: python

      import pandas as pd
      df = pd.read_parquet("catalog/catalog.parquet")
      print(df.columns)

``trigger/*.json``
   Per-event details: waveform reconstruction, pixel maps, sky localization.

``catalog/progress.parquet``
   Processing metadata: which jobs/lags ran, duration of each.

``log/``
   Log files—check here first when something fails.


Common Parameters
-----------------

For your first real search, you'll mainly change these:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Example
     - What It Does
   * - ``gps_start`` / ``gps_end``
     - 1264060000 / 1264063600
     - Time window to analyze
   * - ``fLow`` / ``fHigh``
     - 64 / 2048
     - Frequency range [Hz]
   * - ``ifos``
     - [H1, L1]
     - Which detectors to use
   * - ``lagSize`` / ``lagStep``
     - 100 / 1.0
     - Number of time-shifts for background estimation
   * - ``segLen``
     - 600
     - Length of each analysis segment [s]
   * - ``netRHO``
     - 4.0
     - SNR threshold (lower = more triggers, more background)
   * - ``cluster``
     - condor or slurm
     - Batch system for cluster submission
   * - ``healpix``
     - 7
     - Sky map resolution (higher = finer but slower)

Most other parameters have sensible defaults.


Common Mistakes
---------------

**"My jobs fail with 'frame file not found'"**
   Check ``frFiles`` or ``gwdatafind`` config. Make sure frame paths point to
   valid ``.gwf`` files covering your GPS time window.

**"I get zero triggers"**
   Lower ``netRHO`` to 3–4 for initial tests. Verify ``fLow``/``fHigh`` match
   your data's sample rate. For injection runs, check injection parameters.

**"The run is extremely slow"**
   Reduce ``healpix`` (try 5–6). Reduce ``lagSize`` for testing. Enable
   ``parallel_lag_workers``.

**"My background estimate looks wrong"**
   Verify ``lagOff`` excludes zero-lag. Check ``segLen`` and ``segOverlap``
   don't double-count livetime.

**"Simulations aren't being recovered"**
   Check injection GPS times fall within analysis segments. Verify
   ``iwindow`` is large enough. Lower ``netRHO`` temporarily.


Where to Go Next
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - If You Want To...
     - Read...
   * - Run a real data search
     - :ref:`standard_analysis` → Setup Config Templates
   * - Understand injection/simulation studies
     - :ref:`injection_infrastructure`
   * - Submit jobs to a cluster
     - :ref:`standard_analysis` → :ref:`run_on_clusters`
   * - Learn how the algorithm works
     - :ref:`core_concepts`
   * - Run postproduction on results
     - :ref:`postproduction`
   * - Look up a config parameter
     - :ref:`schema`
   * - Find a term's definition
     - :ref:`glossary`
   * - See complete workflow examples
     - :ref:`analysis_recipes`
