.. _pipeline_lifecycle:

Pipeline Lifecycle
==================

This page walks through the complete pycWB analysis pipeline—from raw detector
data to final detection efficiency. Think of it as a map: each step links to
the detailed guide where you can learn more.

.. contents:: Table of Contents
   :depth: 2
   :local:


Overview
--------

.. mermaid::

   flowchart LR
     A[WaveSegment<br/>frames, noise, injections,<br/>GPS times, IFOs] --> B[Prepare Time Series]

     subgraph prep[Data preparation]
       B1[Read frame data]
       B2[Generate noise]
       B3[Add injections]
       B4[Regression / conditioning]
       B5[Whitening]
       B1 --> B4 --> B5
       B2 --> B5
       B3 --> B5
     end

     prep --> B

     B --> C[Time-frequency transform]
     C --> D[Coherence analysis]
     D --> E[Select significant pixels]
     E --> F[Cluster pixels]
     F --> G[Supercluster]
     G --> H[Likelihood reconstruction]
     H --> I[Events and cluster statistics]


1. Data Ingestion
-----------------

Raw gravitational-wave strain data is read from frame files (``.gwf``) or
streamed from NDS2 servers. The data is resampled to the analysis rate
(``inRate``) and split into detector-specific time series.

- Config: ``frFiles``, ``gwdatafind``, ``inRate``
- Module: :py:mod:`pycwb.modules.read_data`

→ Next: the data time series is split into **segments** for parallel processing.


2. Segment Construction
-----------------------

The continuous data stream is divided into overlapping time windows called
**job segments**. Each segment is an independent unit of work that can run on a
separate cluster node.

- DQ files (CAT0/1/2) define valid science time
- ``segLen``, ``segMLS``, ``segEdge`` control segment boundaries
- Frame files are matched to each segment's GPS window

→ Each segment becomes a **job**, optionally replicated across lags and trials.
See :ref:`job_control`.


3. Data Conditioning
--------------------

Within each segment, the data is prepared for wavelet analysis:

1. **Resampling** to the target rate
2. **Regression** to remove slow instrumental drifts
3. **Whitening** to flatten the noise spectrum

- Methods: wavelet whitening, MESA spectral estimation, or mixed
- Config: ``whiteMethod``, ``whiteWindow``, ``mesaOrder``
- Module: :py:mod:`pycwb.modules.data_conditioning`

→ Output: whitened time series ready for time-frequency decomposition.


4. Time-Frequency Transform
----------------------------

The whitened data is transformed into the time-frequency domain using the
**Wilson-Daubechies-Meyer (WDM)** wavelet transform. Multiple resolution
levels are computed (from ``l_low`` to ``l_high``) to capture signals of
different durations.

- Config: ``l_low``, ``l_high``, ``levelR``
- Module: :py:mod:`pycwb.modules.coherence_native`

→ Output: time-frequency pixels (amplitude vs. time vs. frequency vs. detector).


5. Coherence & Pixel Selection
------------------------------

For each time-frequency pixel, the coherent energy across the detector network
is computed. The data is time-shifted for each sky direction to account for
gravitational-wave travel time differences between detectors. Pixels with
excess coherent power are selected.

- Config: ``bpp``, ``pattern``, ``BATCH``
- Module: :py:mod:`pycwb.modules.coherence_native`

→ Output: selected pixels above threshold, grouped by resolution.


6. Clustering & Superclustering
-------------------------------

Selected pixels are grouped into **clusters** (per resolution level) and then
merged into **superclusters** across resolutions. A sub-network cut removes
clusters unlikely to be astrophysical.

- Config: ``TFgap``, ``Tgap``, ``Fgap``, ``subnet``, ``subcut``
- Module: :py:mod:`pycwb.modules.super_cluster_native`

→ Each supercluster becomes a candidate event. See :ref:`clustering_algorithm`.


7. Likelihood Evaluation
------------------------

For each supercluster, the likelihood pipeline:

1. Scans all sky directions using precomputed time delays
2. Projects data onto the Dominant Polarization Frame (DPF)
3. Computes SNR (:math:`\rho`), network correlation (:math:`cc`), :math:`\chi^2`
4. Selects the best-fit sky position
5. Reconstructs the waveform and computes :math:`h_{rss}`

- Config: ``netRHO``, ``netCC``, ``delta``, ``cfg_gamma``, ``healpix``
- Module: :py:mod:`pycwb.modules.likelihoodWP`

→ Each supercluster becomes an **event** in the trigger catalog.
See :ref:`likelihood_guide`.


8. Event Output
---------------

Events passing thresholds are written to the Parquet trigger catalog
(``catalog/catalog.parquet``) and per-event JSON files (``trigger/``).
Progress metadata is written to ``catalog/progress.parquet``.

- Each event includes: GPS time, frequency, sky position, SNR, :math:`\chi^2`, network correlation
- When ``Search`` is CBC/BBH/IMBHB: chirp mass is also computed

→ Jobs complete. Postproduction begins.


9. Background Estimation
------------------------

Non-zero-lag triggers from all jobs are collected. The false alarm rate (FAR)
is computed as a function of ranking statistic. The background livetime is
the total analyzed time across all non-zero-lag analyses.

- :math:`FAR(\rho^*) = N_{bkg}(\rho \ge \rho^*) / T_{bkg}`
- Train/FAR splitting ensures unbiased estimation

→ See :ref:`postproduction_background`.


10. Ranking & Detection Efficiency
----------------------------------

An XGBoost classifier is trained on background + simulation events to produce
a single **ranking statistic**. This statistic is used to:

1. Assign FAR to each event
2. Measure detection efficiency vs. signal amplitude (:math:`h_{rss}`)
3. Compute hrss50/hrss90 sensitivity figures

→ See :ref:`postproduction_xgboost` and :ref:`postproduction_efficiency`.


Where Each Config Parameter Lives
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Pipeline Stage
     - Key Parameters
     - Detailed Guide
   * - Data Ingestion
     - ``frFiles``, ``gwdatafind``, ``inRate``
     - :ref:`standard_analysis`
   * - Segments & Jobs
     - ``segLen``, ``lagSize``, ``lagStep``, ``lagOff``
     - :ref:`job_control`
   * - Conditioning
     - ``whiteMethod``, ``mesaOrder``
     - :ref:`schema`
   * - TF Transform
     - ``l_low``, ``l_high``, ``levelR``
     - :ref:`schema`
   * - Clustering
     - ``TFgap``, ``Tgap``, ``Fgap``, ``subnet``
     - :ref:`clustering_algorithm`
   * - Likelihood
     - ``netRHO``, ``netCC``, ``healpix``, ``delta``
     - :ref:`likelihood_guide`
   * - Postproduction
     - Workflow YAML, train fraction, FAR threshold
     - :ref:`postproduction`


----

**See also:** :doc:`job_control` · :doc:`clustering_algorithm` · :doc:`likelihood_guide` · :doc:`postproduction`

**Next:** :doc:`job_control` — how pycWB splits work into segments, lags, and trials
