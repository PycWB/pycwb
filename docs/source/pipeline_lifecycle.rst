.. _pipeline_lifecycle:

Pipeline Lifecycle
==================

This page walks through the complete pycWB analysis pipeline—from raw detector
data to final detection efficiency. Think of it as a map: each step links to
the detailed guide where you can learn more.

The stages below are pycWB's modular Python implementation of the same
cWB/cWB-2G search algorithms: WDM time-frequency analysis, coherent pixel
selection, clustering and superclustering, likelihood evaluation, waveform
reconstruction, and postproduction ranking.

.. contents:: Table of Contents
   :depth: 2
   :local:


Overview
--------

.. image:: _static/diagrams/search_pipeline.svg
   :alt: Production search pipeline


cWB-2G Stage Correspondence
---------------------------

The ROOT/C++ cWB-2G notes describe the production algorithm as a sequence of
named stages: data conditioning, WDM setup, coherence, supercluster, and
likelihood. pycWB keeps that same algorithmic decomposition, but stores the
intermediate objects in Python data structures, Parquet catalogs, and trigger
files instead of ROOT job-file cycles.

.. list-table::
   :header-rows: 1
   :widths: 22 42 36

   * - cWB-2G stage
     - Algorithmic role
     - pycWB implementation
   * - Data conditioning
     - Read detector strain, add configured injections, remove lines, estimate
       detector noise RMS, and whiten the data.
     - :py:mod:`pycwb.modules.data_conditioning`
   * - WDM and MRA setup
     - Initialize WDM transforms for each resolution level and load the
       cross-resolution MRA/XTalk catalog used by reconstruction.
     - :py:mod:`pycwb.modules.coherence_native`,
       :py:mod:`pycwb.modules.multi_resolution_wdm`,
       :py:mod:`pycwb.modules.xtalk`
   * - Coherence
     - Build time-frequency maps, compute maximum coherent energy, set the
       black-pixel threshold, select significant pixels per lag, and perform
       single-resolution clustering.
     - :py:mod:`pycwb.modules.coherence_native`
   * - Supercluster
     - Merge per-resolution clusters, compute time-delay amplitudes, apply the
       sub-network cut, and defragment nearby structures.
     - :py:mod:`pycwb.modules.super_cluster_native`
   * - Likelihood
     - Loop over surviving clusters, scan sky directions, evaluate the coherent
       likelihood, reconstruct the waveform, and write event parameters.
     - :py:mod:`pycwb.modules.likelihoodWP`,
       :py:mod:`pycwb.modules.reconstruction`


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

In cWB-2G terminology, this stage produces the whitened detector strain
(``HoT``) and detector noise estimate (``nRMS``). pycWB carries the same
algorithmic products forward as conditioned strain series and per-detector
nRMS maps.

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

The WDM transforms define the time-frequency basis. The MRA/XTalk catalog is a
separate sparse cross-resolution coupling table used later to remove duplicated
support between resolutions and reconstruct waveforms.

- Config: ``l_low``, ``l_high``, ``levelR``
- Module: :py:mod:`pycwb.modules.coherence_native`

→ Output: time-frequency pixels (amplitude vs. time vs. frequency vs. detector).


5. Coherence & Pixel Selection
------------------------------

For each time-frequency pixel, the coherent energy across the detector network
is computed. The data is time-shifted for each sky direction to account for
gravitational-wave travel time differences between detectors. Pixels with
excess coherent power are selected.

This corresponds to the cWB-2G ``maxEnergy`` → threshold → significant-pixel
selection path. Selected pixels are clustered at each resolution before the
multi-resolution supercluster step.

- Config: ``bpp``, ``pattern``, ``BATCH``
- Module: :py:mod:`pycwb.modules.coherence_native`

→ Output: selected pixels above threshold, grouped by resolution.


6. Clustering & Superclustering
-------------------------------

Selected pixels are grouped into **clusters** (per resolution level) and then
merged into **superclusters** across resolutions. A sub-network cut removes
clusters unlikely to be astrophysical.

This is the pycWB equivalent of the cWB-2G ``netcluster::supercluster`` stage:
merge across resolutions, attach time-delay amplitudes, apply
``subNetCut``, and defragment surviving clusters.

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

This corresponds to the cWB-2G ``likelihood2G`` / ``likelihoodWP`` stage:
loop over superclusters, attach time-delay amplitudes to pixels, evaluate the
coherent network likelihood, and output reconstructed event parameters.

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
