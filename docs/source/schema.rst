.. _schema:

User Parameters
===============

All pycWB configuration lives in ``user_parameters.yaml``, validated against
a JSON Schema. This page organizes parameters by category and links to the
detailed guides where each parameter is explained in context.

.. contents:: Table of Contents
   :depth: 2
   :local:


Parameter Categories
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Category
     - Key Parameters
     - Detailed Guide
   * - General / Network
     - ``ifos``, ``gps_start``, ``gps_end``, ``inRate``
     - :ref:`start_here`
   * - Frequency / Resolution
     - ``fLow``, ``fHigh``, ``l_low``, ``l_high``, ``levelR``
     - :ref:`pipeline_lifecycle`
   * - Segment & Job
     - ``segLen``, ``segMLS``, ``segEdge``, ``segOverlap``
     - :ref:`job_control`
   * - Lags & Background
     - ``lagSize``, ``lagStep``, ``lagOff``, ``lagMax``, ``slagSize``
     - :ref:`job_control`
   * - Data Conditioning
     - ``whiteMethod``, ``whiteWindow``, ``mesaOrder``
     - :ref:`pipeline_lifecycle`
   * - Clustering
     - ``TFgap``, ``Tgap``, ``Fgap``, ``subnet``, ``subcut``
     - :ref:`clustering_algorithm`
   * - Likelihood
     - ``netRHO``, ``netCC``, ``healpix``, ``delta``, ``cfg_gamma``
     - :ref:`likelihood_guide`
   * - Injection
     - ``injection``, ``iwindow``, ``simulation``
     - :ref:`injection_infrastructure`
   * - Sky Mask
     - ``sky_mask``, ``EFEC``
     - :ref:`targeted_search`
   * - Batch / Cluster
     - ``cluster``, ``conda_env``, ``job_memory``, ``accounting_group``
     - :ref:`run_on_clusters`
   * - Postproduction
     - Workflow YAML (separate file)
     - :ref:`postproduction`

Parameters marked :math:`^*` are auto-derived (``rateANA``, ``nRES``,
``WDM_level``, ``max_delay``) — do **not** set them manually.


Parameter Quick Reference
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 12 12 54

   * - Parameter
     - Default
     - Range
     - Description
   * - ``ifos``
     - [H1, L1]
     - —
     - List of interferometer names
   * - ``fLow``
     - 64
     - ≥ 0
     - Low frequency [Hz]
   * - ``fHigh``
     - 2048
     - > fLow
     - High frequency [Hz]
   * - ``inRate``
     - 16384
     - —
     - Input data sample rate [Hz]
   * - ``l_low``
     - 3
     - 1–10
     - Lowest WDM resolution level (2\ :sup:`l_low` Hz)
   * - ``l_high``
     - 8
     - > l_low
     - Highest WDM resolution level (2\ :sup:`l_high` Hz)
   * - ``levelR``
     - 2
     - —
     - Resampling level
   * - ``segLen``
     - 600
     - > 0
     - Segment length [s]
   * - ``segMLS``
     - 300
     - ≤ segLen
     - Minimum segment after CAT1 [s]
   * - ``segEdge``
     - 8
     - ≥ 0
     - Wavelet boundary offset [s]
   * - ``segOverlap``
     - 0
     - ≥ 0
     - Overlap between jobs [s]
   * - ``lagSize``
     - 1
     - ≥ 0
     - Number of lags
   * - ``lagStep``
     - 1.0
     - > 0
     - Time between lags [s]
   * - ``lagOff``
     - 6
     - ≥ 0
     - First lag index (0 = include zero-lag)
   * - ``lagMax``
     - 150
     - ≥ lagStep
     - Maximum lag distance [s]
   * - ``slagSize``
     - 0
     - ≥ 0
     - Number of super lags
   * - ``whiteMethod``
     - wavelet
     - wavelet/mesa/mixed
     - Whitening method
   * - ``whiteWindow``
     - 60
     - > 0
     - Whitening time window [s]
   * - ``Acore``
     - 1.414
     - ≥ 0
     - Core pixel threshold (:math:`\sqrt{2}`)
   * - ``netRHO``
     - 4.0
     - > 0
     - Coherent network SNR threshold
   * - ``netCC``
     - 0.5
     - 0–1
     - Network correlation threshold
   * - ``delta``
     - 0.5
     - −1 to 1
     - 2-detector sky regulator
   * - ``cfg_gamma``
     - 0.5
     - −1 to 1
     - Polarization suppression regulator
   * - ``healpix``
     - 7
     - 1–12
     - Sky map HEALPix order
   * - ``TFgap``
     - 6.0
     - ≥ 0
     - TF pixel separation for cluster linking
   * - ``Tgap``
     - 3.0
     - ≥ 0
     - Defragmentation time gap [s]
   * - ``Fgap``
     - 130
     - ≥ 0
     - Defragmentation frequency gap [Hz]
   * - ``subnet``
     - 0.7
     - 0–0.7
     - Sub-network coherence threshold
   * - ``subcut``
     - 0.33
     - 0–1
     - Sub-network skyloop threshold
   * - ``bpp``
     - 0.001
     - 0–1
     - Black pixel selection probability
   * - ``TDSize``
     - 12
     - 2–20
     - Time-delay filter size
   * - ``upTDF``
     - 4
     - ≥ 1
     - TD filter upsample factor
   * - ``iwindow``
     - 5.0
     - > 0
     - Injection time window half-width [s]
   * - ``cluster``
     - —
     - condor / slurm
     - Batch submission backend
   * - ``nproc``
     - 1
     - ≥ 1
     - CPUs per job
   * - ``job_memory``
     - 6GB
     - —
     - Memory per job
   * - ``Search``
     - ""
     - "" / CBC / BBH / IMBHB
     - Enable chirp mass computation
   * - ``xgb_rho_mode``
     - false
     - —
     - Use :math:`\rho_0` instead of :math:`\rho`
   * - ``EFEC``
     - true
     - —
     - Earth-fixed/celestial coordinate conversion


Auto-Derived Fields
-------------------

These are computed automatically from other parameters. **Do not set manually.**

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Derived From
   * - ``rateANA``
     - ``inRate`` ÷ 2\ :sup:`levelR`
   * - ``nRES``
     - ``l_high`` − ``l_low`` + 1
   * - ``WDM_level``
     - ``l_low`` … ``l_high`` (list)
   * - ``max_delay``
     - Detector geometry (baseline ÷ c)


Full Schema
-----------

The complete auto-generated parameter table:

.. exec::
    import json
    from pycwb.constants import user_parameters_schema
    from pycwb.utils.generate_params_table import generate_rst_table, parse_description, parse_type_or_enum

    print(generate_rst_table(user_parameters_schema["properties"]))
