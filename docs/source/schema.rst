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
     - ``ifo``, ``gps_start``, ``gps_end``, ``inRate``
     - :ref:`start_here`
   * - Detector Geometry
     - ``ifo``, ``refIFO``, ``detector_geometry``, ``detector_definitions_file``
     - :ref:`detector_support`
   * - Execution / Memory
     - ``execution_profile``, ``max_energy_backend``, ``coherence_timing``
     - :ref:`execution_profile`
   * - Calculation Conventions
     - ``execution_profile.native_chirp``, ``release_waveform_stats``,
       ``regression_cap``, ``regression_percentile_stride`` (all within the profile)
     - :ref:`native_calculation_conventions`
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
   * - Workflow Scheduling (experimental)
     - ``execution.profile``, ``execution.worker_memory``, ``execution.preload``
     - :ref:`workflow_execution`
   * - Postproduction
     - Workflow YAML (separate file)
     - :ref:`postproduction`

Parameters marked :math:`^*` are auto-derived (``rateANA``, ``nRES``,
``WDM_level``, ``max_delay``) — do **not** set them manually.


Common parameters
-----------------

The following table is generated from the same schema used to validate YAML.
These are software defaults, not a recommended production search configuration.
For example, explicitly choose your detector network rather than relying on the
schema's generic detector list. ``iwindow`` is the full injection window:
``Tinj - iwindow/2`` through ``Tinj + iwindow/2``.

.. exec::

    from pycwb.constants import user_parameters_schema
    from pycwb.utils.generate_params_table import generate_rst_table
    keys = ['ifo', 'refIFO', 'inRate', 'fLow', 'fHigh', 'levelR', 'l_low', 'l_high',
            'segLen', 'segMLS', 'segEdge', 'lagSize', 'lagOff', 'healpix',
            'whiteMethod', 'whiteWindow', 'netRHO', 'netCC', 'subcut', 'iwindow',
            'nproc', 'job_memory', 'detector_geometry', 'detector_definitions_file',
            'max_energy_backend', 'coherence_timing']
    print(generate_rst_table({key: user_parameters_schema['properties'][key] for key in keys}))



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
