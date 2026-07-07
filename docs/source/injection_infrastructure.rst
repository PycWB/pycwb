.. _injection_infrastructure:

Injection Infrastructure
========================

.. rubric:: Pipeline: :doc:`data <pipeline_lifecycle>` → :doc:`segments <job_control>` → :doc:`conditioning <pipeline_lifecycle>` → :doc:`WDM <pipeline_lifecycle>` → :doc:`pixels <clustering_algorithm>` → :doc:`clusters <clustering_algorithm>` → :doc:`likelihood <likelihood_guide>` → **[injections]** ← you are here → :doc:`events <pipeline_lifecycle>` → :doc:`bkg <postproduction_background>` → :doc:`ranking <postproduction_xgboost>` → :doc:`eff <postproduction_efficiency>`

This guide covers pycWB's flexible injection infrastructure for simulation
studies, including injection methods, sky distributions, time scheduling, and
waveform generation.

.. contents:: Table of Contents
   :depth: 2
   :local:


Why this matters
----------------

Injections are how we measure what the search can and cannot detect. Getting
injection parameters right is critical for valid sensitivity estimates. Every
efficiency number (hrss50, hrss90) depends on correct injection configuration.


Overview
--------

pycWB supports injecting simulated gravitational-wave signals into detector
data to measure search sensitivity and validate the analysis pipeline. The
injection system is configured through the ``injection`` block in
``user_parameters.yaml`` and is orchestrated by
:py:func:`pycwb.modules.injection.injection.generate_injection_list_from_config_for_job_segments`.


Injection Configuration
-----------------------

The top-level ``injection`` block accepts these keys:

.. code-block:: yaml

   injection:
     seed: 42                  # Random seed for reproducibility
     repeat_injection: 10      # Repeat each parameter set N times
     parameters:               # Static parameter list (see below)
       - mass1: 35
         mass2: 35
         ...
     parameters_from_python:   # Dynamic parameter generation (Python function)
       module: my_injections
       function: generate_params
     sky_distribution:         # Sky position sampling (see below)
       type: UniformAllSky
     time_distribution:        # Time scheduling (see below)
       type: poisson
       mean_interval: 1000.0
       max_trail: 50

If ``parameters_from_python`` is specified, it takes precedence over
``parameters``. The Python function must accept the injection config dict and
return a list of parameter dictionaries.


Injection Methods
-----------------

Static Parameter Lists
~~~~~~~~~~~~~~~~~~~~~~

Define fixed parameter sets directly in YAML:

.. code-block:: yaml

   injection:
     parameters:
       - mass1: 35
         mass2: 35
         spin1z: 0.0
         spin2z: 0.0
         approximant: IMRPhenomPv2
         f_lower: 20
         delta_t: 1.0/16384
         ra: 1.5
         dec: -0.5
         inclination: 0.0
         coa_phase: 0.0
         distance: 100.0
         hrss: 1e-21
         gps_time: 1264060000

Special replicators are available for parameter scanning:

- :py:func:`~pycwb.modules.injection.par_generator.repeat` — replicate each parameter set *n\_repeat*
  times (use the ``repeat_injection`` config key or call directly from Python).
- :py:func:`~pycwb.modules.injection.par_generator.inc_pol_replicator` — generate
  the Cartesian product of parameter sets with inclination and polarization
  angles.
- :py:func:`~pycwb.modules.injection.par_generator.hrss_scaling` — scale the waveform strain to
  target :math:`h_{rss}` values. The scale factor is computed as
  :math:`h_{rss}^{target} / \sqrt{\sum x_i^2}`.

Dynamic Parameter Generation (Python)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For complex sampling (e.g., mass distributions, random sky positions), provide
a Python function:

.. code-block:: yaml

   injection:
     parameters_from_python:
       module: my_injections
       function: generate_bbh_params

.. code-block:: python

   # my_injections.py
   import numpy as np

   def generate_bbh_params(injection_config):
       n = injection_config.get("n_signals", 100)
       params = []
       for _ in range(n):
           params.append({
               "mass1": np.random.uniform(10, 50),
               "mass2": np.random.uniform(10, 50),
               "approximant": "IMRPhenomPv2",
               "f_lower": 20,
               "delta_t": 1.0 / 16384,
               "ra": np.random.uniform(0, 2 * np.pi),
               "dec": np.arcsin(np.random.uniform(-1, 1)),
               "inclination": np.arccos(np.random.uniform(-1, 1)),
               "coa_phase": np.random.uniform(0, 2 * np.pi),
               "distance": np.random.uniform(100, 1000),
           })
       return params

The function receives the full injection config dict, so you can pass custom
parameters through it.


Waveform Generation
-------------------

pycWB uses the new **GWSignal-based waveform generator**
(:py:func:`pycwb.modules.injection.gwsignal_waveform.get_td_waveform`) as a
drop-in replacement for ``pycbc.waveform.get_td_waveform``. It translates
pycbc-style parameters to ``lalsimulation.gwsignal`` conventions internally:

.. list-table:: Parameter Mapping
   :header-rows: 1
   :widths: 25 25 50

   * - pycbc Parameter
     - GWSignal Parameter
     - Notes
   * - ``mass1``, ``mass2``
     - ``mass1``, ``mass2``
     - Direct pass-through
   * - ``spin1x/y/z``, ``spin2x/y/z``
     - ``spin1x/y/z``, ``spin2x/y/z``
     - Direct pass-through
   * - ``coa_phase``
     - ``phi_ref``
     - Coalescence phase → reference phase
   * - ``f_lower``
     - ``f22_start``
     - Starting frequency (22-mode)
   * - ``delta_t``
     - ``delta_t``
     - Sample interval
   * - ``approximant``
     - ``approximant``
     - Waveform model name
   * - ``distance``
     - ``distance``
     - Luminosity distance [Mpc]
   * - ``inclination``
     - ``inclination``
     - Inclination angle [rad]

Returns ``{'type': 'polarizations', 'hp': TimeSeries, 'hc': TimeSeries}``.

You can also register custom waveform generators via
:py:func:`pycwb.modules.injection.wf_generator.generate_injection`.

Injection from Strain Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For injecting arbitrary signals from pre-computed strain files, use
:py:func:`pycwb.modules.injection.inj_generators.get_strain_from_file`, which
supports HDF5, ``.npy``, and text formats with automatic resampling via
polyphase filtering.


Sky Distributions
-----------------

Control where injections appear on the sky with the ``sky_distribution`` block:

.. code-block:: yaml

   injection:
     sky_distribution:
       type: UniformAllSky       # or Patch, Fixed, Custom
       coordsys: icrs            # Coordinate system (default: icrs)

Uniform All Sky
~~~~~~~~~~~~~~~

.. code-block:: yaml

   sky_distribution:
     type: UniformAllSky

Samples right ascension uniformly in :math:`[0, 2\pi)` and declination
uniformly in :math:`\sin\delta` (isotropic on the sphere).

Patch (Circular Cap)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   sky_distribution:
     type: Patch
     patch:
       center:
         phi: 45.0       # RA in degrees (or radians if unit: rad)
         theta: 30.0     # Dec in degrees (or radians if unit: rad)
       radius: 5.0        # Cap radius in degrees (or radians)
       unit: deg

Samples uniformly within a circular cap of the given radius around
``(phi, theta)``.

Fixed
~~~~~

.. code-block:: yaml

   sky_distribution:
     type: Fixed
     fixed:
       phi: 45.0
       theta: 30.0
       unit: deg

All injections share the same sky position.

Custom (HEALPix Map)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   sky_distribution:
     type: Custom
     custom:
       map_path: /path/to/skymap.fits

Samples from a HEALPix probability map using ``healpy``. Requires the
``healpy`` package.


Time Scheduling
---------------

Injections can be placed in time using three strategies, configured via
``time_distribution``:

Explicit GPS Times
~~~~~~~~~~~~~~~~~~

If no ``time_distribution`` is specified, each injection must have a
``gps_time`` field. Injections are assigned to the first job segment interval
that contains their GPS time.

Fixed Rate
~~~~~~~~~~

.. code-block:: yaml

   time_distribution:
     type: rate
     rate: 0.01          # Injections per second
     jitter: 0.0         # Random jitter in seconds

Places injections at fixed time intervals with optional random jitter.

Poisson
~~~~~~~

.. code-block:: yaml

   time_distribution:
     type: poisson
     mean_interval: 1000.0    # Mean time between injections [s]
     max_trail: 50            # Max injections per trial (job segment)

Samples inter-arrival times from an exponential distribution with the given
mean interval, producing Poisson-distributed injection times. The ``max_trail``
parameter limits the number of injections per trial to avoid excessively long
job segments.

Job and Trial Indexing
~~~~~~~~~~~~~~~~~~~~~~

Each injection receives a ``sim_idx`` (unique across all injections) and a
``trial_idx`` (groups injections that share a job segment). When
``parallel_injection_trail`` is enabled, job segments are flattened so each
trial runs as a separate job with its own ``job_id``.


Related Config Parameters
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``injection``
     - ``{}``
     - Top-level injection config object
   * - ``iwindow`` / ``gap``
     - 5.0
     - Injection time window half-width :math:`[T_{inj} - iwindow/2, T_{inj} + iwindow/2]` [s]
   * - ``analyze_injection_only``
     - false
     - Only analyze time around injected signals
   * - ``injection_padding``
     - 1.0
     - Padding around injection window [s]
   * - ``simulation``
     - null
     - Simulation type string
   * - ``nfactor``
     - 0
     - Number of simulation scaling factors
   * - ``factors``
     - ``[]``
     - Array of simulation scaling factors
   * - ``parallel_injection_trail``
     - false
     - Flatten job segments by trial for parallel processing


Validation Checks
-----------------

After setting up injections, verify:

- **GPS times fall inside job segments**: each injection's ``gps_time`` (or
  scheduled time) must be within a segment's analysis window. Missing
  injections usually mean the time range or DQ segments are wrong.
- **Recovered sim_idx matches injection table**: after running, match
  ``catalog.parquet`` against ``simulations.parquet`` with
  ``pycwb match-simulations``. Unmatched injections indicate recovery failure.
- **Sky distribution matches requested mask**: plot the RA/Dec of injected
  signals to verify they follow the requested distribution (UniformAllSky,
  Patch, Fixed, or Custom).
- **Waveform amplitudes scale correctly**: for ``hrss_scaling``, verify that
  the injected :math:`h_{rss}` matches the target value (± a few percent after
  resampling).


----

**See also:** :doc:`job_control` · :doc:`targeted_search` · :doc:`analysis_recipes`

**Next:** :doc:`targeted_search` — restricting the sky region for follow-up
