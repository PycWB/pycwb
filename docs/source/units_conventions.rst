.. _units_conventions:

Units and Conventions
=====================

This page defines how units cross pycWB's user, numerical, and serialization
boundaries. Coordinate meanings are defined separately in
:ref:`coordinate_systems_angles`.


General rule
------------

User-facing scalar sky angles are Astropy-compatible quantity strings, for
example ``"120 deg"`` or ``"2.094 rad"``. Each value carries its own unit so a
coordinate name, frame, and unit can be validated together.

Python numerical kernels use radians unless their docstring explicitly says
otherwise. Output objects may retain cWB degree conventions for compatibility;
those boundaries are listed below.


Angle boundaries
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 24 18 28

   * - Boundary
     - Representation
     - Unit
     - Notes
   * - YAML fixed or patch coordinate
     - Quantity string
     - Explicit
     - Semantic key must match ``coordsys``
   * - Existing sky table
     - Numeric columns plus table ``unit``
     - Explicit
     - ``columns`` identifies the coordinate pair
   * - Injection and detector-projection Python paths
     - Floating point
     - rad
     - Canonical numerical representation
   * - ``InjectionParams.ra`` / ``dec``
     - Floating point
     - rad
     - Stored inside simulation trigger metadata
   * - Reconstructed ``Trigger.phi`` / ``theta``
     - Floating point
     - deg
     - Earth-fixed cWB output compatibility
   * - Reconstructed ``Trigger.ra`` / ``dec``
     - Floating point
     - deg
     - Celestial catalog output
   * - Sky-map plotting coordinates
     - Floating point
     - deg
     - Plotting boundary converts explicitly

Use ``astropy.units.Quantity`` at parsing and analysis boundaries rather
than relying on variable names such as ``theta`` or on a distant unit flag.


Time
----

Absolute analysis and event times are GPS seconds. Durations, segment lengths,
lags, and time-of-flight delays are seconds unless a field documents another
unit. Sidereal-time conversion uses ``astropy.time.Time`` or the explicit
cWB-compatible GMST function described in :ref:`coordinate_systems_angles`.


Frequency and sampling
----------------------

Frequencies and bandwidths are in hertz. Sampling rates are samples per second,
and ``delta_t`` is seconds per sample. A frequency-bin spacing ``delta_f`` is
in hertz.


Distance, mass, and strain
--------------------------

Waveform-generator distances are luminosity distances in megaparsecs unless
the generator documents otherwise. Compact-object masses are solar masses.
Gravitational-wave strain is dimensionless; :math:`h_{rss}` has units of
:math:`1/\sqrt{\mathrm{Hz}}` when written as strain times the square root of
time.


Compatibility rule
------------------

Legacy fields remain readable where cWB interoperability requires them, but
new YAML and new public APIs should use semantic names and explicit units.
Compatibility is not a reason to propagate an ambiguous representation into a
new interface.


.. _native_calculation_conventions:

Native calculation choices
--------------------------

Several ``execution_profile`` settings select scientific calculations rather
than only changing memory layout or scheduling. They are explicit YAML choices,
resolved at setup and recorded with the run as described in
:ref:`execution_profile`. Their defaults preserve the previous behavior with
execution environment switches unset.

.. list-table::
   :header-rows: 1
   :widths: 28 10 62

   * - Setting
     - Default
     - Calculation affected
   * - ``native_chirp``
     - false
     - Select the native micropixel chirp estimator in the XGB rho branch.
       It also requires ``xgb_rho_mode`` (including its legacy negative-netRHO
       activation), ``optim: false``, an eligible ``cfg_search`` character in
       ``iecrpblsg``, and ``Search`` equal to ``CBC``, ``BBH`` or ``IMBHB``.
       Setting this flag alone does not enable chirp estimation for every search.
       The native branch resets chirp fields before checking eligibility and
       uses the run's explicit seed for its bootstrap estimator.
   * - ``release_waveform_stats``
     - false
     - Use cWB-release-compatible waveform summaries and sky error regions.
       This changes reported statistics and uncertainty calculations; it is
       independent of detector geometry and of chirp-estimator selection.
   * - ``regression_cap``
     - false
     - Apply release-compatible witness-amplitude capping in regression.
       The explicit ``Search: "--regression OLD"`` option retains uncapped
       behavior even when this flag is true. This can change conditioned data.
   * - ``regression_percentile_stride``
     - 1
     - Positive integer sampling stride for regression percentile statistics.
       One uses every sample; larger values subsample the statistic and can
       change its value while reducing the calculation cost.

For example, these are explicit calculation choices, not a general tuning
recommendation:

.. code-block:: yaml

   execution_profile:
     native_chirp: true
     release_waveform_stats: true
     regression_cap: true
     regression_percentile_stride: 1

Hold these choices constant across performance comparisons. The bounded CPU
recipe enables the first three, so selecting the entire recipe changes both
the numerical calculation and the execution strategy.

Detector geometry and exported angles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The per-detector ``detector_geometry`` selection is a physical-input choice;
see :ref:`detector_support` for definitions and defaults.
Vertex vectors are Earth-centered positions in metres, and arm vectors are
dimensionless. The default LAL-derived entries construct vectors from geographic
parameters; the cWB entries retain literal rounded vectors.

For detectors selected with ``:cwb``, event antenna exports also reproduce the
existing cWB convention of narrowing stored sky/polarization angles to float32
before converting degrees to radians and evaluating the antenna response.
This output-precision convention is distinct from the geometry constants and
is applied per detector. It is not controlled by ``release_waveform_stats``.
