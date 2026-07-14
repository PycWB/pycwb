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
