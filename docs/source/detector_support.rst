.. _detector_support:

Detector Support and Geometry
=============================

Use ``ifo`` to select instruments, and ``detector_geometry`` to select their
geometry definitions. Geometry selection is independent of channel names,
frame files and data-quality labels: those continue to use ``H1``, ``L1``, etc.

.. code-block:: yaml

   ifo: [L1, H1, V1]
   refIFO: L1
   detector_geometry:
     L1: L1:cwb
     H1: H1:cwb
     V1: V1:lal@pycwb-1

The mapping belongs at the top level of ``user_parameters.yaml``, alongside
``ifo`` and ``refIFO``, not inside ``execution_profile``. Omit the mapping, or an
individual instrument entry, to use its bundled LAL-derived geometry. Explicit
selections are useful when reproducing cWB results or reviewing differences
between runs.

Available definitions
---------------------

.. list-table::
   :header-rows: 1
   :widths: 29 26 45

   * - Registry ID
     - Instruments
     - Definition
   * - ``<IFO>:lal@pycwb-1``
     - H1, L1, V1, I1, G1, K1, E0, E1, E2, E3
     - Default bundled LAL-derived geographic parameters and conversion to
       Earth-centered vectors. ``<IFO>:lal`` is a shorthand for this definition.
   * - ``H1:cwb``, ``L1:cwb``
     - H1, L1
     - Literal cWB Earth-centered vertex and arm vectors. These canonical
       names have no release-version suffix.

Here ``pycwb-1`` identifies the bundled geographic table and conversion in this
repository. It is not a LALSuite release number and does not select the locally
installed LAL version. Use instrument names present in the bundled or custom
registry; there is no ``J1`` entry in this table. Supply strain data and noise
models separately for the detectors in your analysis.

The cWB definitions are available for H1 and L1. A network can combine them
with bundled definitions for other detectors. Selecting an unavailable entry
such as ``V1:cwb`` fails.

Configuration loading resolves defaults and aliases into a complete per-detector
mapping, which is recorded in catalog metadata and restored by workers.
Unknown IDs, assignments to inactive detectors, and mismatches such as
``H1: L1:cwb`` are rejected. The former global string
``detector_geometry: cwb_6.4.6.9`` must be replaced by the per-detector mapping.

.. _custom_detector_definitions:

Custom detectors from a JSON file
---------------------------------

Add ``detector_definitions_file`` at the top level of your analysis YAML to
extend or override the bundled geometry registry. Relative paths are resolved
against the directory containing the YAML file, independently of the working
directory. Absolute paths are also accepted.

For example, add the following to your otherwise complete analysis configuration:

.. code-block:: yaml

   ifo: [H1, L1, X1]
   refIFO: H1
   detector_definitions_file: ./detectors.json
   detector_geometry:
     X1: X1:custom-v1

Create ``detectors.json`` beside the YAML file:

.. code-block:: json

   {
     "schema_version": 1,
     "geometries": {
       "X1:custom-v1": {
         "detector": "X1",
         "parameters": {
           "name": "Example custom detector",
           "lat": 0.7853981633974483,
           "lon": 0.17453292519943295,
           "elevation": 100.0,
           "x": {
             "az": 1.5707963267948966,
             "alt": 0.0,
             "midpoint": 2000.0
           },
           "y": {
             "az": 0.0,
             "alt": 0.0,
             "midpoint": 2000.0
           }
         }
       }
     }
   }

These are illustrative parameters, not a surveyed detector. Replace them with
your instrument's geometry. Adding geometry does not provide strain data,
channels, frame files, noise spectra or data-quality settings; configure those
for the same instrument name separately.

All fields shown in each definition are required:

* ``lat`` and ``lon`` are geodetic latitude and east-positive longitude in
  **radians**, with ranges [-pi/2, pi/2] and [-pi, pi].
* ``elevation`` is height above the reference Earth ellipsoid in **metres**,
  using the same geographic conversion as the bundled LAL-derived definitions.
* Each arm's ``az`` is azimuth in **radians**, measured from north towards east;
  any finite angle is accepted, with periodic trigonometric interpretation.
* Each arm's ``alt`` is its angle above the local horizon in **radians**, in
  [-pi/2, pi/2]. It is not the site's elevation.
* ``midpoint`` is half the arm length in **metres** and must be positive.
  A value of 2000 means a 4000-metre arm.
* ``name`` is a descriptive label. ``detector`` is the instrument identifier
  used in ``ifo``, channel settings and the geometry ID prefix. Identifiers
  start with a letter and contain letters, digits or underscores.

The JSON format currently accepts geographic definitions only. User-supplied
Earth-centered vectors and response tensors are not accepted. The loader
rejects missing or unknown fields, non-finite numbers, duplicate JSON keys,
mismatched IDs and collinear arms (absolute arm dot product >= 1 - 1e-12).
Non-orthogonal interferometers are supported. Unknown geometry selections fail
before derived quantities such as maximum network delay are computed.

Extending and overriding definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A new ID, such as ``H1:survey-v2``, adds an alternative geometry. Set its
``detector`` to ``H1`` and select it with ``H1: H1:survey-v2`` in YAML.
A completely new instrument such as ``X1`` also requires an explicit selection,
as shown above; the loader does not choose arbitrarily between custom entries.

An exact existing **canonical** ID replaces that whole entry for this
configuration. For example, a JSON entry named ``H1:lal@pycwb-1`` with
``detector: H1`` replaces H1's default geometry, even when H1 is omitted from
``detector_geometry``. To replace the cWB entry, use ``H1:cwb`` and select it
in YAML. Such a replacement is a custom geographic definition: it does not
retain the bundled literal vectors or cWB-specific event-angle export behavior.
Use a distinct ID when retaining both alternatives is useful.

Definitions cannot use selector aliases such as ``H1:lal``; use the canonical
ID ``H1:lal@pycwb-1`` instead. Replacements must provide all parameters; there
is no partial field merging. Unmentioned definitions and ordinary bundled
defaults remain available. Loading a configuration does not modify another
configuration or the global bundled registry.

Workers, provenance and Python use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The loaded configuration stores the effective registry in ``detector_registry``.
Catalog metadata saves this snapshot together with the selected geometry IDs
and ``detector_definitions_provenance``: the resolved source path, SHA-256 hash
of the original JSON bytes, format version and replaced bundled IDs.
Workers restoring configuration metadata use the snapshot and do not reopen
the JSON file. Editing or deleting the original file therefore does not change
a saved run; loading the YAML again reads the new file contents. Older metadata
without custom definitions continues to use bundled geometry.

Config constructs one ``Detector`` instance per active instrument when loading
YAML or restoring metadata. Analysis stages reuse these instances for network
delays, sky responses, injections and event output. Geometry selections and
registry lookup are confined to configuration and detector construction.

Use the initialized instances from Python:

.. code-block:: python

   from pycwb.types.detector import Detector, DetectorNetwork

   custom = config.get_detector("X1")
   detectors = config.detectors  # tuple in config.ifo order
   subset = config.get_detectors(["X1", "H1"])  # preserves requested order
   assert subset[0] is custom
   network = DetectorNetwork(detectors=detectors)

Treat the shared detector instances as read-only. Reload the configuration to
change its geometry; changing ``detector_geometry`` or ``detector_registry``
in place does not rebuild existing instances. Requesting an inactive detector
through ``get_detector`` or ``get_detectors`` raises an error.

``max_delay``, ``compute_sky_delay_and_patterns`` and ``project_to_detector``
now take initialized detector instances rather than names or geometry-selection
arguments. For example:

.. code-block:: python

   from pycwb.utils.network import max_delay

   delay = max_delay(config.detectors)
   # Standalone use: construct the desired geometry explicitly.
   delay = max_delay([Detector("H1:cwb"), Detector("L1:cwb")])

Use ``config.to_dict()`` for serializable configuration metadata. Both catalog
formats use it to exclude runtime instances and save the definitions needed to
reconstruct them. ``Config.load_from_dict(...)`` creates fresh instances on the
worker; subsequent calls reuse those worker-local objects.

``Detector("X1")`` alone has no access to a configuration's custom registry.
For standalone geometry work without a full analysis configuration, load the
same JSON explicitly:

.. code-block:: python

   from pycwb.config.detector_definitions import load_detector_definitions

   registry, provenance = load_detector_definitions(
       "./detectors.json", "./user_parameters.yaml"
   )
   custom = Detector("X1:custom-v1", geometry_registry=registry)

What changes when geometry changes?
-----------------------------------

Selected vertices and arm vectors feed maximum network delay, sky time delays,
antenna response, injection projection and event timing. Apply the same
selection throughout a comparison; changing geometry changes physical inputs
even if every execution-profile flag remains fixed.

For cWB-selected detectors, event antenna exports retain the release's stored
angle precision convention. See :ref:`native_calculation_conventions` for that
boundary and its distinction from waveform-statistic options.

The bundled LAL-derived definition reconstructs vectors from geographic
angles; the cWB definition uses literal rounded vectors. Use ``:cwb`` to match
the cWB geometry when comparing results. The numerical comparison is described
in :ref:`detector_geometry_reference`.

For the upstream constants, see
`LALDetectors.h <https://lscsoft.docs.ligo.org/lalsuite/lal/_l_a_l_detectors_8h.html>`_.
Its directly tabulated vectors should not be conflated with PycWB's reconstruction
from the bundled geographic parameters.

Python use and definition location
----------------------------------

All bundled geometry constants and selections are centralized in
``pycwb/constants/detectors.py``. ``DETECTORS`` contains geographic parameters;
``DETECTOR_GEOMETRIES`` registers selectable definitions.

.. code-block:: python

   from pycwb.types.detector import Detector

   detector = Detector("H1:cwb")
   assert detector.name == "H1"          # Instrument/data identity
   assert detector.geometry_id == "H1:cwb"  # Selected definition

For settings concerned with computation rather than geometry, see
:ref:`execution_profile`; for angular units, see :ref:`units_conventions`.
