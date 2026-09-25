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
installed LAL version. The registry lists geometry definitions, not a guarantee
of end-to-end data access or scientific validation for every instrument. Use
instrument names present in the registry; there is no ``J1`` entry in this table.

The cWB vectors were checked against ``wat/detector.cc`` from release 6.4.6.9,
commit ``e03cf7f``. This identifies the verification source; it is not part of
the user-facing geometry name. Only H1/L1 have validated cWB definitions here.
A network can explicitly combine those entries with bundled definitions for
other detectors. Selecting an unavailable entry such as ``V1:cwb`` fails.

Configuration loading resolves defaults and aliases into a complete per-detector
mapping, which is recorded in catalog metadata and restored by workers.
Unknown IDs, assignments to inactive detectors, and mismatches such as
``H1: L1:cwb`` are rejected. The former global string
``detector_geometry: cwb_6.4.6.9`` must be replaced by the per-detector mapping.

What changes when geometry changes?
-----------------------------------

Selected vertices and arm vectors feed maximum network delay, sky time delays,
antenna response, injection projection and event timing. Apply the same
selection throughout a comparison; changing geometry changes physical inputs
even if every execution-profile flag remains fixed.

For cWB-selected detectors, event antenna exports retain the release's stored
angle precision convention. See :ref:`native_calculation_conventions` for that
boundary and its distinction from waveform-statistic options.

The cWB definition was added to reproduce release outputs. PycWB's bundled
LAL-derived definition reconstructs vectors from geographic angles, whereas
cWB uses literal rounded vectors. The geometry audit measured:

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Detector
     - Vertex displacement
     - Maximum absolute antenna-response difference in sampled sky
   * - H1
     - 5.367 mm
     - 3.006 × 10\ :sup:`−6`
   * - L1
     - 3.524 mm
     - 2.921 × 10\ :sup:`−6`

The antenna comparison covered 4,099 directions. These are absolute differences,
not relative errors or bounds over the continuous sky. With matched literal
vectors, differences from the cWB antenna oracle were below 1.6 × 10\ :sup:`−15`.
The stored oracle and its provenance are under
``pycwb/types/tests/reference/RELEASE_GEOMETRY.md``.

These checks establish reproduction of cWB, not which set is physically closer
to the surveyed instrument. More decimal digits alone do not establish physical
accuracy. Keep the bundled default to preserve existing PycWB geometry; use
``:cwb`` when comparing against the validated cWB reference. Neither choice is
a performance preset.

For the upstream constants, see
`LALDetectors.h <https://lscsoft.docs.ligo.org/lalsuite/lal/_l_a_l_detectors_8h.html>`_.
Its directly tabulated vectors should not be conflated with PycWB's reconstruction
from the bundled geographic parameters.

Python use and definition location
----------------------------------

All bundled geometry constants and selections are centralized in
``pycwb/constants/detectors.py``. ``DETECTORS`` contains geographic parameters;
``DETECTOR_GEOMETRIES`` registers selectable definitions. There is no separate
release-geometry table module.

.. code-block:: python

   from pycwb.types.detector import Detector

   detector = Detector("H1:cwb")
   assert detector.name == "H1"          # Instrument/data identity
   assert detector.geometry_id == "H1:cwb"  # Selected definition

For settings concerned with computation rather than geometry, see
:ref:`execution_profile`; for angular units, see :ref:`units_conventions`.
