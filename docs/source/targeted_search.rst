.. _targeted_search:

Targeted Search
===============

.. rubric:: Pipeline: :doc:`data <pipeline_lifecycle>` → :doc:`segments <job_control>` → :doc:`conditioning <pipeline_lifecycle>` → :doc:`WDM <pipeline_lifecycle>` → :doc:`pixels <clustering_algorithm>` → :doc:`clusters <clustering_algorithm>` → **[sky mask]** ← you are here → :doc:`likelihood <likelihood_guide>` → :doc:`events <pipeline_lifecycle>` → :doc:`bkg <postproduction_background>` → :doc:`ranking <postproduction_xgboost>` → :doc:`eff <postproduction_efficiency>`

This guide explains how to configure pycWB for targeted (pointed) searches,
including sky masks, coordinate systems, and HEALPix resolution settings.

.. contents:: Table of Contents
   :depth: 2
   :local:


Why this matters
----------------

Most users run all-sky searches. Targeted searches are for follow-up of
external triggers (GRBs, neutrino events, supernovae). If you're not
following up a specific event, you likely don't need to change these settings.


Overview
--------

By default, pycWB scans the entire sky for gravitational-wave bursts. For
targeted searches—such as follow-up of external triggers (GRBs, neutrino
events, supernovae) or known sources—you can restrict the likelihood sky scan
to a specific region using a **sky mask**. This reduces computational cost and
improves sensitivity by avoiding spurious sky locations.


Sky Mask Configuration
----------------------

Sky masks are configured via the ``sky_mask`` block in
``user_parameters.yaml``. The mask uses the same distribution types as
injection sky distributions:

.. code-block:: yaml

   sky_mask:
     type: Patch            # UniformAllSky, Patch, Fixed, or Custom
     coordsys: icrs
     patch:
       center:
         ra: "45 deg"
         dec: "30 deg"
       radius: "5 deg"

When ``sky_mask`` is not specified, the full sky is scanned (equivalent to
``type: UniformAllSky``).


Mask Types
----------

Uniform All Sky (Default)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   sky_mask:
     type: UniformAllSky

No restriction—all HEALPix sky directions are evaluated. This is the default
behavior when no ``sky_mask`` is configured.


Patch (Circular Region)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   sky_mask:
     type: Patch
     coordsys: icrs
     patch:
       center:
         ra: "197.5 deg"
         dec: "-23.4 deg"
       radius: "3 deg"

Restricts the likelihood sky scan to a circular cap around the declared
right ascension and declination.
The angular distance between a sky pixel center and the target is computed as:

.. math::

   d = \arccos\left(\sin\delta_1 \sin\delta_2 +
   \cos\delta_1 \cos\delta_2 \cos(\alpha_2 - \alpha_1)\right)

Pixels with :math:`d \leq` ``radius`` are included in the scan.

When using the **HEALPix path** (``healpy`` available), sky mask pixels are
selected via ``healpy.query_disc()``, which provides fast and accurate
pixel-in-region queries.

Fixed (Single Direction)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   sky_mask:
     type: Fixed
     coordsys: icrs
     coordinates:
       ra: "45 deg"
       dec: "30 deg"

Scans only the single HEALPix pixel nearest to the specified sky direction.

Custom (HEALPix Probability Map)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   sky_mask:
     type: Custom
     coordsys: icrs
     custom:
       healpix_map: /path/to/skymap.fits
       ordering: ring
       threshold: 0.5        # Keep pixels above this probability threshold

Uses a HEALPix probability map (e.g., from an external localization) and keeps
only pixels above a probability threshold. Requires the ``healpy`` package.

This is particularly useful for following up LIGO-Virgo-KAGRA compact binary
coalescence (CBC) events, where the sky localization is provided as a HEALPix
FITS file.


Coordinate Systems
------------------

User inputs support three explicit frames: ``icrs`` uses ``ra``/``dec``,
``geo`` uses ``longitude``/``latitude``, and ``cwb`` uses
``phi_geo``/``theta_cwb``. The cWB ``theta_cwb`` angle is a co-latitude in the
range 0 to 180 degrees, not a latitude. Every scalar YAML angle includes its
own unit.

The exact ranges, GMST sign, time convention, and conversions are defined in
:ref:`coordinate_systems_angles`. Do not reproduce those definitions from a
recipe or infer them from the legacy names ``phi`` and ``theta``.


HEALPix Resolution
------------------

The sky resolution is controlled by the ``healpix`` parameter:

.. code-block:: yaml

   healpix: 7    # Sky map resolution (n_pixels = 12 × 4^healpix)

.. list-table:: HEALPix Orders
   :header-rows: 1
   :widths: 15 20 30 35

   * - ``healpix``
     - :math:`N_{pix}`
     - Resolution [deg]
     - Typical Use
   * - 4
     - 3,072
     - ~3.7°
     - Coarse scan, fast
   * - 5
     - 12,288
     - ~1.8°
     - Low-resolution search
   * - 6
     - 49,152
     - ~0.9°
     - Medium resolution
   * - 7
     - 196,608
     - ~0.45°
     - Default, standard search
   * - 8
     - 786,432
     - ~0.22°
     - High-resolution targeted

Higher ``healpix`` values increase angular resolution but also increase
computation time linearly with :math:`N_{pix}`.

The parameter ``MIN_SKYRES_HEALPIX`` (default: 4) sets the minimum HEALPix
resolution used for sub-network cut operations, allowing a coarser sky grid
for intermediate selection steps.

``MIN_SKYRES_ANGLE`` (default: 3°) provides an alternative angle-based minimum
resolution for sub-network cuts.

The ``nSky`` parameter controls how many sky-map probability pixels are written
to ASCII output files.


Targeted Search Workflow
------------------------

A typical targeted search setup:

1. **Define the sky region** in ``sky_mask`` using a ``Patch`` or ``Custom``
   type.
2. **Optionally restrict injections** to the same sky region using the
   ``sky_distribution`` block in the injection config (see
   :ref:`injection_infrastructure`).
3. **Set ``healpix`` resolution** appropriate for the region size—a smaller
   patch benefits from higher resolution at the same computational cost.
4. **Verify the mask** by checking the sky-map output or using the
   ``pycwb progress`` command to inspect trigger sky locations.

For external trigger follow-up (e.g., GRB or neutrino events), use ``Patch``
with the event's RA/Dec and an appropriate error radius. For CBC event
follow-up, use ``Custom`` with the published HEALPix localization.


Related Config Parameters
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``sky_mask``
     - *(none)*
     - Sky mask config (same types as injection ``sky_distribution``)
   * - ``healpix``
     - 7
     - HEALPix order for sky map
   * - ``MIN_SKYRES_HEALPIX``
     - 4
     - Minimum HEALPix resolution for sub-network cuts
   * - ``MIN_SKYRES_ANGLE``
     - 3
     - Minimum angular resolution for sub-network cuts [°]
   * - ``nSky``
     - *(computed)*
     - Number of sky probability pixels in ASCII output
   * - ``EFEC``
     - true
     - Use Earth-fixed/celestial coordinate conversion
   * - ``skyMaskFile``
     - ``""``
     - Legacy sky mask file path
   * - ``skyMaskCCFile``
     - ``""``
     - Legacy sky mask CC file path


----

**See also:** :doc:`injection_infrastructure` ·
:doc:`coordinate_systems` · :doc:`likelihood_guide` · :doc:`analysis_recipes`

**Next:** :doc:`clustering_algorithm` — how pixels become candidate events
