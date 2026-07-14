.. _coordinate_systems_angles:

Coordinate Systems
==================

This page is the canonical definition of sky coordinates in pycWB. It applies
to injection positions, sky masks, detector projection, reconstructed event
positions, catalog fields, and sky-map plots.

.. contents:: On this page
   :depth: 2
   :local:


Canonical names
---------------

.. list-table::
   :header-rows: 1
   :widths: 16 22 34 18 10

   * - ``coordsys``
     - Coordinate pair
     - Meaning
     - Range
     - Internal unit
   * - ``icrs``
     - ``ra``, ``dec``
     - Celestial right ascension and declination
     - :math:`0\leq\alpha<2\pi`,
       :math:`-\pi/2\leq\delta\leq\pi/2`
     - rad
   * - ``geo``
     - ``longitude``, ``latitude``
     - Earth-fixed geographic longitude and latitude
     - :math:`-\pi\leq\lambda<\pi`,
       :math:`-\pi/2\leq\beta\leq\pi/2`
     - rad
   * - ``cwb``
     - ``phi_geo``, ``theta_cwb``
     - Earth-fixed cWB longitude and **co-latitude**
     - :math:`0\leq\phi<2\pi`,
       :math:`0\leq\theta\leq\pi`
     - rad

``theta_cwb`` is zero at the north pole, 90 degrees on the equator, and 180
degrees at the south pole. It is not declination or latitude. Generic
``phi``/``theta`` names are retained only in legacy compatibility interfaces
where their meaning is fixed by cWB.


Earth-fixed and celestial conversion
------------------------------------

The geographic latitude and cWB co-latitude describe the same polar direction:

.. math::

   \mathrm{latitude} = \mathrm{dec} = \frac{\pi}{2} - \theta_{\mathrm{cWB}}.

At GPS time :math:`t`, Earth-fixed cWB longitude and celestial right ascension
are related by Greenwich mean sidereal time (GMST):

.. math::

   \mathrm{ra}(t) =
   \operatorname{wrap}_{[0,2\pi)}\left(\phi_{\mathrm{geo}} +
   \operatorname{GMST}(t)\right),

.. math::

   \phi_{\mathrm{geo}}(t) =
   \operatorname{wrap}_{[0,2\pi)}\left(\mathrm{ra} -
   \operatorname{GMST}(t)\right).

The sign is important: Earth-fixed longitude to right ascension **adds** GMST;
the inverse conversion **subtracts** it. A GPS epoch is therefore required to
convert between ``icrs`` and either Earth-fixed frame.

The conversion functions are
:py:func:`pycwb.utils.skymap_coord.convert_phi_to_ra`,
:py:func:`pycwb.utils.skymap_coord.convert_ra_to_phi`,
:py:func:`pycwb.utils.skymap_coord.convert_theta_to_dec`, and
:py:func:`pycwb.utils.skymap_coord.convert_dec_to_theta`.


GMST models
-----------

pycWB exposes two explicit sidereal-time models:

``cwb``
   :py:func:`pycwb.utils.skymap_coord.gmst_cwb` transcribes the polynomial in
   ``cwb-core/skymap.hh``. It is used when numerical parity with serialized cWB
   event coordinates is required.

``astropy``
   :py:func:`pycwb.utils.skymap_coord.gmst_astropy` uses Astropy mean sidereal
   time. It is used by physical detector projection paths.

Use the same model for both directions of a round trip. Do not convert with one
model and invert with the other.


User parameter YAML
-------------------

New YAML uses semantic coordinate names, an explicit ``coordsys``, and a unit
on every scalar angle. For an ICRS fixed position:

.. code-block:: yaml

   sky_distribution:
     type: Fixed
     coordsys: icrs
     coordinates:
       ra: "120 deg"
       dec: "-30 deg"

For a circular ICRS patch:

.. code-block:: yaml

   sky_mask:
     type: Patch
     coordsys: icrs
     patch:
       center:
         ra: "197.5 deg"
         dec: "-23.4 deg"
       radius: "5 deg"

The equivalent Earth-fixed cWB form makes the co-latitude explicit:

.. code-block:: yaml

   sky_distribution:
     type: Fixed
     coordsys: cwb
     coordinates:
       phi_geo: "290 deg"
       theta_cwb: "60 deg"

Astropy-compatible angular units such as ``deg``, ``rad``, ``arcmin``, and
``arcsec`` are accepted. Bare numbers and a detached ``unit`` field are
ambiguous. The old numeric ``phi``/``theta`` form is deprecated and should not
appear in new configurations.


HEALPix maps and tables
-----------------------

For an existing two-column sky table, the unit applies to every table value
and the columns must match the declared frame:

.. code-block:: yaml

   sky_distribution:
     type: existing
     coordsys: icrs
     existing_path: input/sky_positions.txt
     columns: [ra, dec]
     unit: deg

HEALPix maps must declare their ordering when it is not the default RING
ordering. ``theta`` returned by ``healpy.pix2ang`` is a HEALPix
co-latitude; it is converted before being exposed as ``dec`` or ``latitude``.


Reconstructed and injected catalog values
-----------------------------------------

Catalog output preserves cWB compatibility at the reconstructed boundary:

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
     - Unit
   * - ``Trigger.phi``, ``Trigger.theta``
     - Reconstructed Earth-fixed cWB longitude and co-latitude
     - deg
   * - ``Trigger.ra``, ``Trigger.dec``
     - Reconstructed celestial right ascension and declination
     - deg
   * - ``Trigger.injection.ra``, ``Trigger.injection.dec``
     - Injected celestial right ascension and declination
     - rad

Always check the field definition at a serialization boundary; do not infer a
unit from the coordinate name alone. See :ref:`units_conventions` for the
complete boundary rule.


Common mistakes
---------------

* Treating ``theta_cwb`` as latitude or declination.
* Passing degrees to a Python conversion function that accepts radians.
* Omitting GPS time during ICRS/Earth-fixed conversion.
* Adding GMST in both directions, or subtracting it in both directions.
* Combining ``coordsys: icrs`` with ``phi_geo``/``theta_cwb`` keys.
* Assuming an injection's radian catalog fields have the same unit as a
  reconstructed trigger's degree fields.
