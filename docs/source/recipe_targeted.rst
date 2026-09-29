.. _recipe_targeted:

Targeted External-Trigger Search
================================

**Goal:** Search a specific sky region around an external trigger (GRB, neutrino).

**Inputs:**

- External trigger RA, Dec, and error radius
- Same data and DQ as all-sky search

**Key Config (add to ``user_parameters.yaml``):**

.. code-block:: yaml

   sky_mask:
     type: Patch
     coordsys: icrs
     patch:
       center:
         ra: "197.5 deg"
         dec: "-23.4 deg"
       radius: "5 deg"

   healpix: 8           # Higher resolution for smaller patch

**Commands:** Same as all-sky search; ``sky_mask`` restricts the likelihood scan.

**Expected Outputs:** Same format, but triggers clustered around the target.

**Validation Checks:**

- All trigger sky positions within patch radius
- Fewer total triggers than all-sky (restricted region)
- Higher healpix gives finer sky localization

**Common Failure Modes:**

- Patch center in wrong coordinate system (check ``coordsys``)
- Radius too small for trigger localization uncertainty
- ``healpix`` too low for small patch (use ≥ 7)

See :doc:`analysis_recipes` for the other analysis tasks.
