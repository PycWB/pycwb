.. _recipe_targeted:

Targeted External-Trigger Search
================================

**Use this route for:** an external trigger with a defined time window and a
sky location, uncertainty region or mask map. Supply the same frame/DQ inputs
as an all-sky analysis.

1. Follow :doc:`targeted_search` to select a fixed direction, circular patch or
   custom HEALPix mask and specify its coordinate system and units.
2. Check that the chosen numerical sky grid resolves the input region. Verify
   the selected pixels and the interpretation of an external map threshold.
3. Run with the configuration and execution procedures in :doc:`standard_analysis`.
4. Use matching search settings, sky selection and event cuts for the background
   and sensitivity study in :doc:`postproduction_study`.

**Completion check:** the mask is interpreted in the intended frame at the event
time, recovered positions are inspected against it, and the background describes
the same search. Measure the recovery and ranking changes; a smaller mask or a
finer grid does not guarantee fewer triggers or better localization.

**Worked example:** :doc:`tutorial_sky_masks`.

See :doc:`analysis_recipes` for the other analysis tasks.
