.. _recipe_efficiency:

Efficiency Study
================

**Use this route for:** measuring recovery at a stated significance threshold.
Supply eligible simulation truth, recovered/scored triggers, the ranking model
when used, and a FAR mapping built from the appropriate background exposure.

1. Follow :doc:`postproduction_study` to score independent evaluation simulations
   and generate efficiency products with the maintained workflow.
2. Use :doc:`postproduction_efficiency` to define source eligibility, matching,
   amplitude coordinates, waveform groups and uncertainty.
3. Keep the chosen threshold and population definition with the curves and model.

**Completion check:** retain missed eligible injections in the denominator and
handle duplicate matches explicitly. Report hrss50/hrss90 or fitted crossings
only where the population samples support them. Investigate unexpected recovery
using waveform support, timing, grouping and the selected threshold.

**Worked precursor:** :doc:`tutorial_population` demonstrates recovered and missed
sources before a larger sensitivity study.

See :doc:`analysis_recipes` for the other analysis tasks.
