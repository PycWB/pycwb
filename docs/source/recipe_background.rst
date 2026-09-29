.. _recipe_background:

Background-Only Production
==========================

**Use this route for:** estimating a noise background from real detector data.
Supply frame/DQ inputs and a search configuration without synthetic injections.

1. Prepare a BKG configuration with :doc:`config_repository` and define its
   nonzero lags and superlags using :doc:`job_control`.
2. Follow :doc:`run_on_clusters` for execution and completion accounting.
3. Use :doc:`postproduction_trainingset` and :doc:`postproduction_workflow` to
   select background events and their analyzed exposure together. Exclude
   physical zero lag from the FAR sample and keep training intervals separate.
4. Interpret the resulting FAR curve with :doc:`postproduction_background`;
   continue to :doc:`postproduction_study` when applying a trained ranking.

**Completion check:** selected exposure agrees with completed intervals after
vetoes, padding and overlap accounting. The cumulative rate is computed from
the selected sample and exposure; a small sample need not give a smooth curve.
Report finite exposure when no event exceeds a threshold.

**Worked example:** :doc:`tutorial_background`.

See :doc:`analysis_recipes` for the other analysis tasks.
