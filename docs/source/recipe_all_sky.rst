.. _recipe_all_sky:

All-Sky Short Burst Search
==========================

**Use this route for:** a search of your detector data without a sky mask.
Prepare frame lists and channel names, DQ intervals, a detector network and a
complete search configuration.

1. Create the run from :doc:`config_repository`, or adapt existing inputs using
   :ref:`analysis_local_frames`. Select the intended search family and band.
2. Use :doc:`job_control` to check segment boundaries, padding, lags and superlags.
   Decide explicitly whether this run analyzes zero lag, background, or both.
3. Follow :doc:`run_on_clusters` for batch setup, resource requests and submission.
4. Inspect completion and the saved catalog/progress described in
   :doc:`understanding_results`. For a background estimate, continue with
   :doc:`recipe_background`.

**Completion check:** requested work is accounted for, selected exposure is
understood, and catalog rows use the intended lag selection. An empty catalog
alone does not mean the run failed, and a nonempty one does not prove correctness.

**Worked example:** :doc:`tutorial_open_data`.

See :doc:`analysis_recipes` for the other analysis tasks.
