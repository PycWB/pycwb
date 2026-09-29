.. _recipe_debugging:

Debugging a Failed Production
=============================

**Use this route for:** a failed, incomplete or unexpected run. Preserve the
configuration, logs, catalog manifests and progress before changing inputs.

1. Start with :doc:`troubleshooting` to distinguish import/data failures,
   incomplete work and a completed run with no candidates.
2. Check frames, channels and requested GPS coverage with :ref:`analysis_local_frames`;
   inspect DQ windows, padding and lag selection with :doc:`job_control`.
3. For memory or worker failures, inspect allocation and execution diagnostics
   using :doc:`run_on_clusters` and :doc:`workflow_execution`.
4. For unexpected background or recovery, inspect the selected exposure, zero-lag
   separation, injection timing and truth matching with :doc:`postproduction_study`.
5. Identify unfinished job/trial/lag tuples before retrying. For the scalable
   executor, follow :ref:`execution_resume`; other modes have their own recovery
   constraints in :doc:`troubleshooting`.

**Completion check:** identify the failed stage and reproduce the problem or
account for the unexpected result. Do not use catalog size or the shape of a
ranking distribution alone as a pass/fail test.

**Worked precursor:** :doc:`tutorial_resources` demonstrates completion-aware
restart on a small, unchanged run.

See :doc:`analysis_recipes` for the other analysis tasks.
