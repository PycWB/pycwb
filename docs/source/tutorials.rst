.. _tutorials:
.. _learning-path:

Tutorials
=========

Complete :doc:`start_here` before these lessons. The sequence below develops
injection and batch-analysis skills. Times are approximate reading/work-through
estimates; downloads, compilation, and analysis runtime can take longer.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Lesson
     - Prerequisite / estimate
     - What you will learn
   * - :doc:`tutorial_search`
     - First-search demo; intermediate; ~15 min
     - Inspect the search function, job setup, configuration, and pipeline internals.
   * - :doc:`tutorial_injection`
     - First-search demo; ~20 min
     - Inject and recover simulated signals, configure waveforms, and inspect triggers.
   * - :doc:`tutorial_multi_injection`
     - Injection search; ~25 min
     - Use GPS-time scheduling, parameter lists, and scheduled-injection options.
   * - :doc:`tutorial_customized_wf_gen`
     - Injection search; ~20 min
     - Use burst-waveform, white-noise-burst, and Python waveform generators.
   * - :doc:`tutorial_batch_inj`
     - Local injection run; ~15 min
     - Submit injection campaigns to HTCondor or SLURM.

The intermediate :doc:`tutorial_search` walkthrough lives under
:doc:`core_concepts`. The injection lessons are collected here:

.. toctree::
   :maxdepth: 1

   tutorial_injection
   tutorial_multi_injection
   tutorial_customized_wf_gen
   tutorial_batch_inj

.. _after-the-learning-path:

After the tutorials
-------------------

* :doc:`analysis_recipes`: task-specific workflows, including all-sky, targeted,
  injection, background, training, efficiency, and debugging tasks.
* :doc:`standard_analysis`: configuration templates and cluster submission.
* :doc:`postproduction`: background, ranking, and detection efficiency workflows.
* :doc:`core_concepts`: pipeline, clustering, likelihood, and job-control explanations.
