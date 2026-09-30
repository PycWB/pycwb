.. _tutorials:
.. _learning-path:

Tutorials
=========

Start with :doc:`start_here`, then learn by running a prepared experiment.
Each lesson supplies an example, a change to try, and an output to inspect.
The synthetic lessons share generated inputs; the public-data lesson provides
a download command. To work with your own dataset or production configuration,
go directly to :doc:`run_analyses`.

For the basics of loading catalogs, job segments, progress and simulation
matches in Python, start with :doc:`understanding_results`.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Learning path
     - Research questions
     - Start with
   * - :doc:`tutorial_signals`
     - What did I recover, and how do data quality, sky restrictions and detector geometry affect it?
     - The first-search demo and the shared synthetic inputs.
   * - :doc:`tutorial_significance`
     - How do recovered signals change between runs, and how is a small background rate calculated?
     - The prepared sky-mask runs, or the public-data download.
   * - :doc:`tutorial_extensions`
     - How do a custom waveform, a changed frequency band, a report, or a resource budget affect the example?
     - A completed synthetic run and its saved products.

For a short first sequence, prepare the inputs in :doc:`tutorial_signals`,
inspect an event, compare sky masks, then measure recovery in the small
injection population. Then compare those runs or work through the small
background exercise.

.. toctree::
   :hidden:
   :maxdepth: 1

   tutorial_signals
   tutorial_significance
   tutorial_extensions

.. _after-the-learning-path:

After the tutorials
-------------------

* :doc:`run_analyses`: procedures for your own searches and production workflows.
* :doc:`analysis_recipes`: choose the guides needed for a research task.
* :doc:`core_concepts`: pipeline, clustering, likelihood, and job-control explanations.
* :doc:`reference`: exact parameters, commands and data formats.
