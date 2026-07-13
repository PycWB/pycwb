.. _tutorials:

=============
Learning Path
=============

These lessons build progressively—each one adds a skill you'll need for
production searches. Start at the top and work down, or jump to a specific
lesson.

For copy-paste solutions to specific tasks, see :ref:`analysis_recipes`.
For help choosing what to learn next, see :ref:`decision_guides`.


.. raw:: html

   <div style="font-family: monospace; background: #f5f5f5; border-radius: 8px; padding: 1.2em; margin: 1.5em 0; line-height: 2;">

   <strong>Start Here</strong> &nbsp;→&nbsp; <a href="start_here.html">What is pycWB?</a><br>
   &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
   <strong>Lesson 1</strong> &nbsp;→&nbsp; <a href="tutorial_search.html">Your First Search</a><br>
   &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
   <strong>Lesson 2</strong> &nbsp;→&nbsp; <a href="tutorial_injection.html">Injection Search</a><br>
   &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
   <strong>Lesson 3</strong> &nbsp;→&nbsp; <a href="tutorial_multi_injection.html">Multi-Injection</a><br>
   &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
   <strong>Lesson 4</strong> &nbsp;→&nbsp; <a href="tutorial_customized_wf_gen.html">Custom Waveforms</a><br>
   &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
   <strong>Lesson 5</strong> &nbsp;→&nbsp; <a href="tutorial_batch_inj.html">Batch Production</a><br>
   &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
   <strong>Next</strong> &nbsp;→&nbsp; <a href="analysis_recipes.html">Analysis Recipes</a> &nbsp;|&nbsp; <a href="core_concepts.html">Core Concepts</a>

   </div>


----

**Lesson 1**  ⭐⭐⭐⭐⭐ Beginner  |  ~15 min

:doc:`tutorial_search`
   Run your first burst search. Understand the search function, job setup,
   and the config file. This is the foundation for everything else.

----

**Lesson 2**  ⭐⭐⭐⭐☆  |  ~20 min

:doc:`tutorial_injection`
   Inject simulated signals and recover them. Learn how injection parameters
   work, how to configure waveforms, and how to inspect recovered triggers.

----

**Lesson 3**  ⭐⭐⭐☆☆  |  ~25 min

:doc:`tutorial_multi_injection`
   Run multiple injections with different parameters. Understand GPS-time
   scheduling, parameter lists, and the newer scheduled-injection options.

----

**Lesson 4**  ⭐⭐⭐☆☆  |  ~20 min

:doc:`tutorial_customized_wf_gen`
   Use custom waveform generators—burst-waveform, white-noise-burst, and
   Python-based waveform modules. Learn when to use each approach.

----

**Lesson 5**  ⭐⭐⭐⭐☆  |  ~15 min

:doc:`tutorial_batch_inj`
   Submit injection campaigns to HTCondor or SLURM. Convert a local search
   into a production-scale batch run.

----

After the Learning Path
-----------------------

.. list-table::
   :header-rows: 0
   :widths: 30 70

   * - :ref:`analysis_recipes`
     - Copy-paste workflows for specific tasks: all-sky, targeted, efficiency
   * - :ref:`core_concepts`
     - Understand how the algorithms work: clustering, likelihood, job control
   * - :ref:`standard_analysis`
     - Set up production searches with config templates and cluster submission
   * - :ref:`postproduction`
     - Run background estimation, XGBoost ranking, and detection efficiency
