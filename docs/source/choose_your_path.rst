.. _choose_your_path:

Choose Your Path
================

pycWB serves different audiences. Pick the path that matches what you're
trying to do.

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1em; margin: 2em 0;">

   <div style="border: 1px solid #ccc; border-radius: 8px; padding: 1.2em;">
     <h3 style="margin-top:0;">🆕 New User</h3>
     <p>Never used pycWB before. Want to understand what it does and run a first search.</p>
     <a href="start_here.html" style="font-weight:bold;">Start Here →</a>
     <hr>
     <small>Then: <a href="tutorials.html">Tutorials</a> · <a href="glossary.html">Glossary</a></small>
   </div>

   <div style="border: 1px solid #ccc; border-radius: 8px; padding: 1.2em;">
     <h3 style="margin-top:0;">🔍 Search Analyst</h3>
     <p>Running production searches. Need config templates, cluster submission, and postproduction.</p>
     <a href="standard_analysis.html" style="font-weight:bold;">Standard Analysis →</a>
     <hr>
     <small>Then: <a href="analysis_recipes.html">Recipes</a> · <a href="postproduction.html">Postproduction</a></small>
   </div>

   <div style="border: 1px solid #ccc; border-radius: 8px; padding: 1.2em;">
     <h3 style="margin-top:0;">🔬 Algorithm Researcher</h3>
     <p>Understanding the physics. Want to know how clustering, likelihood, and ranking work.</p>
     <a href="core_concepts.html" style="font-weight:bold;">Core Concepts →</a>
     <hr>
     <small>Then: <a href="pipeline_lifecycle.html">Pipeline Lifecycle</a> · <a href="likelihood_guide.html">Likelihood</a></small>
   </div>

   <div style="border: 1px solid #ccc; border-radius: 8px; padding: 1.2em;">
     <h3 style="margin-top:0;">💻 Developer</h3>
     <p>Contributing code. Need architecture, setup, module conventions, and performance guides.</p>
     <a href="dev_architecture.html" style="font-weight:bold;">Architecture →</a>
     <hr>
     <small>Then: <a href="dev_setup.html">Dev Setup</a> · <a href="dev_modules.html">Module Dev</a></small>
   </div>

   <div style="border: 1px solid #ccc; border-radius: 8px; padding: 1.2em;">
     <h3 style="margin-top:0;">🛠️ Maintainer</h3>
     <p>Managing releases, CI, reviewing PRs. Need build/test workflows and contribution guidelines.</p>
     <a href="dev_build_test.html" style="font-weight:bold;">Build & Test →</a>
     <hr>
     <small>Then: <a href="dev_contributing.html">Contributing</a> · <a href="dev_cxx_core.html">C++ Core</a></small>
   </div>

   </div>


Not Sure Where to Start?
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - I want to...
     - Go to...
   * - Run pycWB for the first time ever
     - :ref:`start_here`
   * - Understand what each config parameter does
     - :ref:`pipeline_lifecycle` (param cross-ref table)
   * - Set up a production search on a cluster
     - :ref:`standard_analysis` → :ref:`run_on_clusters`
   * - Run an injection campaign
     - :ref:`analysis_recipes` → Injection Campaign
   * - Understand how the algorithm works
     - :ref:`core_concepts` → :ref:`pipeline_lifecycle`
   * - Debug a failed run
     - :ref:`analysis_recipes` → Debugging a Failed Production
   * - Look up a term
     - :ref:`glossary`
   * - Contribute code
     - :ref:`dev_architecture`
   * - Find a Python function's API
     - :ref:`modules`
