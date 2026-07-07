.. pycWB documentation master file, created by
   sphinx-quickstart on Wed Mar  8 13:16:45 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pycWB's documentation!
===================================

.. raw:: html

   <p>
     <a href="https://pycwb.readthedocs.io/en/latest/">
       <img src="https://readthedocs.org/projects/pycwb/badge/?version=latest" alt="Documentation">
     </a>
     <a href="https://git.ligo.org/yumeng.xu/pycwb/-/pipelines">
       <img src="https://git.ligo.org/yumeng.xu/pycwb/badges/main/pipeline.svg" alt="Build Status">
     </a>
     <a href="https://git.ligo.org/yumeng.xu/pycwb/-/releases">
       <img src="https://git.ligo.org/yumeng.xu/pycwb/-/badges/release.svg" alt="Releases">
     </a>
     <a href="https://badge.fury.io/py/pycWB">
       <img src="https://badge.fury.io/py/pycWB.svg" alt="PyPI version">
     </a>
     <a href="https://git.ligo.org/yumeng.xu/pycwb/-/blob/main/LICENSE">
       <img src="https://img.shields.io/badge/license-GPLv3-blue" alt="License">
     </a>
   </p>


PycWB is a modularized Python package for coherent gravitational-wave burst
searches based on the core algorithms of cWB.


.. toctree::
   :hidden:
   :maxdepth: 5

   credit

.. toctree::
   :hidden:
   :caption: User Guides
   :maxdepth: 2

   start_here
   install
   Learning Path <tutorials>
   analysis_recipes
   decision_guides
   core_concepts
   Production Analysis <standard_analysis>
   postproduction
   schema
   modules_guide
   API Reference <modules>
   cli_reference
   glossary

.. toctree::
   :hidden:
   :caption: Developer Guides
   :maxdepth: 1

   dev_architecture
   dev_setup
   dev_modules
   dev_performance
   dev_build_test
   dev_contributing
   dev_cxx_core


What is pycWB?
--------------

pycWB is a Python package for **coherent gravitational-wave burst searches**.
It takes strain data from LIGO, Virgo, and KAGRA detectors and looks for
short, unmodeled signals—from supernovae, gamma-ray bursts, or unexpected
phenomena—by finding excess coherent power across the detector network.

Unlike template-based searches that look for specific waveforms, pycWB
identifies **any statistically significant coherence** between detectors,
making it sensitive to both known and unknown source types.

.. mermaid::

   flowchart LR
     A[Input data] --> B[Production search]
     B --> C[Triggers]
     C --> D[Postproduction]
     D --> E[Background statistics]
     D --> F[Detection efficiency]
     D --> G[Candidate significance]


Choose Your Path
----------------

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.8em; margin: 1.5em 0;">

   <div style="border: 1px solid #ddd; border-radius: 6px; padding: 1em;">
     <strong>🆕 New to pycWB?</strong><br>
     <a href="start_here.html">Start Here →</a>
   </div>

   <div style="border: 1px solid #ddd; border-radius: 6px; padding: 1em;">
     <strong>🔍 Run a search</strong><br>
     <a href="standard_analysis.html">Standard Analysis →</a>
   </div>

   <div style="border: 1px solid #ddd; border-radius: 6px; padding: 1em;">
     <strong>📋 Solve a task</strong><br>
     <a href="analysis_recipes.html">Analysis Recipes →</a>
   </div>

   <div style="border: 1px solid #ddd; border-radius: 6px; padding: 1em;">
     <strong>🤔 Make a choice</strong><br>
     <a href="decision_guides.html">Decision Guides →</a>
   </div>

   <div style="border: 1px solid #ddd; border-radius: 6px; padding: 1em;">
     <strong>🔬 Understand algorithms</strong><br>
     <a href="core_concepts.html">Core Concepts →</a>
   </div>

   <div style="border: 1px solid #ddd; border-radius: 6px; padding: 1em;">
     <strong>📊 Postproduction</strong><br>
     <a href="postproduction.html">Background, XGBoost, Efficiency →</a>
   </div>

   <div style="border: 1px solid #ddd; border-radius: 6px; padding: 1em;">
     <strong>📖 Look up parameters</strong><br>
     <a href="schema.html">Schema →</a>
   </div>

   <div style="border: 1px solid #ddd; border-radius: 6px; padding: 1em;">
     <strong>💻 Contribute code</strong><br>
     <a href="dev_architecture.html">Developer Guides →</a>
   </div>

   </div>


Quick Start
-----------

.. code-block:: bash

   # Install
   pip install pycwb

   # Copy example
   cp -r examples/injection my_first_search && cd my_first_search

   # Run
   pycwb run user_parameters_injection.yaml

See :ref:`start_here` for a guided first run, or :ref:`installing_pycwb`
for detailed installation options.


Documentation Map
-----------------

.. list-table::
   :header-rows: 0
   :widths: 25 75

   * - :ref:`start_here`
     - What pycWB does, first run in 10 minutes, common mistakes
   * - :ref:`tutorials`
     - Learn by example: injection, multi-injection, batch
   * - :ref:`analysis_recipes`
     - Copy-paste workflows: all-sky, targeted, injection campaign, debugging
   * - :ref:`decision_guides`
     - Flowcharts: which settings, which recipe, which split strategy
   * - :ref:`core_concepts`
     - Algorithms: pipeline lifecycle, job control, clustering, likelihood
   * - :ref:`standard_analysis`
     - Config templates, cluster submission (Condor & SLURM)
   * - :ref:`postproduction`
     - Background estimation, XGBoost ranking, detection efficiency
   * - :ref:`schema`
     - All parameters: defaults, ranges, descriptions, cross-references
   * - :ref:`modules`
     - Auto-generated API reference from docstrings
   * - :ref:`glossary`
     - ~60 key terms: lag, DPF, FAR, rho, supercluster, etc.
   * - Developer Guides
     - Architecture, setup, build/test, modules, performance, contributing


CLI Reference
-------------

Most users only need these three commands:

.. code-block:: bash

   pycwb run         # Run a single search
   pycwb batch-setup # Generate Condor/SLURM submission scripts
   pycwb post-process# Run postproduction workflow

See the :ref:`run_on_clusters` page for full CLI details, or run
``pycwb --help`` for all available commands.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
