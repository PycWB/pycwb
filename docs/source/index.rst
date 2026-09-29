.. _welcome-to-pycwb-s-documentation:

PycWB documentation
===================

PycWB is a modular Python implementation of coherent WaveBurst (cWB/cWB-2G)
for gravitational-wave burst searches.

.. raw:: html

   <figure class="docs-hero">
     <a href="pipeline_lifecycle.html#search-lifecycle-animation">
       <video class="docs-hero-video" autoplay muted loop playsinline preload="auto"
              width="1280" height="720" poster="_static/media/pycwb_hero_poster.png"
              aria-label="A simulated H1–L1 burst spelling pycWB is whitened, its coherent pixels are selected, and the sky scan peaks on the H1–L1 delay ring. Opens the full search animation.">
         <source src="_static/media/pycwb_hero.mp4" type="video/mp4">
       </video>
       <img class="docs-hero-still" src="_static/media/pycwb_hero_poster.png" width="1280" height="720"
            alt="Whitened H1 and L1 time-frequency maps of a simulated burst spelling pycWB, with its selected pixels outlined, and the sky map with the H1–L1 delay ring. Opens the full search animation.">
     </a>
     <figcaption>A simulated H1–L1 burst through whitening, coherent pixel selection and
       the sky scan, computed by the search.
       <a href="pipeline_lifecycle.html#search-lifecycle-animation">Watch the full 60-second search lifecycle</a>.</figcaption>
   </figure>

.. _quick-start:

Get started
-----------

.. container:: docs-start

   * :doc:`Install PycWB <install>`

     Install the package and its dependencies.

   * :doc:`Run your first search <start_here>`

     Recover a simulated burst and inspect the results.

.. _choose-your-path:
.. _documentation-map:

Explore the documentation
-------------------------

.. container:: docs-paths

   * :doc:`Tutorials <tutorials>`

     Run prepared experiments and inspect signals, masks, recovery, and reports.

   * :doc:`How-to guides <run_analyses>`

     Adapt searches, cluster execution, and postproduction to your own inputs.

   * :doc:`Concepts and methods <core_concepts>`

     Understand the pipeline, coherent reconstruction, and significance.

   * :doc:`Reference <reference>`

     Look up parameters, commands, data formats, conventions, and Python APIs.

Contributing code? See :doc:`development` for setup, architecture, and testing.

.. _what-is-pycwb:
.. _id1:

How the search works
--------------------

.. image:: _static/diagrams/pipeline_overview.svg
   :alt: PycWB pipeline from detector data to reconstructed events and postproduction
   :width: 100%

Watch the :ref:`60-second search lifecycle animation <search_lifecycle_animation>`
and explore each stage in :doc:`pipeline_lifecycle`.
For the project introduction, see :doc:`about`.

.. _cli-reference:
.. _indices-and-tables:

See :doc:`cli_reference` for commands, or :ref:`reference_indexes` for the
complete indexes and search. For questions, see :doc:`support`; for publication
credits, see :doc:`credit`.

.. toctree::
   :hidden:
   :maxdepth: 6

   getting_started
   tutorials
   run_analyses
   core_concepts
   reference
   development

.. toctree::
   :hidden:
   :caption: Project
   :maxdepth: 6

   credit
   about
