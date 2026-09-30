.. _tutorial_custom_postproduction:

Build a Reusable Postproduction Report
======================================

**Question:** How can you assemble your own analysis from reusable actions?
Use the three completed runs from :doc:`tutorial_comparisons`.

Start from a complete workflow
------------------------------

.. literalinclude:: ../../examples/tutorials/compare.yaml
   :language: yaml

``vars`` describes your dataset. ``inputs`` declares the data each action
consumes; ``args`` selects its behavior. ``outputs`` names files, and
``@step.key`` references a returned result. The ``tmp://`` paths use the
workflow's temporary directory. The report embeds the HTML plots returned
by the plotting action.

Make a useful change
--------------------

Copy the workflow, add a fourth catalog entry, and change the title to describe
your comparison. To study detector networks, use the same injected source with
``all_sky``, ``hlv`` and ``custom_network``. To study source amplitude, run
separate configurations with the same source except for amplitude and label
those runs explicitly. Keep units and truth checks in the plotting action.

.. code-block:: bash

   cp examples/tutorials/compare.yaml tutorial-work/my_report.yaml
   # Edit vars.runs and the report title in tutorial-work/my_report.yaml.
   pycwb post-process tutorial-work/my_report.yaml --diagram-only
   pycwb post-process tutorial-work/my_report.yaml --no-diagram

Read ``tutorial-work/public/comparison/index.html`` and check that every input
run appears. Keep its copied plot assets together when sharing the report.

Extend with your own action
---------------------------

An action receives named inputs and returns a dictionary of products. A custom
plot action should return a ``plots`` list whose entries include a title and
an ``html_file`` path; the generic report can then consume that list unchanged.
Use :doc:`postproduction_workflow` for action resolution and
:doc:`dev_postproduction` for the supported extension contract. The short
``postprocess.*`` action names above resolve below ``pycwb.modules``.

**Result to keep:** a small workflow with explicit data dependencies and a
portable report, so the same analysis can be repeated on another set of catalogs.
