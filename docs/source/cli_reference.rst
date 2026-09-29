.. _cli_reference:

CLI Reference
=============

Use ``pycwb --help`` to list commands and ``pycwb COMMAND --help`` for a
command's options. ``python -m pycwb`` runs the CLI with the active interpreter.

Common workflows
----------------

.. code-block:: bash

   pycwb validate user_parameters.yaml
   pycwb run user_parameters.yaml
   pycwb progress --work-dir my_search
   pycwb merge --work-dir my_search
   pycwb merge --work-dir my_search --wave
   pycwb xtalk input.bin --output_dir converted

``validate`` checks a rendered YAML configuration without data access.
``batch-setup`` builds job metadata and scheduler files, including catalog
fragments for planned batches; it can download a missing cross-talk catalog.
See :ref:`start_here`, :ref:`troubleshooting`, and :ref:`run_on_clusters` for
complete workflows.

Search, batch, and postproduction
---------------------------------

.. code-block:: bash

   pycwb run         # Run a single search
   pycwb batch-setup # Generate Condor/SLURM submission scripts
   pycwb post-process # Run postproduction workflow

For cluster submission details, see :doc:`run_on_clusters`.


Command options
---------------

.. contents::
   :local:
   :depth: 1

.. include:: _cli_help.rst.inc
