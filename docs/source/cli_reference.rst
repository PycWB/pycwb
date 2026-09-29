.. _cli_reference:

CLI Reference
=============

This reference is generated from the command parser for documentation version
|release|. Check your installation with ``pycwb --version``. Each command also
accepts ``--help``. ``python -m pycwb`` runs the CLI with the active interpreter.

Common workflows
----------------

.. code-block:: bash

   pycwb doctor
   pycwb demo my_first_search --run
   pycwb validate my_first_search/user_parameters.yaml
   pycwb progress --work-dir my_first_search
   pycwb merge --work-dir my_search
   pycwb merge --work-dir my_search --wave
   pycwb xtalk input.bin --output_dir converted

``validate`` checks a rendered YAML configuration without data access.
``batch-setup`` builds job metadata and scheduler files, including catalog
fragments for planned batches; it can download a missing cross-talk catalog.
See :ref:`start_here`, :ref:`troubleshooting`, and :ref:`run_on_clusters` for
complete workflows.

Command options
---------------

.. contents::
   :local:
   :depth: 1

.. include:: _cli_help.rst.inc
