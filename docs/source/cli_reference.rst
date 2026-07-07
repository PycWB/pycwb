.. _cli_reference:

CLI Reference
=============

Every pycWB command with its most common options. Run ``pycwb --help`` or
``pycwb <command> --help`` for the full list.

.. contents:: Table of Contents
   :depth: 1
   :local:


Search & Analysis
-----------------

.. code-block:: bash

   pycwb run <user_parameters.yaml>
       Run a single search locally.

   pycwb flow <user_parameters.yaml>
       Run a search through the Prefect flow wrapper (advanced).

   pycwb online <config.yaml>
       Run a live streaming gravitational-wave search.


Batch Submission
----------------

.. code-block:: bash

   pycwb batch-setup <user_parameters.yaml> [OPTIONS]
       Generate HTCondor or SLURM submission scripts.

       --cluster condor|slurm     Batch system (required)
       --submit                   Submit immediately after generating
       --work-dir PATH            Working directory (default: .)
       --conda-env NAME           Conda environment on worker node
       --n-proc N                 CPUs per job
       --memory SIZE              Memory per job (e.g. 6GB)
       --disk SIZE                Disk per job (e.g. 8GB)
       --job-per-worker N         Jobs bundled per worker script
       --accounting-group GROUP   HTCondor accounting group
       --container-image URI      Container image (condor)
       --walltime TIME            SLURM wall-clock limit (e.g. 72:00:00)
       --slurm-partition NAME     SLURM partition
       --slurm-constraint FEAT    SLURM node constraint
       --n-retries N              App-level retries (SLURM, default 5)
       --list-n-jobs              Print job count, don't submit
       --force-overwrite          Overwrite existing job directories

   pycwb config-setup <workdir> [OPTIONS]
       Create a project from a configuration repository.

       --config-base-path PATH    Path to config repo
       --machine NAME             Machine profile (machine/<name>.yaml)
       --datatype TYPE            Data source (gwosc, igwn-osg, local)
       --cluster condor|slurm     Submission backend
       --submit                   Submit after setup
       --dry-run                  Preview without writing files
       (+ all batch-setup options)

   pycwb clone-dir <source> <target>
       Clone an existing directory layout to a new working directory.


Postproduction
--------------

.. code-block:: bash

   pycwb post-process <workflow.yaml>
       Run a post-production workflow (background, XGBoost, efficiency).

       --diagram-only             Print dependency graph, don't run

   pycwb simulation-summary <user_parameters.yaml> [OPTIONS]
       Build a per-simulation summary Parquet file.

       --work-dir PATH            Working directory
       --output PATH              Output file path

   pycwb match-simulations <catalog> <simulations> [OPTIONS]
       Match trigger catalogs to simulation summaries.

       --how outer|inner|right    Match type (default: right)
       --output PATH              Output file path


Data & Utilities
----------------

.. code-block:: bash

   pycwb gwosc <event_name> [OPTIONS]
       Download GWOSC data and set up an event analysis.

   pycwb gwosc-data <user_parameters.yaml>
       Download GWOSC data for an existing user-parameter file.

   pycwb get-external-modules <user_parameters.yaml>
       Fetch configured external modules.

   pycwb merge <catalog|wave> <input>... <output>
       Merge catalog or wave files.

   pycwb xtalk <input> <output>
       Convert an xtalk file between formats.

   pycwb progress [OPTIONS]
       Summarize run progress from catalog/progress Parquet files.

       --work-dir PATH            Working directory


All Commands (Alphabetical)
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Description
   * - ``pycwb batch-runner``
     - Run one batch job payload
   * - ``pycwb batch-setup``
     - Generate HTCondor or SLURM batch scripts
   * - ``pycwb clone-dir``
     - Clone an existing directory layout
   * - ``pycwb config-setup``
     - Create project from config repository and optionally submit
   * - ``pycwb flow``
     - Run search through Prefect flow wrapper
   * - ``pycwb get-external-modules``
     - Fetch configured external modules
   * - ``pycwb gwosc``
     - Download GWOSC data and set up event analysis
   * - ``pycwb gwosc-data``
     - Download GWOSC data for existing user-parameter file
   * - ``pycwb match-simulations``
     - Match trigger catalogs to simulation summaries
   * - ``pycwb merge``
     - Merge catalog or wave files
   * - ``pycwb online``
     - Run live streaming gravitational-wave search
   * - ``pycwb post-process``
     - Run post-production workflow
   * - ``pycwb progress``
     - Summarize run progress from Parquet files
   * - ``pycwb run``
     - Run a single search locally
   * - ``pycwb simulation-summary``
     - Build per-simulation summary Parquet file
   * - ``pycwb xtalk``
     - Convert xtalk file between formats


See Also
--------

- :ref:`run_on_clusters` — full batch submission guide with YAML config examples
- :ref:`postproduction_workflow` — YAML workflow structure and step reference
- :ref:`analysis_recipes` — ready-to-run workflows using these commands
