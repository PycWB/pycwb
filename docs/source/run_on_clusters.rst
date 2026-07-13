.. _run_on_clusters:

Run on Clusters
===============

This guide explains how to configure and submit pycWB jobs to **HTCondor (on
LDG)** and **SLURM** batch systems using the ``pycwb batch-setup`` and
``pycwb config-setup`` commands.

.. contents:: Table of Contents
   :depth: 2
   :local:


Configuration Sources
---------------------

Job submission settings can be provided via three mechanisms, in order of
priority (highest first):

1. **CLI flags** — passed directly to ``pycwb batch-setup`` or ``pycwb config-setup``
2. ``batch_setup()`` keyword arguments — when calling the Python API directly
3. ``user_parameters.yaml`` — configuration file in your working directory
4. **Schema defaults** — built-in fallback values


Configuration File (``user_parameters.yaml``)
---------------------------------------------

Add a ``job_submission`` section to your ``user_parameters.yaml`` to set
persistent defaults for your project.

Common Settings (both Condor and SLURM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Batch system: "condor" or "slurm"
   cluster: slurm

   # Conda environment to activate on the worker node
   conda_env: igwn-py310

   # Additional shell commands to run before pycwb (e.g. module loads)
   additional_init: ""

   # Number of CPU cores per job
   nproc: 4

   # Memory per job
   job_memory: "6GB"

   # Disk per job
   job_disk: "8GB"

   # Number of analysis jobs bundled into a single worker script
   job_per_worker: 10


HTCondor (on LDG)
=================

HTCondor is the batch system used on the LIGO Data Grid (LDG). This section
covers Condor-specific settings, job structure, and submission.

Condor-Specific Settings
------------------------

.. code-block:: yaml

   # HTCondor accounting group (required for condor)
   accounting_group: ligo.dev.o4.burst.ebbh.cwb

   # Container image URI (Apptainer/Singularity or Docker)
   # Setting this automatically enables file transfer
   container_image: ""

   # Explicitly enable HTCondor file transfer (auto-enabled when container_image is set)
   should_transfer_files: false

.. important::

   Condor **requires** ``accounting_group``. If not set in YAML or via
   ``--accounting-group``, the condor setup will raise an error.


Generating and Submitting Condor Jobs
-------------------------------------

Using ``pycwb batch-setup`` (already-initialised working directory):

.. code-block:: bash

   # Generate and submit
   pycwb batch-setup user_parameters.yaml \
       --cluster condor \
       --work-dir /path/to/workdir \
       --accounting-group ligo.dev.o4.burst.ebbh.cwb \
       --conda-env igwn-py310 \
       --n-proc 2 \
       --memory 6GB \
       --submit

   # With container (no shared filesystem)
   pycwb batch-setup user_parameters.yaml \
       --cluster condor \
       --work-dir /path/to/workdir \
       --accounting-group ligo.dev.o4.burst.ebbh.cwb \
       --image docker://igwn/software:latest \
       --submit

Using ``pycwb config-setup`` (starting from config repository):

.. code-block:: bash

   pycwb config-setup O4_K02_C00_BurstLF_LH_BKG_standard \
       --config-base-path ./config \
       --machine cit \
       --cluster condor --submit \
       --accounting-group ligo.dev.o4.burst.cwb \
       --container-image /cvmfs/container.gwdc.org/pycwb_latest.sif


Condor Job Structure
--------------------

When ``--cluster condor`` is used, the following files are created under
``<workdir>/condor/``:

.. list-table::
   :header-rows: 1

   * - File
     - Purpose
   * - ``run.sh``
     - Worker shell script
   * - ``merge.sh``
     - Merge shell script
   * - ``dag/job.dag``
     - DAGMan file — wires up workers → merge dependency

Submission Flow:

.. code-block:: bash

   condor_submit_dag dag/job.dag

DAGMan runs all ``JOB`` nodes first, then the ``CHILD merge`` node after all
workers succeed.


SLURM
=====

SLURM is the batch system used on many HPC clusters (e.g. Picasso, CIT).
This section covers SLURM-specific settings, job structure, and submission.

SLURM-Specific Settings
-----------------------

.. code-block:: yaml

   # Wall-clock time limit per array job (HH:MM:SS or D-HH:MM:SS)
   # Default: "72:00:00"
   job_walltime: "72:00:00"

.. note::

   ``slurm_constraint`` and ``slurm_partition`` are **not** stored in the
   config file. They must be passed as CLI flags or Python API arguments
   each time, since they are site-specific.


Generating and Submitting SLURM Jobs
------------------------------------

Using ``pycwb batch-setup``:

.. code-block:: bash

   # Generate scripts only
   pycwb batch-setup user_parameters.yaml \
       --cluster slurm \
       --work-dir /path/to/workdir \
       --conda-env igwn-py310 \
       --n-proc 4 \
       --memory 8GB \
       --walltime 48:00:00 \
       --slurm-partition burst \
       --slurm-constraint skylake \
       --n-retries 3

   # Generate and submit
   pycwb batch-setup user_parameters.yaml \
       --cluster slurm \
       --work-dir /path/to/workdir \
       --submit

Using ``pycwb config-setup``:

.. code-block:: bash

   pycwb config-setup O4_K02_C00_BurstLF_LH_BKG_standard \
       --config-base-path ./config \
       --machine picasso-small \
       --cluster slurm --submit


SLURM Job Structure
-------------------

When ``--cluster slurm`` is used, the following files are created under
``<workdir>/slurm/``:

.. list-table::
   :header-rows: 1

   * - File
     - Purpose
   * - ``run.sh``
     - SLURM array job script — runs one worker per array task
   * - ``merge.sh``
     - Single-node merge job — runs ``pycwb merge-catalog`` after all workers finish

Submission Flow:

.. code-block:: bash

   sbatch --array=0-<N> run.sh          # returns <array_job_id>
   sbatch --dependency=afterok:<array_job_id> merge.sh

The merge job only starts if **all** array tasks exit with code 0. If the
array job is cancelled or the dependency is invalid, the merge job is
cancelled automatically (``--kill-on-invalid-dep=yes``).

Automatic Retry
~~~~~~~~~~~~~~~

Each worker script retries the pycwb command up to ``n_retries`` times
(default: 5) with a 30-second pause between attempts. The job is submitted
with ``--requeue`` so SLURM can also reschedule it on node failure.

Generated ``run.sh`` Headers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   #SBATCH --job-name=<workdir_basename>
   #SBATCH --output=log/output_%A_%a.out
   #SBATCH --error=log/error_%A_%a.err
   #SBATCH --array=0-<N>
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=<n_proc>
   #SBATCH --time=<walltime>
   #SBATCH --mem=<memory>
   #SBATCH --requeue
   # (optional) #SBATCH --constraint=<slurm_constraint>
   # (optional) #SBATCH --partition=<slurm_partition>


CLI Reference
=============

``pycwb batch-setup``
---------------------

Use this command when your working directory is already initialised.

.. code-block:: text

   pycwb batch-setup <user_parameters.yaml> [OPTIONS]

.. list-table:: Options
   :header-rows: 1
   :widths: 28 12 60

   * - Flag
     - Short
     - Description
   * - ``--cluster``
     - ``-c``
     - Batch system: ``condor`` or ``slurm``
   * - ``--submit``
     - ``-s``
     - Submit immediately after generating scripts
   * - ``--work-dir``
     - ``-d``
     - Working directory (default: ``.``)
   * - ``--force-overwrite``
     - ``-f``
     - Overwrite existing job directories
   * - ``--conda-env``
     - ``-e``
     - Conda environment name
   * - ``--additional-init``
     - ``-a``
     - Extra shell commands before pycwb
   * - ``--accounting-group``
     - ``-g``
     - HTCondor accounting group
   * - ``--n-proc``
     - ``-n``
     - CPUs per job
   * - ``--job-per-worker``
     - ``-j``
     - Jobs bundled per worker script
   * - ``--memory``
     - ``-m``
     - Memory per job (e.g. ``6GB``)
   * - ``--disk``
     - ``-k``
     - Disk per job (e.g. ``8GB``)
   * - ``--container-image`` / ``--image``
     -
     - Container image URI (condor)
   * - ``--should-transfer-files``
     -
     - Enable HTCondor file transfer
   * - ``--walltime``
     -
     - SLURM time limit (e.g. ``72:00:00``)
   * - ``--slurm-constraint``
     -
     - SLURM ``--constraint`` value (e.g. ``cal``)
   * - ``--slurm-partition``
     -
     - SLURM partition name
   * - ``--n-retries``
     -
     - App-level retry count on failure (SLURM, default: ``5``)
   * - ``--list-n-jobs``
     -
     - Print job count and exit (no submission)


``pycwb config-setup``
----------------------

Use this command when starting from a config repository. It combines project
directory setup and batch job generation in one step.

.. code-block:: text

   pycwb config-setup <workdir> [OPTIONS]

All ``batch-setup`` flags are available, plus:

.. list-table:: Additional Options
   :header-rows: 1
   :widths: 28 12 60

   * - Flag
     - Short
     - Description
   * - ``--datatype``
     - ``-d``
     - Data source type (e.g. ``igwn-osg``, ``gwosc``, ``local``)
   * - ``--machine``
     -
     - Machine profile name (loads ``machine/<name>.yaml``)
   * - ``--config-base-path``
     - ``-c``
     - Path to config repository (default: ``./prototypes/config``)
   * - ``--dry-run``
     -
     - Preview setup without writing files


Full Parameter Reference
------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 22 18 10 10 18

   * - Parameter
     - YAML key
     - CLI flag
     - Condor
     - SLURM
     - Default
   * - Batch system
     - ``cluster``
     - ``--cluster``
     - ✓
     - ✓
     - *(none)*
   * - Conda environment
     - ``conda_env``
     - ``--conda-env``
     - ✓
     - ✓
     - *(none)*
   * - Extra init commands
     - ``additional_init``
     - ``--additional-init``
     - ✓
     - ✓
     - ``""``
   * - CPUs per job
     - ``nproc``
     - ``--n-proc``
     - ✓
     - ✓
     - ``1``
   * - Memory per job
     - ``job_memory``
     - ``--memory``
     - ✓
     - ✓
     - ``6GB``
   * - Disk per job
     - ``job_disk``
     - ``--disk``
     - ✓
     - ✓
     - ``8GB``
   * - Jobs per worker
     - ``job_per_worker``
     - ``--job-per-worker``
     - ✓
     - ✓
     - ``1``
   * - Accounting group
     - ``accounting_group``
     - ``--accounting-group``
     - ✓
     - —
     - ``""``
   * - Container image
     - ``container_image``
     - ``--container-image``
     - ✓
     - —
     - ``""``
   * - Transfer files
     - ``should_transfer_files``
     - ``--should-transfer-files``
     - ✓
     - —
     - ``false``
   * - Walltime
     - ``job_walltime``
     - ``--walltime``
     - —
     - ✓
     - ``72:00:00``
   * - Constraint
     - *(CLI only)*
     - ``--slurm-constraint``
     - —
     - ✓
     - *(none)*
   * - Partition
     - *(CLI only)*
     - ``--slurm-partition``
     - —
     - ✓
     - *(none)*
   * - Retry count
     - *(CLI only)*
     - ``--n-retries``
     - —
     - ✓
     - ``5``


Tips
====

- **Check job count before submitting:** use ``--list-n-jobs`` to verify the
  expected number of array tasks without generating any scripts.
- **Config-file defaults vs. CLI:** CLI flags override ``user_parameters.yaml``.
  Set site-wide defaults in YAML; use CLI flags for one-off overrides.
- **Condor requires ``accounting_group``:** if not set in YAML or via
  ``--accounting-group``, the condor setup will raise an error.
- **SLURM ``constraint`` / ``partition`` are site-specific:** they are not
  stored in ``user_parameters.yaml`` by design. Pass them explicitly each
  time, or pre-set them in a site-specific wrapper script.
