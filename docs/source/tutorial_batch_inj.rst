.. _tutorial_batch_inj:

Batch Injection Runs
--------------------

Batch injection runs use the same YAML configuration as a local search, but
``pycwb batch-setup`` writes scheduler files for HTCondor or SLURM.

.. code-block:: bash

   pycwb batch-setup user_parameters.yaml \
     --cluster condor \
     --work-dir runs/injection_campaign \
     --n-proc 4

Submit immediately by adding ``--submit``:

.. code-block:: bash

   pycwb batch-setup user_parameters.yaml \
     --cluster slurm \
     --work-dir runs/injection_campaign \
     --n-proc 4 \
     --submit

Useful options include ``--job-per-worker`` for grouping multiple pycWB jobs in
one scheduler payload, ``--container-image`` for containerized HTCondor runs,
``--memory`` and ``--disk`` for resource requests, and the SLURM-specific
``--walltime``, ``--slurm-partition``, and ``--slurm-constraint`` flags.

For one-command project setup from a configuration repository, use
``pycwb config-setup``:

.. code-block:: bash

   pycwb config-setup O4_example_run \
     --config-base-path ./prototypes/config \
     --cluster condor \
     --submit
