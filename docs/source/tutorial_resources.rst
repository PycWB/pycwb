.. _tutorial_resources:
.. _bound-resources-resume-and-merge:

Compare Execution and Restart
=============================

**Question:** Does the same small search retain its result under a resource
budget, and does a second invocation recognize completed work? Use the
``all_sky`` baseline and prepared inputs from :doc:`tutorial_signals`.

Run with an explicit CPU budget
-------------------------------

``bounded.yaml`` changes only the execution settings of the shared example:

.. code-block:: yaml

   execution:
     profile: scalable
     memory_limit: 8GiB
     worker_memory: 2GiB
     cache_limit: 512MiB
     headroom: 512MiB
     batch_size: 1
     preload: "off"

.. code-block:: bash

   pycwb run tutorial-work/bounded.yaml --work-dir tutorial-work/runs/bounded
   pycwb progress --work-dir tutorial-work/runs/bounded --verbose

These reservations are for this small synthetic exercise. Admission is also
limited by currently available memory. Inspect ``execution/catalog.metrics.json``
for completion, the admitted budget and measured resource use. The general
meaning and limits of the settings are in :doc:`workflow_execution`.

Compare the recovered candidates:

.. code-block:: python

   from pathlib import Path
   import pandas as pd

   for name in ["all_sky", "bounded"]:
       run = Path("tutorial-work/runs") / name
       events = pd.read_parquet(run / "catalog/catalog.parquet")
       print(name, len(events))
       print(events[["id", "time_H1", "time_L1", "rho", "net_cc"]])

Both runs use the same source and noise. Record candidate counts, arrival
times and statistics before interpreting any timing or memory difference.

Resume the same experiment
--------------------------

For the same configuration and work directory, allow reuse of the existing
output directory explicitly:

.. code-block:: bash

   pycwb run tutorial-work/bounded.yaml \
     --work-dir tutorial-work/runs/bounded --force-overwrite

Run the inspection block again. The catalog and progress should retain the
same completed job without duplicate events. In the checked example, the
second invocation completed without processing a new segment. Keep the YAML
unchanged: the flag allows the existing output directory, and the saved
configuration must still match.

**Result to keep:** the baseline/bounded catalog comparison, execution metrics
and unchanged event count after the second invocation.

Apply this to your analysis
---------------------------

.. raw:: html

   <span id="prepare-a-batch-run"></span>
   <span id="optional-gpu-comparison"></span>

Use :doc:`workflow_execution` for choosing budgets and recovering unfinished
work, :doc:`run_on_clusters` for submission and merging, and :doc:`backends`
for a separate hardware comparison.
