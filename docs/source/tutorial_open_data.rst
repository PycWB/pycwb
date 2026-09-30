.. _tutorial_open_data:

Search Public Detector Data
===========================

**Question:** Can you recover a known event from detector strain and identify
it in the output catalog? **Prerequisite:** :doc:`start_here` and the input
preparation in :doc:`tutorial_signals`.

Download a short interval
-------------------------

``tutorial-work/open_data.yaml`` selects H1/L1 data for a 20-minute interval
around GW150914. It removes the synthetic injection block, uses 4096-Hz input
and the ``L1:LOSC-STRAIN`` / ``H1:LOSC-STRAIN`` channels in the downloaded
LOSC V1 frames, and analyzes zero lag.

.. code-block:: bash

   pycwb gwosc-data tutorial-work/open_data.yaml --work-dir tutorial-work
   pycwb run tutorial-work/open_data.yaml \
     --work-dir tutorial-work/runs/open_data --list-jobs
   pycwb run tutorial-work/open_data.yaml \
     --work-dir tutorial-work/runs/open_data
   pycwb progress --work-dir tutorial-work/runs/open_data

The download requires network access and writes frames, frame lists and DQ
files under ``tutorial-work/input``. Actual downloads can cover larger frame
intervals than the requested analysis. Inspect the listed jobs before starting
the search; the listing also prepares the work directory.

Locate the candidate
--------------------

.. code-block:: python

   import pandas as pd

   events = pd.read_parquet("tutorial-work/runs/open_data/catalog/catalog.parquet")
   candidate = events.loc[(events["time_H1"] - 1126259462.4).abs() < 1.0]
   print(candidate[["id", "time_H1", "time_L1", "rho", "net_cc"]])

Check completion first if this selection is empty. Inspect trigger products
with :doc:`tutorial_event_inspection`: compare detector timing, reconstructed
waveforms, and time-frequency structure. Do not require exactly one catalog
row; selections and clustering can produce multiple candidates.

This experiment demonstrates event recovery. It does not reproduce the
published event significance: that requires an appropriate background and
search configuration. Continue with :doc:`tutorial_background` for the
background calculation.

Use your own frames
-------------------

After completing this experiment, follow :ref:`analysis_local_frames` to adapt
the workflow to your own frames, channels and DQ. Configuration-repository
setup and data-source profiles are maintained in :doc:`config_repository`.

**Result to keep:** the selected candidate rows, their plots, the input
configuration, and a record of the data interval and DQ selection.
