.. _tutorial_data_quality:

Choose Data and Account for Vetoes
==================================

**Question:** Which parts of the requested interval actually contribute to
the search? Start with the downloaded inputs from :doc:`tutorial_open_data`.

Inspect the DQ selection
------------------------

Each ``DQF`` entry contains detector, filename, category, time shift,
inversion flag and four-column-format flag. For example:

.. code-block:: yaml

   DQF:
     - [H1, input/H1_cat0.txt, CWB_CAT0, 0.0, false, false]
     - [H1, input/H1_cat1.txt, CWB_CAT1, 0.0, false, false]
     - [H1, input/H1_cat2.txt, CWB_CAT2, 0.0, false, false]

Include the corresponding entries for every detector. ``false`` for inversion
means the file describes accepted intervals; do not put a list of bad intervals
there without configuring its interpretation. CAT0/CAT1 define available good
data for job construction. CAT2 selection is applied through accepted windows
and lag-aware exposure. See :doc:`job_control` for the exact pipeline behavior.

Work through a small intersection
---------------------------------

.. code-block:: python

   from pycwb.modules.job_segment.dq_segment import merge_seg_list

   # H1 is available for [1000, 1060); L1 has a gap from 1020 to 1030.
   h1 = ([1000.0], [1060.0])
   l1 = ([1000.0, 1030.0], [1020.0, 1060.0])
   start, stop = merge_seg_list(h1, l1)
   print(list(zip(start, stop)))
   print(sum(b - a for a, b in zip(start, stop)))

The common intervals are ``[(1000, 1020), (1030, 1060)]``: 50 seconds,
before segment-length, edge and other selection rules. Despite the helper's
name, this operation intersects the two accepted-interval lists.

Inspect actual jobs and exposure
--------------------------------

.. code-block:: bash

   pycwb run tutorial-work/open_data.yaml \
     --work-dir tutorial-work/runs/open_data --list-jobs
   pycwb progress --work-dir tutorial-work/runs/open_data --verbose

Compare requested GPS limits with job boundaries. ``segEdge`` supplies wavelet
padding; ``segMLS`` constrains minimum job duration and ``segTHR`` constrains
the duration surviving CAT2 selection. A short accepted interval can therefore
produce no usable job. Changing these values changes the analyzed experiment.

.. code-block:: python

   import pandas as pd

   progress = pd.read_parquet("tutorial-work/runs/open_data/catalog/progress.parquet")
   print(progress.columns.tolist())
   print(progress[["job_id", "lag_idx", "livetime"]])

Use completed, selected intervals and their time shifts when estimating
background exposure. Do not multiply requested duration by requested lag count
without checking completion, overlap and vetoes. :doc:`tutorial_background`
uses the selection action to obtain exposure together with its trigger sample.

**Exercise:** add a known excluded interval to a copy of the selection, use a
new work directory, and compare job count, accepted exposure and candidate
recovery. Keep the original DQ files unchanged so the comparison is reproducible.

**Result to keep:** a table of requested, available and analyzed intervals,
with an explanation of every loss of exposure. For automatic time vetoes,
continue with :doc:`tutorial_conditioning`.
