.. _tutorial_background:

Build a Small Time-slide Background
===================================

**Question:** How often does shifted detector data produce a candidate above
a chosen ranking threshold? First download the data in :doc:`tutorial_open_data`.

Run nonzero lags
----------------

``background.yaml`` reuses those frames with a small number of nonzero lags:

.. code-block:: yaml

   lagSize: 9
   lagStep: 1.0
   lagOff: 1
   lagMax: 0

.. code-block:: bash

   pycwb run tutorial-work/background.yaml \
     --work-dir tutorial-work/runs/background --list-jobs
   pycwb run tutorial-work/background.yaml \
     --work-dir tutorial-work/runs/background
   pycwb progress --work-dir tutorial-work/runs/background --verbose

Inspect the actual lag vectors and completion. ``lag_idx`` alone does not
describe all superlag/job shifts; the catalog's job metadata matters.

Select triggers and exposure together
-------------------------------------

.. code-block:: python

   from pathlib import Path
   import pandas as pd
   from pycwb.modules.postprocess.selection import trigger_selection

   run = Path("tutorial-work/runs/background").resolve()
   selected = trigger_selection(
       work_dir=str(run),
       catalog_file="catalog/catalog.parquet",
       progress_file="catalog/progress.parquet",
       exclude_zero_lag=True,
       outputs={"triggers_file": "selected_background.parquet"},
   )
   exposure = selected["livetime"]["seconds"]
   events = pd.read_parquet(run / "selected_background.parquet")
   print("Selected exposure [s]:", exposure)
   print("Selected triggers:", len(events))

This uses the same selection for event rows and analyzed exposure, including
the supported interval/lag accounting. See :doc:`postproduction_background`.

Plot a cumulative rate
----------------------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   rho = pd.to_numeric(events["rho"], errors="coerce").to_numpy()
   rho = rho[np.isfinite(rho)]
   if exposure <= 0 or len(rho) == 0:
       raise ValueError("Need positive selected exposure and background triggers to plot a rate")
   thresholds = np.unique(rho)
   far = np.array([(rho >= value).sum() / exposure for value in thresholds])
   plt.step(thresholds, far, where="post")
   plt.xlabel("Coherent ranking threshold")
   plt.ylabel("Empirical FAR [1/s]")
   plt.yscale("log")
   plt.savefig(run / "far.png")

An empty background is an outcome to investigate, not an infinite-significance
claim. With no events above a threshold, the empirical count is zero; report
the finite exposure and an appropriate limit rather than infinite IFAR. This
small exercise does not reproduce GW150914's published significance.

Before comparing a candidate with this curve, match its detector network,
search configuration, cuts and ranking definition. If you train a ranking
model, reserve separate background for FAR evaluation. Follow
:doc:`postproduction_study` to apply that workflow to your own study catalogs.

**Result to keep:** the selected exposure, trigger sample and FAR curve,
including the search and selection used to construct them.
