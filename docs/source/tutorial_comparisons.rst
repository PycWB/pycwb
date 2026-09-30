.. _tutorial_comparisons:

Compare Runs with a Controlled Change
=====================================

**Question:** What changes when you modify one search choice?
Complete ``all_sky``, ``fixed`` and ``patch`` in :doc:`tutorial_sky_masks`.
Keep each run's full ``catalog`` directory, including referenced job manifests.

Build the comparison
--------------------

.. code-block:: bash

   pycwb post-process examples/tutorials/compare.yaml --diagram-only
   pycwb post-process examples/tutorials/compare.yaml --no-diagram

Open ``tutorial-work/public/comparison/index.html``. The workflow reads the
three catalogs, retains their run identities, obtains scheduled injection truth
from job metadata, generates angle-comparison plots, and assembles an HTML report.
The inputs are declared explicitly:

.. literalinclude:: ../../examples/tutorials/compare.yaml
   :language: yaml
   :start-at: vars:
   :end-before: runtime:

The tutorial uses the default ``tutorial-work`` directory. If preparation used
another destination, copy the workflow and update ``vars.work_dir`` before
running it. Catalog paths are relative to that directory.

Read the plots as an experiment
-------------------------------

Compare recovered-source counts, sky-angle error and ranking statistics.
The reader preserves scheduled sources that produced no candidate, so a run
cannot appear better simply because its difficult cases vanished from the
input table. One loud source is a workflow demonstration; repeat over a
population before interpreting a distribution or claiming an improvement.

The angle comparison declares injection angles in radians and recovered
angles in degrees. The preparation script assigns stable ``sim_idx`` and
``trial_idx`` identities to each source; keep those identities aligned when
comparing the same population across runs. Keep ``strict_truth: true`` to
surface missing or invalid truth instead of silently comparing incompatible
angles.

Extend the comparison
---------------------

Add ``custom`` and ``offset`` runs to compare all mask types. For a network
experiment, substitute the ``hlv`` and ``custom_network`` catalogs from
:doc:`tutorial_detector_networks` and rename the report. Other useful controlled
changes are frequency band, wavelet resolution, clustering gaps and conditioning.
Record all differences in the configurations, including numerical conventions.

For matched-noise comparisons retain the shared detector seeds and source
population. For population-level conclusions also vary noise realizations;
a result from one favorable noise realization is not a sensitivity estimate.

**Result to keep:** a portable HTML report and manifest identifying each input
run and the one intended configuration change. To modify the workflow itself,
continue with :doc:`tutorial_custom_postproduction`.
