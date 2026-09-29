.. _understanding_results:

Understanding Your Results
==========================

A trigger is a reconstructed candidate that passed the configured search cuts.
It is not automatically an astrophysical detection. The synthetic demo checks
that the software can recover a known signal; it does not measure a false-alarm
rate, population efficiency or production sensitivity.

Inspect a catalog
-----------------

.. code-block:: python

   from pycwb.modules.catalog import Catalog
   import pandas as pd

   catalog = Catalog.open("my_first_search/catalog/catalog.parquet")
   jobs = catalog.jobs  # also validates the referenced job manifest
   events = pd.read_parquet("my_first_search/catalog/catalog.parquet")
   print(events[["job_id", "lag_idx", "time_H1", "time_L1", "rho", "net_cc"]])

Each row is a trigger. Current detector-specific columns use the detector name
as a suffix; older catalogs may contain list-valued fields. Check ``events.columns``
and the catalog's ``pycwb_version`` metadata before combining releases.

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
     - Unit / interpretation
   * - ``job_id``, ``trial_idx``, ``lag_idx``
     - Analysis job, injection trial and lag identifier
     - Integer identifiers; resolve lag shifts through job metadata
   * - ``time_H1``, ``time_L1``
     - Reconstructed event time in each detector
     - GPS seconds; arrival times may differ between detectors
   * - ``central_freq_H1``, ``central_freq_L1``
     - Central frequency. As in cWB, the first detector's column holds the
       network waveform centroid; the other detectors hold a pixel-based
       network estimate, not a per-detector measurement
     - Hz; ``bandwidth_*`` and ``duration_*`` follow the same convention
       (:ref:`event_output`)
   * - ``rho``
     - Coherent ranking statistic for the configured search mode
     - Dimensionless; not a probability or FAR
   * - ``net_cc``
     - Network correlation statistic
     - Dimensionless; use with the documented likelihood definition
   * - ``ra``, ``dec``
     - Reconstructed celestial position
     - Degrees in trigger output; see :ref:`units_conventions`
   * - ``injection``
     - Associated simulation parameters when available
     - Nullable struct; missing association is not proof of a missed injection

The complete current field definitions and serialization rules are in
:py:class:`pycwb.types.trigger.Trigger`. Do not substitute zero for missing
values: zero is a valid value for some statistics, while null means unavailable.

Output directory
----------------

.. list-table::
   :header-rows: 1

   * - Path
     - Purpose
   * - ``config/user_parameters.yaml``
     - Configuration copied during run setup
   * - ``catalog/catalog.parquet``
     - Trigger rows and configuration/version metadata
   * - ``catalog/jobs.parquet``
     - Immutable job descriptions referenced by the master catalog
   * - ``catalog/progress.parquet``
     - Job/trial/lag completion, trigger counts and analyzed livetime
   * - ``trigger/``
     - Event-specific products selected by the output flags
   * - ``output/``
     - Additional pipeline products, including waveform files when enabled

See :doc:`catalog_format` for manifest identity, selection provenance, and
compatibility rules.

The CLI writes its run log to the terminal; redirect it to a file when needed.

Batch runs have fragment catalogs until merged. See :ref:`run_on_clusters` and
``pycwb merge --help``. Transfer the full catalog directory and any externally
referenced manifests; a lone Parquet file may not contain its job descriptions.

Empty catalogs and failed jobs
------------------------------

Use ``pycwb progress --work-dir my_first_search`` before interpreting trigger
counts. A completed job with zero triggers is different from a missing or failed
job. An empty catalog can be scientifically reasonable for background or weak
injections. The deliberately loud CLI example must recover its injection.

FAR, IFAR and efficiency
------------------------

FAR is estimated from an appropriate background and its selected analyzed
exposure. IFAR is its reciprocal when FAR is positive. State the time units,
selection, ranking model, and finite-background limitations when reporting it.
Neither a high ``rho`` nor a single successful injection establishes significance.
Use :ref:`postproduction_background` for background estimation and
:ref:`postproduction_efficiency` for injection-population recovery fractions.

Keep training samples separate from the background used to evaluate FAR.
Compute exposure from completed jobs and the actual interval/lag selection,
including vetoes and overlap handling. ``N_jobs × N_lags × segLen`` is only an
idealized check when every job has that usable duration and no selection losses.

Next, preserve an analysis using :ref:`reproducibility` and check the scope of
validation in :ref:`validation_status`.
