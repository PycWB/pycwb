.. _recipe_debugging:

Debugging a Failed Production
=============================

**Goal:** Diagnose and fix a failed or suspicious production run.

**Checklist:**

1. **Check logs** — ``log/`` directory, look for ERROR or traceback
2. **Check progress** — ``pycwb progress --work-dir .`` shows failed lags/jobs
3. **Check catalog** — is ``catalog.parquet`` non-empty? Reasonable row count?
4. **Check DQ** — do CAT0 segments cover your GPS range? Are CAT2 windows reasonable?
5. **Check frames** — do ``.gwf`` files exist for all detectors and times?
6. **Check zero-lag** — is it excluded from background?
7. **Check SNR distribution** — does it peak near ``netRHO``? Long tail?
8. **Check livetime** — compare selected exposure with completed progress and intervals, including veto and overlap losses
9. **Check memory** — did any job hit OOM? Check ``job_memory`` setting.
10. **Rescue failed lags** — identify incomplete job/trial/lag tuples with ``progress`` and consult :ref:`troubleshooting` before selecting work to rerun

**Common Root Causes:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Symptom
     - Likely Cause
   * - Zero triggers
     - ``netRHO`` too high, wrong frequency range, bad frame data
   * - All triggers at same time
     - Injection GPS times overlap with glitch
   * - FAR curve flat
     - Zero-lag leaked into background
   * - Efficiency near 0%
     - Injections not recovered (check SNR, GPS times, waveform params)
   * - Job OOM
     - ``segLen`` too long, ``healpix`` too high, ``job_memory`` too low
   * - Run time too long
     - ``healpix`` too high, ``lagSize`` too large, no parallelization

See :doc:`analysis_recipes` for the other analysis tasks.
