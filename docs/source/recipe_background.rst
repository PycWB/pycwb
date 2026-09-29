.. _recipe_background:

Background-Only Production
==========================

**Goal:** Run a production background search (no injections) for FAR estimation.

**Inputs:**

- Real detector data with DQ files
- No ``injection`` block in config

**Key Config:**

.. code-block:: yaml

   # Exclude injection block entirely
   # Use production DQ and frame settings
   lagSize: 200          # More lags for better FAR statistics
   slagSize: 10          # Super lags for multi-detector

**Commands:** Same as all-sky search with ``--job-type BKG``.

**Expected Outputs:**

- Trigger catalog with zero-lag and non-zero-lag events
- Progress file with livetime per lag

**Validation Checks:**

- Zero-lag excluded from FAR calculation
- FAR vs. rho curve is smooth and monotonically decreasing
- Selected livetime agrees with completed progress and selected intervals after vetoes, edges and overlap accounting

**Common Failure Modes:**

- CAT2 veto windows applied as segments instead of windows
- ``lagOff`` incorrectly set (zero-lag leaks into background)
- Train/FAR split not respecting interval boundaries

See :doc:`analysis_recipes` for the other analysis tasks.
