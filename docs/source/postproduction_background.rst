.. _postproduction_background:

Background Estimation
=====================

.. rubric:: Postproduction: :doc:`triggers <postproduction_workflow>` → **[background & FAR]** ← you are here → :doc:`ranking <postproduction_xgboost>` → :doc:`efficiency <postproduction_efficiency>` → :doc:`report <postproduction>`

This guide explains how pycWB estimates the accidental coincidence background
and constructs a false-alarm-rate (FAR) lookup table.

.. contents:: Table of Contents
   :depth: 2
   :local:


Overview
--------

The background—the rate at which noise fluctuations produce candidate events
with a given ranking statistic—is estimated from **time-shifted (non-zero lag)
data**. By applying time delays between detectors that are larger than the
gravitational-wave travel time, any real signal coincidence is broken, and the
resulting triggers represent the accidental background.


Background Estimation Method
----------------------------

pycWB uses the **lag-based** background estimation:

1. **Non-zero lags**: For each job segment, the analysis is repeated at
   multiple time shifts (lags) where one detector's data is shifted relative
   to the others. All lags except the physical zero-lag produce background
   triggers.

2. **Livetime accounting**: Each non-zero lag contributes an independent
   background measurement. The total background livetime is:

   .. math::

      T_{bkg} = N_{jobs} \times (N_{lags} - 1) \times T_{seg}

   where :math:`T_{seg}` is the effective analysis time per segment.

3. **FAR computation**: For a given ranking statistic threshold :math:`\rho^*`,
   the false alarm rate is:

   .. math::

      FAR(\rho^*) = \frac{N_{bkg}(\rho \geq \rho^*)}{T_{bkg}}

   where :math:`N_{bkg}(\rho \geq \rho^*)` is the number of background
   triggers with ranking statistic at or above :math:`\rho^*`.

The FAR computation is implemented in streaming fashion in
:py:func:`pycwb.modules.postprocess.far.far_rho_plot`, which reads Parquet
row groups to handle arbitrarily large catalogs without loading everything
into memory.


Zero-Lag Identification
-----------------------

Physical (zero-lag) triggers are identified by
:py:func:`pycwb.modules.postprocess.lag_filters.zero_lag_mask`:

- **Regular lag**: :math:`\text{lag\_idx} = 0` (or the index where lag offset = 0)
- **Segment shift**: :math:`\text{shift\_idx} = 0` (no super-lag shift)

All other lag/shift combinations contribute to the background.


Train/FAR Data Splitting
------------------------

To avoid bias, the background data is split into independent **training** and
**FAR** subsets:

- **Training set**: used to train the XGBoost ranking model
  (:ref:`postproduction_xgboost`).
- **FAR set**: held out for unbiased FAR computation.

Splitting strategies (configured via the ``split.by`` field):

- ``interval_livetime``: Splits by time intervals within jobs. Ensures
  the same physical data segment does not appear in both train and FAR
  through different lag/shift combinations. Recommended for production.
- ``job``: Splits by whole jobs. Simpler but may leak correlated noise
  between train and FAR if jobs overlap.
- ``fraction``: Simple random split of triggers. Fast but least rigorous.

Example split configuration:

.. code-block:: yaml

   - id: bkg_split
     name: Split Background Train/FAR
     action: postprocess.selection.trigger_selection
     inputs:
       catalog_file: ${paths.bkg_catalog}
       progress_file: ${paths.bkg_progress}
     args:
       exclude_zero_lag: true        # Only use non-zero-lag for BKG
       returns: [jobs, triggers, livetime]
       split:
         by: interval_livetime
         seed: 42
         fractions:
           train: 0.1                # 10% for training
           far: 0.9                  # 90% for FAR


FAR Lookup Table
----------------

The FAR vs. ranking statistic lookup table maps each possible ranking statistic
value to its corresponding false alarm rate. This table is used to:

1. Assign a FAR to each zero-lag candidate.
2. Determine detection thresholds for alerts (e.g., FAR < 1/year).
3. Compute search sensitivity (integrated FAR above threshold).

The FAR at a given threshold :math:`\rho^*` is computed by counting background
triggers above :math:`\rho^*` and dividing by the total background livetime.


Zero-Lag Significance
---------------------

For zero-lag candidates, the Poisson significance of observing :math:`k` or
more background events with ranking statistic :math:`\geq \rho` is:

.. math::

   p(k; \lambda) = 1 - e^{-\lambda} \sum_{i=0}^{k-1} \frac{\lambda^i}{i!}

where :math:`\lambda = FAR(\rho) \times T_{live}` and :math:`T_{live}` is the
total analyzed livetime.

This is implemented in
:py:func:`pycwb.modules.postprocess.zero_lag.zero_lag_report`.


Blind Analysis (Fake Open Box)
------------------------------

For blind analyses, pycWB supports a "fake open-box" procedure
(:py:func:`pycwb.modules.postprocess.fake_openbox.fake_openbox_report`) that
randomly selects time intervals to simulate an unblinding without looking at
the actual zero-lag data.


Config & CLI
------------

Key actions for background workflows:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Action
     - Purpose
   * - :py:func:`~pycwb.modules.postprocess.selection.trigger_selection`
     - Split triggers into train/FAR subsets
   * - :py:func:`~pycwb.modules.postprocess.evaluate.evaluate_far_rho`
     - Score background and build FAR lookup
   * - :py:func:`~pycwb.modules.postprocess.far.far_rho_plot`
     - Generate FAR vs. ranking statistic plots
   * - :py:func:`~pycwb.modules.postprocess.zero_lag.zero_lag_report`
     - Zero-lag significance analysis
   * - :py:func:`~pycwb.modules.postprocess.lag_filters.zero_lag_mask`
     - Identify physical zero-lag triggers
   * - :py:func:`~pycwb.modules.postprocess.random_filter.random_filter_parquet`
     - Randomly downsample catalogs


Validation Checks
-----------------

After running background estimation, verify:

- **Zero-lag is excluded from FAR background**: check that the number of
  background triggers matches :math:`N_{lags} - 1` times the per-lag average.
  If zero-lag leaks in, the FAR will be overestimated (too conservative).
- **Livetime matches expected**: :math:`T_{bkg} = N_{jobs} \times N_{nonzero-lags} \times T_{seg}`.
  Compare with ``progress.parquet`` totals.
- **Train/FAR split has no leakage**: verify that no physical time interval
  appears in both the training and FAR subsets. Plot interval overlap from
  progress files.
- **FAR curve is smooth and monotonically decreasing**: a bumpy or flat FAR
  curve indicates problems with livetime accounting or train/FAR leakage.


----

**See also:** :doc:`postproduction_xgboost` · :doc:`postproduction_trainingset` · :doc:`likelihood_guide`

**Next:** :doc:`postproduction_xgboost` — training a ranking classifier
