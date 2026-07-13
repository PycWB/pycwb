.. _postproduction_efficiency:

Detection Efficiency
====================

.. rubric:: Postproduction: :doc:`triggers <postproduction_workflow>` → :doc:`background <postproduction_background>` → :doc:`ranking <postproduction_xgboost>` → **[efficiency]** ← you are here → :doc:`report <postproduction>`

This guide explains how pycWB computes detection efficiency—the probability
of recovering an injected signal as a function of its parameters—and how
efficiency curves are used to characterize search sensitivity.

.. contents:: Table of Contents
   :depth: 2
   :local:


Overview
--------

Detection efficiency measures the fraction of simulated signals recovered by
the search pipeline. Efficiency is typically reported as a function of:

- **Signal amplitude** (:math:`h_{rss}` for bursts, distance for CBC)
- **Waveform type** (sine-Gaussian, BBH, etc.)
- **Sky location** or other injection parameters

The key metrics are **hrss50** and **hrss90**—the root-sum-squared strain
amplitude at which 50% and 90% of injections are recovered, respectively.


Computing Efficiency
--------------------

Efficiency computation follows these steps:

1. **Score simulations**: Apply the trained XGBoost model to the simulation
   trigger catalog.

2. **Apply FAR threshold**: Select a fixed false-alarm-rate threshold (e.g.,
   FAR < 1/year) and count recovered injections above this threshold.

3. **Count injections per amplitude bin**: Group injections by :math:`h_{rss}`
   (or distance) and count both injected and recovered.

4. **Compute efficiency per bin**:

   .. math::

      \epsilon(h_{rss}) = \frac{N_{recovered}(h_{rss})}{N_{injected}(h_{rss})}

   with binomial error bars:

   .. math::

      \sigma_\epsilon = \sqrt{\frac{\epsilon (1 - \epsilon)}{N_{injected}}}

5. **Fit efficiency curve**: Fit a sigmoid function to the binned efficiency
   values for smooth interpolation.


Efficiency Workflow Steps
-------------------------

Scoring simulations:

.. code-block:: yaml

   - id: score_sim
     name: Score Simulation Catalog
     action: postprocess.evaluate.score_mdc_catalog
     inputs:
       catalog_file: "@sim_match.matched_file"
       model_file: "@model.model_file"
     args:
       ranking_statistic: xgb_ranking
     outputs:
       output_file: tmp://sim_scored.parquet

Computing efficiency:

.. code-block:: yaml

   - id: efficiency
     name: Compute Detection Efficiency
     action: postprocess.evaluate.evaluate_efficiency
     inputs:
       scored_file: "@score_sim.output_file"
       simulation_file: ${paths.simulations}
     args:
       far_threshold: 0.001           # 1/1000 years → ~1/year
       amplitude_column: hrss
       waveform_groups:               # Group by waveform type
         - name: SG_Q9
           filter: approximant == "SineGaussian" and Q == 9
         - name: BBH_35_35
           filter: mass1 == 35 and mass2 == 35
     outputs:
       efficiency_file: tmp://efficiency.parquet
       plots_dir: tmp://efficiency_plots/


hrss50 and hrss90
-----------------

The hrss50 and hrss90 values are computed by interpolating the efficiency
curve at 50% and 90% efficiency:

.. math::

   h_{rss}^{50} &= h_{rss} \text{ where } \epsilon(h_{rss}) = 0.50 \\
   h_{rss}^{90} &= h_{rss} \text{ where } \epsilon(h_{rss}) = 0.90

Implementation in
:py:func:`pycwb.modules.postprocess.efficiency_metrics._interpolate_hrss50`:

- Linear interpolation in log-space between efficiency bins
- Sigmoid fit (:py:func:`~._fit_efficiency_curve`) for smooth curves when
  statistics are limited


Efficiency by Waveform Type
---------------------------

Efficiency is typically computed separately for each waveform family to
characterize the search's sensitivity to different signal morphologies:

- **Sine-Gaussian bursts** at various central frequencies and Q-factors
- **BBH mergers** at various mass combinations
- **White-noise bursts** (WNB) for agnostic searches
- **Generic ADE** waveforms

The ``waveform_groups`` argument in the efficiency action defines filters
based on injection parameters to group signals for separate efficiency
computation.


Efficiency vs. Sky Location
---------------------------

When sufficient simulation statistics are available, efficiency can be mapped
across the sky using HEALPix to produce a **sensitivity sky map**:

.. math::

   \epsilon(\phi, \theta) = \frac{N_{recovered}(\phi, \theta)}{N_{injected}(\phi, \theta)}

This reveals directional sensitivity variations due to antenna pattern
asymmetries.


Visualization
-------------

Efficiency curves are plotted via
:py:func:`pycwb.modules.postprocess.efficiency_plots` and included in the
HTML report (:ref:`postproduction_workflow`). Typical plots include:

- **Efficiency vs.** :math:`h_{rss}` with hrss50/hrss90 annotations
- **Multi-panel** plots by waveform type
- **Sigmoid fit** overlay on binned data points
- **Sky map** of efficiency (when applicable)


Configurable Thresholds
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Typical Value
     - Description
   * - ``far_threshold``
     - 0.001 (1/1000 yr)
     - FAR threshold for detection :math:`[\text{yr}^{-1}]`
   * - ``amplitude_column``
     - ``hrss``
     - Column name for signal amplitude
   * - ``n_amplitude_bins``
     - 20
     - Number of amplitude bins for efficiency calculation
   * - ``confidence_level``
     - 0.9
     - Confidence level for error bars (binomial)


Interpreting Efficiency Results
-------------------------------

- **hrss50** represents the amplitude at which the search is 50% efficient—a
  common figure of merit for burst searches.
- **hrss90** is often quoted as the "sensitive range" of the search.
- **Flat efficiency at high amplitude**: All loud signals should be recovered
  (efficiency → 1). Failure to saturate at 100% indicates a pipeline bug.
- **Efficiency at low amplitude**: Should approach the false-alarm probability
  (not zero) due to accidental coincidences with background triggers.
- **Statistical uncertainty**: Binomial error bars shrink with more
  injections. For precise hrss50/hrss90, aim for at least several hundred
  injections per waveform type.


Validation Checks
-----------------

After computing efficiency, verify:

- **Efficiency saturates at 100% for loud signals**: the efficiency curve
  should approach 1.0 at high :math:`h_{rss}`. If it plateaus below 100%,
  check for a pipeline bug (e.g., injections outside segments, waveform
  generation errors).
- **hrss50/hrss90 are consistent across waveform families**: similar waveform
  types should have similar sensitivity. Large outliers suggest injection
  parameter errors.
- **Binomial error bars are reasonable**: with N injections per bin, the
  error is :math:`\sqrt{\epsilon(1-\epsilon)/N}`. Error bars > 20% indicate
  insufficient statistics.
- **Efficiency at low amplitude approaches FAR probability**: very faint
  signals are indistinguishable from background, so efficiency should
  approach (not equal) the false-alarm probability at threshold.


----

**See also:** :doc:`postproduction_xgboost` · :doc:`postproduction_background` · :doc:`injection_infrastructure`

**Next:** :doc:`analysis_recipes` — copy-paste workflows for production tasks
