.. _postproduction_efficiency:

Detection Efficiency
====================

.. stage-nav:: postproduction
   :current: efficiency

This guide explains how pycWB computes detection efficiency—the probability
of recovering an injected signal as a function of its parameters—and how
efficiency curves are used to characterize search sensitivity.

.. contents:: Table of Contents
   :depth: 2
   :local:


Overview
--------

Detection efficiency measures the fraction of simulated signals recovered by
the search pipeline. pycWB reports efficiency as a function of:

- **Signal amplitude**: the injected :math:`h_{rss}` (``sim_hrss``). Only
  fixed-:math:`h_{rss}` populations are supported; target-SNR or SNR-scaled
  injections are rejected by the efficiency actions.
- **Waveform**: one curve per injected waveform name (``sim_name``).

The key metrics are **hrss50** and **hrss90**—the root-sum-squared strain
amplitude at which 50% and 90% of injections are recovered, respectively.


Computing Efficiency
--------------------

Efficiency computation follows these steps (as implemented by the
``postprocess.plot_efficiency`` actions):

1. **Match simulations**: ``match_simulations`` with ``how: right`` writes one
   row per injection (unique ``sim_sim_idx``). Recovered injections carry the
   matched trigger's columns (non-null ``id``); missed injections keep null
   trigger columns and stay in the denominator.

2. **Score simulations**: ranking values are joined to the matched rows by
   trigger ``id``, either from a pre-scored SIM catalog (``scored_file``) or
   by scoring with ``model_file``. Injections without a trigger, or whose
   trigger was removed by the prediction cuts, have no score.

3. **Apply the IFAR threshold**: an injection is *detected* if it is recovered
   and the inclusive empirical background tail of its score satisfies

   .. math::

      \frac{N_{bkg}(\rho_{bkg} \geq \rho)}{T_{bkg}} \le \frac{1}{\text{IFAR}}

   where the background values are the ``ranking_par`` column of
   ``bkg_catalog`` (the scored FAR background), :math:`T_{bkg}` is
   ``livetime`` in seconds and IFAR is parsed from ``ifar``.

4. **Count injections per waveform and per injected amplitude**: for each
   ``sim_name`` and each distinct ``sim_hrss`` value (no amplitude binning):

   .. math::

      \epsilon(h_{rss}) = \frac{N_{detected}(h_{rss})}{N_{injected}(h_{rss})}

   where :math:`N_{injected}` counts unique ``sim_sim_idx``. With
   ``exclude_vetoed: true``, injections flagged ``sim_vetoed_cat0``,
   ``sim_vetoed_cat1``, ``sim_vetoed_cat2`` or ``sim_across_segments`` are
   removed from numerator and denominator; with the default ``false`` they
   remain and count as missed unless detected.

   The plotted error bars are the binomial standard error:

   .. math::

      \sigma_\epsilon = \sqrt{\frac{\epsilon (1 - \epsilon)}{N_{injected}}}

5. **Fit efficiency curve**: a cWB sigmoid (``logNfit``) is fitted to the
   per-amplitude efficiencies of each waveform (see `hrss50 and hrss90`_).


Efficiency Workflow Steps
-------------------------

Matching and scoring simulations (``evaluate_efficiency`` writes the scored
SIM catalog used as ``scored_file``; its ``threshold`` only affects the
returned summary, the fraction of scored rows with ``xgb_prob`` at or above
it):

.. code-block:: yaml

   - id: sim_eval_match
     name: Match SIM Evaluation Catalog
     action: postprocess.matching.match_simulations
     inputs:
       catalog_file: ${paths.sim_eval_catalog}
       simulation_file: ${paths.sim_eval_simulations}
     args:
       how: right                    # one row per simulation for efficiency
       window_buffer: 0.0
     outputs:
       output_file: tmp://sim_eval_matched_right.parquet

   - id: sim_efficiency_score
     name: Score SIM Evaluation Triggers
     action: postprocess.evaluate.evaluate_efficiency
     inputs:
       catalog_file: ${paths.sim_eval_catalog}
       model_file: ${paths.model_file}
       config_file: ${paths.config_file}
     args:
       threshold: 0.5
     outputs:
       output_file: ${paths.sim_eval_scored}

Computing efficiency vs. :math:`h_{rss}` per waveform at a fixed IFAR
(``bkg_catalog`` is the ``scored_catalog`` written by ``evaluate_far_rho``, and
``livetime`` is the livetime of that FAR background):

.. code-block:: yaml

   - id: waveform_hrss_curves_100yr
     name: Sensitivity Curves At 100-Year IFAR
     action: postprocess.plot_efficiency.compute_efficiency_vs_hrss_by_waveform
     inputs:
       sim_catalog: ${paths.sim_eval_catalog}
       matched_file: "@sim_eval_match.matched_file"
       bkg_catalog: ${paths.bkg_far_scored}
       model_file: ${paths.model_file}
       config_file: ${paths.config_file}
     args:
       livetime: "@bkg_split.far.livetime.seconds"
       ranking_par: rhor
       scored_file: ${paths.sim_eval_scored}
       ifar: 100yr
       use_unique_sim: true
       exclude_vetoed: false
     outputs:
       output_file: ${paths.output_dir}/simulations/efficiency_vs_hrss_by_waveform_100yr.png
       fit_parameters_file: ${paths.output_dir}/simulations/fit_parameters_by_waveform_100yr.csv


hrss50 and hrss90
-----------------

The per-waveform actions
(:py:func:`~pycwb.modules.postprocess.plot_efficiency.compute_efficiency_vs_hrss_by_waveform`
and
:py:func:`~pycwb.modules.postprocess.plot_efficiency.compute_hrss50_by_waveform_csv`)
fit the cWB sigmoid
:py:func:`pycwb.modules.statistics.sigmoid_fit.logNfit` to the points
:math:`(\log_{10} h_{rss}, \epsilon)` with Minuit
(:py:func:`pycwb.modules.statistics.sigmoid_fit.fit`). With
:math:`y = \pm(\log_{10} h_{rss} - \log_{10} h_{rss}^{50})` (the sign set by
the orientation flag), the fitted curve is

.. math::

   \epsilon = \begin{cases}
     \tfrac{1}{2}\,\mathrm{erfc}\!\left(|y|/s\right), & y < 0,
       \quad s = \sigma\, e^{\beta_- y} \\
     1 - \tfrac{1}{2}\,\mathrm{erfc}\!\left(y/s\right), & y > 0,
       \quad s = \sigma\, e^{\beta_+ y}
   \end{cases}

and :math:`\epsilon = 0.5` at :math:`y = 0`. When :math:`\beta_+ y > 1` the
code uses :math:`s = \sigma \beta_+ e` and :math:`y = 1`. Both orientation
flags are tried and the fit with the lower :math:`\chi^2` is kept; the
:math:`\chi^2` is unweighted (residuals divided by the standard deviation of
the efficiency values), not binomially weighted.

- **hrss50** is the fitted parameter :math:`10^{\log_{10} h_{rss}^{50}}`
  (bounded to :math:`10^{-25}`–:math:`10^{-19}`), with ``hrssEr`` from its
  fit error.
- **hrss10** and **hrss90** are the amplitudes where the fitted curve crosses
  0.1 and 0.9, root-found only inside the sampled :math:`h_{rss}` range; they
  are NaN when the curve does not reach that level there.
- The fit is not attempted when all efficiencies are below 0.5
  (status ``above_sampled_range``, ``hrss50`` empty, ``bound`` = largest
  :math:`h_{rss}`), all are above 0.5 (``below_sampled_range``), or fewer
  than three amplitudes are available (``skipped``).

:py:func:`~pycwb.modules.postprocess.plot_efficiency.compute_hrss50` (and its
plotting alias
:py:func:`~pycwb.modules.postprocess.plot_efficiency.plot_efficiency_vs_hrss`)
instead pools all injections in ``matched_right_file`` into one curve and
reports only hrss50, by linear interpolation in :math:`\log h_{rss}` between
the two amplitudes that bracket 50% efficiency. It returns no value when 50%
is not bracketed, and computes no hrss90.


Efficiency by Waveform Type
---------------------------

Efficiency is computed separately for each injected waveform name
(``sim_name``), characterizing the search's sensitivity to different signal
morphologies (for example sine-Gaussian bursts at various central frequencies
and Q-factors, or white-noise bursts), provided each population has fixed
:math:`h_{rss}` values.

Grouping is automatic; there is no filter argument. For plotting, the Q-factor
and frequency are parsed from the waveform name (native ``..._Q<q>_...`` /
``..._<f>Hz...`` names or cWB ``SG<f>Q<q>`` / ``SGE<f>Q<q>`` names); names
without a Q-factor are drawn in a ``Q = 0`` panel.
:py:func:`~pycwb.modules.postprocess.plot_efficiency.compute_efficiency_by_waveform`
reports one efficiency per waveform, pooled over all amplitudes, together
with the fraction recovered by cWB regardless of the IFAR threshold.


Visualization
-------------

Efficiency figures are rendered by the helpers in
:py:mod:`pycwb.modules.postprocess.efficiency_plots` and included in the
HTML report (:ref:`postproduction_workflow`) through the ``plots`` of each
``simulation_runs`` entry. The plots are:

- **Efficiency vs.** :math:`h_{rss}` **by waveform**: one panel per Q-factor,
  one curve per frequency, binomial error bars, the fitted sigmoid overlaid
  (when the fit succeeded) and a dotted line at each fitted hrss50
  (``compute_efficiency_vs_hrss_by_waveform``)
- **Pooled efficiency vs.** :math:`h_{rss}` with the interpolated hrss50
  (``compute_hrss50`` / ``plot_efficiency_vs_hrss``)
- **Per-waveform bar chart** of detection efficiency
  (``compute_efficiency_by_waveform``)


Configurable Thresholds
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Parameter
     - Default
     - Description
   * - ``ifar``
     - ``1mo`` (per-waveform actions); ``1yr`` (``compute_hrss50``,
       ``plot_efficiency_vs_hrss``)
     - IFAR threshold for detection (see `IFAR duration syntax`_)
   * - ``ifars``
     - ``1mo,1yr,10yr``
     - Comma-separated IFARs for ``compute_hrss50_by_waveform_csv``
   * - ``ranking_par``
     - ``xgb_prob``
     - Statistic used for both the background calibration and the injection
       scores
   * - ``scored_file``
     - none
     - Pre-scored SIM catalog, joined by ``id``; takes precedence over
       ``model_file``
   * - ``livetime``
     - required
     - Background livetime of ``bkg_catalog`` in seconds
   * - ``exclude_vetoed``
     - ``false``
     - Remove CAT0/1/2-vetoed and across-segment injections from the
       denominator
   * - ``use_unique_sim``
     - ``true``
     - Must be ``true``; other values raise an error


Interpreting Efficiency Results
-------------------------------

- **hrss50** represents the amplitude at which the search is 50% efficient—a
  common figure of merit for burst searches.
- **hrss90** is often quoted as the "sensitive range" of the search.
- **Flat efficiency at high amplitude**: loud signals should be recovered
  (efficiency → 1). With the default ``exclude_vetoed: false``, vetoed and
  across-segment injections, and injections whose trigger fails the
  prediction cuts, count as missed, so a plateau below 100% is expected when
  such injections exist. A plateau below 100% with ``exclude_vetoed: true``
  points to a pipeline problem.
- **Efficiency at low amplitude**: Should approach the false-alarm probability
  (not zero) due to accidental coincidences with background triggers.
- **Statistical uncertainty**: Binomial error bars shrink with more
  injections. For precise hrss50/hrss90, aim for at least several hundred
  injections per waveform type.


Validation Checks
-----------------

After computing efficiency, verify:

- **Efficiency saturates at 100% for loud signals**: with
  ``exclude_vetoed: true``, the efficiency curve should approach 1.0 at high
  :math:`h_{rss}`. If it plateaus below 100%, check for a pipeline bug (e.g.,
  waveform generation errors or matching problems). With the default
  ``exclude_vetoed: false``, compare the plateau with the fraction of
  non-vetoed injections first.
- **hrss50/hrss90 are consistent across waveform families**: similar waveform
  types should have similar sensitivity. Large outliers suggest injection
  parameter errors. Check the fit ``status`` column of the fit-parameters CSV
  (``fit_status`` in the hrss50 CSV) before comparing values.
- **Binomial error bars are reasonable**: with N injections per amplitude, the
  error is :math:`\sqrt{\epsilon(1-\epsilon)/N}`. Error bars > 20% indicate
  insufficient statistics.
- **Efficiency at low amplitude approaches FAR probability**: very faint
  signals are indistinguishable from background, so efficiency should
  approach (not equal) the false-alarm probability at threshold.


----

**See also:** :doc:`postproduction_xgboost` · :doc:`postproduction_background` · :doc:`injection_infrastructure`

**Next:** :doc:`postproduction_report` — assembling the final reports

**Apply the method:** :doc:`postproduction_study` describes the study workflow;
:doc:`analysis_recipes` routes other production tasks to their guides.

Manual simulation summary paths
-------------------------------

``pycwb simulation-summary --work-dir /production`` defaults to
``/production/config/user_parameters.yaml`` and writes
``/production/catalog/simulations.parquet``. An explicitly supplied config
or ``--output`` path is relative to the caller's directory. Relative paths
*inside* the config (DQ, frames, waveform inputs) are resolved from the
production directory, consistently with batch setup and execution.

IFAR duration syntax
--------------------

IFAR accepts positive numeric seconds (including scientific notation) or
positive durations with ``s``, ``day``, ``wk``, ``mo``, and ``yr`` suffixes,
such as ``100yr``. Existing presets retain their exact historical values;
``1mo`` is 30 days, whereas the historical ``6mo`` is half a Julian year.
Unknown or nonpositive values fail explicitly, including in MDC scoring.

Choosing a simulation association rule
--------------------------------------

Native ``match_simulations`` associates waveform/trigger interval overlaps
within the same trial and scheduled job, then chooses unique pairs. This is
the practical default for mixed waveform families and extended signals:
requested injection GPS time can denote a waveform endpoint, while trigger
GPS time is a reconstructed detector arrival. A blanket 0.1-second cut on
these two scalar columns does not reproduce cWB's detector-time cut.

Use ``ranking_par: rho_alt`` (or ``pycwb match-simulations --ranking-par
rho_alt``) when choosing unique events with the cWB ``pp_irho=1`` statistic.
The default ``rho`` remains available for native analyses. Rank selection
alone does not make the association algorithm cWB-equivalent.

For exact cWB postproduction comparison, use ``import_cwb_simulation`` on
ROOT/MDC truth: it uses cWB's recorded injection association and applies
``time_window: 0.1`` after unique selection. Failed winners stay missed;
a quieter candidate is not substituted. For native detector-time recovery
cuts, first establish compatible injected/reconstructed detector timing
and injection identity. Do not silently substitute a geocentric GPS cut.
Dense overlapping injections remain potentially ambiguous under interval
association; controlled populations with adequate separation are preferable.

Keeping background and sensitivity selections consistent
--------------------------------------------------------

Use the same ranking statistic for FAR, sensitivity, and the report. The
standard workflow uses ``rhor`` and supplies its already-scored SIM catalog::

    args:
      ranking_par: rhor
      scored_file: ${paths.sim_eval_scored}
      ifar: 100yr

These options are supported by the per-waveform efficiency, hrss curves,
hrss50, and hrss50 CSV actions. For ``compute_hrss50``, supply the injection
truth via ``matched_right_file``. Scores are joined by event ID; missed
injections and events removed by prediction cuts stay in the denominator.
Without ``scored_file``, model scoring applies the same user ranking hooks
and prediction cuts as ``score_catalog``. Probability ranking remains the
backward-compatible default, ``ranking_par: xgb_prob``.

Calibration uses inclusive empirical background tail counts, including ties.
No background exceedances means empirical FAR zero, not a measured infinite
exposure. Results identify ``ranking_par``, ``ranking_threshold``, and
``ifar_convention``. The legacy ``prob_threshold`` field is populated only
for probability ranking. Plot labels identify the selected statistic.
