.. _postproduction_xgboost:

XGBoost Classification
======================

.. stage-nav:: postproduction
   :current: ranking

This guide explains how pycWB uses XGBoost gradient-boosted trees to build a
ranking classifier that separates gravitational-wave signals from background
noise triggers.

.. contents:: Table of Contents
   :depth: 2
   :local:


Overview
--------

While the coherent network SNR :math:`\rho` is a powerful single statistic,
combining multiple event features with a machine-learning classifier
significantly improves search sensitivity. pycWB uses **XGBoost**
(eXtreme Gradient Boosting) to train a binary classifier on background and
simulated signal events. The classifier outputs a signal probability
(``xgb_prob``) per trigger; a ranking statistic used for FAR assignment and
detection efficiency (for example ``rhor``) is derived from it by a
user-defined hook (see `Inference (Scoring)`_).


Why XGBoost?
------------

XGBoost is chosen for several reasons:

- **State-of-the-art on tabular data**: Gradient-boosted trees consistently
  outperform deep learning on structured event features.
- **Fast training and inference**: histogram-based tree construction
  (``tree_method: hist``) handles millions of events.
- **Interpretable**: Feature importance scores reveal which event properties
  drive the classification.
- **Robust to hyperparameters**: Works well with default settings; tuning
  provides modest gains.


Input Features
--------------

The feature list (``ML_list``) comes from
:py:func:`pycwb.modules.cwb_xgboost.config.xgb_config` for the chosen
``search`` (``blf``, ``bhf``, ``bld``, ``bbh`` or ``imbhb``) and may be
modified by the user's ``update_config``. Catalog columns are renamed to cWB
names (``coherent_energy`` → ``ecor``, ``q_veto`` → ``qveto``,
``q_factor`` → ``qfactor``, per-detector ``<name>_<ifo>`` → ``<name><n>`` in
the catalog's ``ifo_list`` order), and derived features are computed by
:py:func:`pycwb.modules.cwb_xgboost.preprocess_events`.

**Default feature list**:

- ``blf`` / ``bhf`` / ``bld``: ``norm``, ``netcc0``, ``penalty``, ``Lveto2``,
  ``chunk``, ``sSNR<n>/likelihood`` for :math:`n = 0 \ldots n_{ifo}-2`,
  ``rho0_20d0``, ``Qa``, ``Qp``.
- ``bbh`` / ``imbhb``: ``norm``, ``netcc0``, ``penalty``, ``frequency0``,
  ``bandwidth0``, ``duration0``, ``Lveto2``, ``chirp1``, ``chirp3``, ``chunk``,
  ``sSNR<n>/likelihood`` for :math:`n = 0 \ldots n_{ifo}-2`, ``rho0_11d0``,
  ``Qa``, ``Qp``.

**Feature definitions** (as computed by ``preprocess_events``):

- :math:`\rho_0 = \sqrt{e_{cor} / (1 + p\,(\max(1, p) - 1))}` with
  :math:`p` = ``penalty`` (``ML_options['rho0(define)'] = 1``, the default);
  with ``rho0(define) = 0``, :math:`\rho_0` is the stored ``rho[0]``.
- ``rho0_<cap>`` — :math:`\rho_0` clipped at the cap ``ML_caps['rho0']``
  (20 for burst searches, 11 for ``bbh``/``imbhb``); the name encodes the cap,
  e.g. ``ML_caps['rho0'] = 40`` gives ``rho0_40d0``.
- ``Qa`` — :math:`\sqrt{Q_{veto}[0]}` (from ``q_veto``).
- ``Qp`` — :math:`Q_{veto}[1] / (2\sqrt{\log_{10}\min(200, e_{cor})})`
  (from ``q_factor``).
- ``sSNR<n>/likelihood`` — per-detector ``sSNR`` over ``likelihood``.
- ``netcc0`` — network correlation (``net_cc``); ``norm`` — packet
  normalisation (``packet_norm``); ``Lveto2`` — ``Lveto[2]``; ``chunk`` — a
  constant 0.
- ``frequency0``, ``bandwidth0``, ``duration0`` — values of the first detector
  in the catalog's detector order; ``chirp1``, ``chirp3`` — elements 1 and 3
  of the cWB ``chirp`` array.

Additional derived features available to ``update_config`` include
``ecor/likelihood``, ``mSNR/likelihood`` (minimum ``sSNR`` over detectors,
divided by ``likelihood``) and ``noise``
(:math:`(\sum_n 1/\text{noise}_n^2)^{-1/2}`).


Training Configuration
----------------------

Training is configured through the workflow YAML using the
:py:func:`~pycwb.modules.postprocess.train_xgboost.train_xgboost` action
(``bkg_split`` is the split step shown in :ref:`postproduction_background`):

.. code-block:: yaml

   - id: model
     name: Train XGBoost Classifier
     action: postprocess.train_xgboost.train_xgboost
     inputs:
       bkg_catalogs:                    # Background selected upstream
         - "@bkg_split.train.triggers_file"
         - /path/to/bkg_train_1/selected_background.parquet
       sim_catalogs:                    # Recovered, non-vetoed SIM triggers
         - /path/to/sim_train_1/clean_matched.parquet
         - /path/to/sim_train_2/clean_matched.parquet
       config_file: ${paths.config_file}   # update_config(...) and ranking hooks
     args:
       model_file: ${paths.model_file}
       search: blf
       dump: false
       dump_training_review: true
       verbose: false
     outputs:
       training_settings_file: ${paths.xgb_training_settings}
       training_output_file: ${paths.xgb_training_output}

Training uses these catalogs as given. Prepare every input upstream: select
background with ``trigger_selection`` (``exclude_zero_lag: true`` removes zero
lag), and keep recovered, non-vetoed simulation triggers with
``filter_real_simulation``. Training logs a warning when a background catalog
still contains triggers with no time or segment shift, but does not remove them,
so a workflow may deliberately use a different reference lag.

XGBoost hyper-parameters are **not** action arguments: extra keys such as
``n_estimators`` or ``max_depth`` under ``args`` are ignored. They are set in
``xgb_params`` inside ``update_config``. Defaults from ``xgb_config``:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Parameter
     - ``blf`` / ``bhf`` / ``bld``
     - ``bbh`` / ``imbhb``
   * - ``learning_rate``
     - 0.03
     - 0.03
   * - ``max_depth``
     - 6
     - 13
   * - ``min_child_weight``
     - 5.0
     - 10.0
   * - ``subsample`` / ``colsample_bytree``
     - 0.6 / 1.0
     - 0.6 / 1.0
   * - ``gamma``
     - 2.0
     - 2.0
   * - ``n_estimators``
     - 20000
     - 20000

Common to all searches: ``objective: binary:logistic``, ``tree_method: hist``,
``grow_policy: lossguide``, ``scale_pos_weight: 1.0``, ``seed: 150914``,
``nthread: 1``; unless set in ``xgb_params``, training uses
``eval_metric: [logloss, auc, aucpr]`` and ``early_stopping_rounds: 50``.

**Training procedure** (as implemented):

1. BKG catalogs selected upstream are concatenated and labelled
   ``classifier = 0``; training preserves that selection and warns about any
   unshifted (zero-lag) triggers it contains. SIM catalogs are
   labelled ``classifier = 1``;
   rows flagged ``sim_vetoed_cat0``, ``sim_vetoed_cat2`` or
   ``sim_across_segments`` are removed when those columns exist. Every other
   SIM row is treated as signal, so clean SIM catalogs first with
   :py:func:`~pycwb.modules.postprocess.selection.filter_real_simulation`.
2. ``preprocess_events`` builds the features, then ``cuts(training)``
   (default :math:`\rho_0 > 6.5`) is applied to both classes.
3. **Tail balance** (``tail(training)``, default on): SIM events with
   ``rho0_<cap>`` :math:`\geq` cap are resampled (seeded) to the number of
   BKG events in that tail; all BKG events are kept.
4. The merged set is split 90 % / 10 % into training and evaluation samples
   (``random_state`` = ``xgb_params['seed']``).
5. **Bulk balance** (``bulk(training)``, default on): below the cap, BKG
   events receive sample weights per bin,
   :math:`W_i = (N_{sim,i}/N_{bkg,i})\,A^{(1 - i/(n_{bins}-1))^{q}}`, with
   ``nbins(training)`` = 100 bins whose edges are SIM percentiles
   (``binning(training) = 'sim(percentiles)'``). Defaults are
   :math:`q = 6, A = 20` for burst searches and :math:`q = 1, A = 14` for
   ``bbh``/``imbhb``. SIM events keep weight 1.
6. ``XGBClassifier.fit`` runs with these sample weights and the evaluation
   sample for early stopping.

XGBoost Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``config_file`` (conventionally ``xgb_config.py``) must define
``update_config``, which receives the five objects returned by ``xgb_config``
and mutates them in place. It may also define ranking hooks used at scoring
time (``postprocess_scores``, ``postprocess_dataframe``, ``getrhor``, or names
listed in ``ML_options['ranking_statistics(functions)']``). The class label is
not configurable: it is the ``classifier`` column set from the catalog lists.

.. code-block:: python

   # Example xgb_config.py
   import numpy as np

   def update_config(xgb_params, ML_list, ML_caps, ML_balance, ML_options):
       xrho0 = 'sqrt(ecor/(1+penalty*(max(1.0,penalty)-1)))'
       ML_caps['rho0'] = 40                       # feature becomes rho0_40d0
       ML_balance['cuts(training)'] = xrho0 + '>6.5'
       xgb_params['max_depth'] = 4
       xgb_params['n_estimators'] = 500
       ML_list.append('ecor/likelihood')

   # Ranking hook: adds the ``rhor`` column after scoring
   def getrhor(xdp, search):
       source = "MLstat" if "MLstat" in xdp.columns else "xgb_prob"
       prob = xdp[source].astype(float).clip(upper=0.99999999)
       xdp["rhor"] = -np.log(1.0 - prob)
       return xdp


Model Output
------------

With a ``.ubj`` (Universal Binary JSON) or ``.json`` extension the model is
written with XGBoost's native ``save_model``; any other extension is pickled.
The native file holds the trained booster, feature names and the
``cwb-compatible-v1`` catalog preprocessing identifier.
Derived features are recomputed from ``config_file`` at scoring time, so the
same ``config_file`` must be passed to every scoring action.

With ``dump_training_review`` (or ``dump``, or an explicit
``training_settings_file`` / ``training_output_file``) two review files are
also written:

- ``training_settings_file`` (default ``<model stem>.cfg``) — ``xgb_params``,
  ``ML_list``, ``ML_caps``, ``ML_balance``, ``ML_options`` and row counts at
  each stage.
- ``training_output_file`` (default ``<model stem>.out``) — training summary,
  tree statistics, evaluation history and feature importances.

The action returns ``model_file`` and ``auc``, which is XGBoost's
``best_score`` on the evaluation sample: the last configured ``eval_metric``,
``aucpr`` by default.


Inference (Scoring)
-------------------

Training consumes background catalogs already selected upstream. Perform the
zero-lag split with ``trigger_selection`` before training or FAR estimation, as
in ``examples/postproduction/standard_analysis_10pct_workflow.yaml``. For that
selected FAR holdout, use ``exclude_zero_lag: false``; do not discard a selected
superlag merely because its regular ``lag_idx`` is zero.

Newly trained models record the ``cwb-compatible-v1`` catalog field conventions
alongside their trees. Scoring rejects a missing or incompatible convention
identifier before prediction. Retrain older PycWB native-catalog models: the
``norm`` and ``sSNR`` definitions and detector-indexed ordering have changed,
even though the feature names are unchanged.

Use ``postprocess.model_io.import_cwb_model`` for a trusted original cWB pickle.
For an unversioned portable model whose training inputs have been independently
verified to use the current cWB conventions, declare that in the existing
scoring configuration hook:

.. code-block:: python

   def update_config(xgb_params, ML_list, ML_caps, ML_balance, ML_options):
       ML_options["model_preprocessing"] = "cwb-compatible-v1"

This declaration does not convert incompatible old models and cannot override
an incompatible identifier already recorded in a model. Keep the training
feature caps and other configuration choices consistent with scoring as well.

Trained models are applied to new catalogs via the scoring actions:

- :py:func:`~pycwb.modules.postprocess.evaluate.evaluate_far_rho` — score the
  FAR background and build the FAR lookup table
- :py:func:`~pycwb.modules.postprocess.evaluate.score_catalog` — score any
  catalog (``lag_selection``: ``all``, ``zero_lag`` or ``nonzero_lag``)
- :py:func:`~pycwb.modules.postprocess.evaluate.evaluate_efficiency` — score a
  SIM catalog and write it for the efficiency actions
  (:ref:`postproduction_efficiency`)
- :py:func:`~pycwb.modules.postprocess.evaluate.score_mdc_catalog` — score
  the zero-lag triggers of a blind MDC catalog and list detections above an
  IFAR threshold

Each scoring action:

1. builds the model's features with ``preprocess_events`` (features absent
   from the catalog are filled with 0);
2. writes ``xgb_prob`` (the positive-class probability) and a copy named
   ``MLstat``;
3. runs the ranking hooks from ``config_file``, e.g. ``getrhor`` above giving
   :math:`\rho_r = -\ln(1 - \min(\text{MLstat}, 0.99999999))`;
4. drops triggers that fail ``cuts(prediction)`` (default
   :math:`\rho_0 > 7.2` for burst searches, :math:`\rho_0 > 6.5` for
   ``bbh``/``imbhb``).

The statistic used for FAR is selected with ``ranking_par``. It defaults to
``rho`` in ``evaluate_far_rho``, so set it explicitly.

.. code-block:: yaml

   - id: far_rho
     name: Score FAR Holdout And Build FAR(ρ)
     action: postprocess.evaluate.evaluate_far_rho
     inputs:
       catalog_file: "@bkg_split.far.triggers_file"
       model_file: ${paths.model_file}
       config_file: ${paths.config_file}
     args:
       livetime: "@bkg_split.far.livetime.seconds"
       exclude_zero_lag: false      # Preserve the upstream background selection
       ranking_par: rhor
       bin_size: 0.0001
       vmin: 0.0
       vmax: 10.0
     outputs:
       output_file: ${paths.far_rho_file}              # binned FAR JSON
       scored_catalog: tmp://bkg_far_scored.parquet    # scored FAR background


Performance Considerations
--------------------------

- **Multi-catalog batching**: The training action accepts lists of background
  and simulation catalogs via ``bkg_catalogs`` and ``sim_catalogs``, enabling
  training across multiple observing chunks simultaneously. Only the columns
  needed by the configured features are read from Parquet.
- **Class balancing**: SIM/BKG imbalance is handled by the tail resampling and
  bulk sample weights described above (``ML_balance``);
  ``scale_pos_weight`` stays at 1.0 unless changed in ``update_config``.
- **Threads and GPU**: ``xgb_params['nthread']`` defaults to 1; raise it in
  ``update_config`` to train on more cores. ``tree_method: gpu_hist`` is
  rejected by current XGBoost releases; for GPU training set
  ``xgb_params['device'] = 'cuda'`` and keep ``tree_method: hist``.


Feature Importance
------------------

After training, feature importance scores are written to the
``training_output_file`` (``.out``). It contains one section for each XGBoost
importance type (``weight``, ``gain``, ``cover``, ``total_gain``,
``total_cover``), with features sorted by decreasing score:

.. code-block:: text

   [feature_importance_gain]
     rho0_20d0                      = <score>
     netcc0                         = <score>
     penalty                        = <score>
     ...


.. raw:: html

   <span id="validation-checks"></span>

Evaluate the trained model
--------------------------

Keep model-training data separate from FAR and sensitivity evaluation samples.
With ``interval_livetime``, compare the selected (``shift_key``, ``lag_idx``)
intervals; the same job IDs can appear in both partitions. With a whole-job
split, compare the job lists.

Compare the trained ranking with the original statistic at the same FAR on an
independent evaluation population. Inspect feature distributions across data
chunks and the training diagnostics in :doc:`postproduction_report` when
investigating a change in performance.

----

**See also:** :doc:`postproduction_background` · :doc:`postproduction_trainingset` · :doc:`postproduction_efficiency`

**Next:** :doc:`postproduction_efficiency` — measuring detection sensitivity
