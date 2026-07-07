.. _postproduction_xgboost:

XGBoost Classification
======================

.. rubric:: Postproduction: :doc:`triggers <postproduction_workflow>` → :doc:`background <postproduction_background>` → **[XGBoost ranking]** ← you are here → :doc:`efficiency <postproduction_efficiency>` → :doc:`report <postproduction>`

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
simulated signal events, producing a single **ranking statistic** used for
FAR assignment and detection efficiency.


Why XGBoost?
------------

XGBoost is chosen for several reasons:

- **State-of-the-art on tabular data**: Gradient-boosted trees consistently
  outperform deep learning on structured event features.
- **Fast training and inference**: GPU-accelerated, handles millions of events.
- **Interpretable**: Feature importance scores reveal which event properties
  drive the classification.
- **Robust to hyperparameters**: Works well with default settings; tuning
  provides modest gains.


Input Features
--------------

Features are derived from each trigger by
:py:func:`pycwb.modules.cwb_xgboost.preprocess_events`. The feature set
includes:

**Coherent Statistics**:
- :math:`\rho` — coherent network SNR
- :math:`\rho_0` — unsubtracted SNR (:math:`\sqrt{E_c}`)
- ``ecor`` / ``likelihood`` — core energy over likelihood ratio
- Network correlation :math:`cc`
- :math:`\chi^2` — goodness-of-fit statistic
- Null energy, coherent energy, disbalance

**Signal Morphology**:
- ``Qa`` — quality factor (central frequency / bandwidth)
- ``Qp`` — peak quality factor
- Frequency range (min, max, central)
- Duration, bandwidth

**Source Parameters** (when ``Search`` is ``CBC`` / ``BBH`` / ``IMBHB``):
- Chirp mass :math:`\mathcal{M}`
- ``rho0_40d0`` — SNR at reference frequency
- Frequency evolution slope

**Sky & Network**:
- Sky position (RA, Dec)
- Number of active interferometers
- Ellipticity, polarization fraction


Training Configuration
----------------------

Training is configured through the workflow YAML using the
:py:func:`~pycwb.modules.postprocess.train_xgboost.train_xgboost` action:

.. code-block:: yaml

   - id: model
     name: Train XGBoost
     action: postprocess.train_xgboost.train_xgboost
     inputs:
       bkg_catalogs:                    # Background training catalogs
         - /path/to/bkg_train_1/catalog.parquet
         - /path/to/bkg_train_2/catalog.parquet
       sim_catalogs:                    # Simulation training catalogs
         - /path/to/sim_train_1/catalog.parquet
         - /path/to/sim_train_2/catalog.parquet
     args:
       xgb_config: /path/to/xgb_config.py   # Feature & hyperparameter config
       n_estimators: 500
       max_depth: 6
       learning_rate: 0.1
       subsample: 0.8
       colsample_bytree: 0.8
       scale_pos_weight: auto          # Auto-balance BKG/SIM ratio
     outputs:
       model_file: /path/to/model.ubj

XGBoost Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``xgb_config.py`` file defines which features to use and how to construct
the ranking statistic:

.. code-block:: python

   # Example xgb_config.py
   features = [
       "rho", "rho0", "ecor_likelihood",
       "chi2", "network_correlation",
       "Qa", "Qp", "central_freq",
       "n_ifo", "ellipticity",
   ]

   # Feature transformations
   def transform(df):
       df["ecor_likelihood"] = df["ecor"] / df["likelihood"]
       return df

   # Label: 1 = signal, 0 = background
   label_column = "is_signal"


Model Output
------------

The trained model is saved in **UBJ** (Universal Binary JSON) format
(``.ubj`` extension), which is more compact and faster to load than standard
JSON. The model includes:

- The trained XGBoost ``Booster`` object
- Feature names and transformations
- Training metadata (number of events, feature importances)


Inference (Scoring)
-------------------

Trained models are applied to new catalogs via the scoring actions:

- :py:func:`~pycwb.modules.postprocess.evaluate.evaluate_far_rho` — score background for FAR
- :py:func:`~pycwb.modules.postprocess.evaluate.score_mdc_catalog` — score simulations for
  efficiency

The model outputs a single **ranking statistic** value per trigger, which
combines all input features into an optimal detection score.

.. code-block:: yaml

   - id: score_bkg
     name: Score Background FAR
     action: postprocess.evaluate.evaluate_far_rho
     inputs:
       triggers_file: "@bkg_split.far.triggers_file"
       model_file: "@model.model_file"
     args:
       ranking_statistic: xgb_ranking
     outputs:
       output_file: tmp://bkg_far_scored.parquet


Performance Considerations
--------------------------

- **Multi-catalog batching**: The training action accepts lists of background
  and simulation catalogs via ``bkg_catalogs`` and ``sim_catalogs``, enabling
  training across multiple observing chunks simultaneously.
- **Auto class balancing**: ``scale_pos_weight: auto`` automatically adjusts
  for imbalanced BKG/SIM event counts.
- **GPU acceleration**: XGBoost training and inference can use GPUs when
  available (set ``tree_method: gpu_hist`` in the config).


Feature Importance
------------------

After training, feature importance scores are available in the model metadata.
These show which event properties contribute most to the classification:

.. code-block:: text

   Feature          Importance
   ---------        ----------
   rho              0.285
   ecor_likelihood  0.192
   chi2             0.154
   Qa               0.123
   network_corr     0.098
   ...


Validation Checks
-----------------

After training an XGBoost model, verify:

- **Train and FAR samples are disjoint**: check that no job ID or time
  interval appears in both sets. Use progress Parquet files to verify.
- **Features are stable across chunks**: plot feature distributions for each
  training chunk. Large shifts indicate data quality issues or different
  noise conditions.
- **Model improves separation without pathological background sculpting**:
  the ranking statistic should separate BKG and SIM distributions clearly.
  The FAR curve with the model should be steeper than the SNR-only curve,
  but should never be flatter or bumpy.
- **Feature importances are physically reasonable**: SNR should be the
  dominant feature. If a low-level feature dominates, investigate data
  leakage or label errors.


----

**See also:** :doc:`postproduction_background` · :doc:`postproduction_trainingset` · :doc:`postproduction_efficiency`

**Next:** :doc:`postproduction_efficiency` — measuring detection sensitivity
