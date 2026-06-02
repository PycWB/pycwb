"""XGBoost config override for O4_K21 BurstLF (blf) search.

Loaded by the training workflow via the ``--config`` / ``config_file``
parameter.  Must define:

    def update_config(xgb_params, ML_list, ML_caps, ML_balance, ML_options):
        ...

The function receives the five dicts/lists already populated by the base
``xgb_config("blf", nifo)`` call and mutates them in-place.

Adapted from ``examples/cwb_results_conversion/xgb_config.py`` for the
O4_K21 dataset characteristics.
"""


def update_config(xgb_params, ML_list, ML_caps, ML_balance, ML_options):
    """Apply O4_K21-specific overrides on top of the base blf xgb_config."""

    # rho0 expression used in cut strings (penalty-weighted coherent energy)
    xrho0 = 'sqrt(ecor/(1+penalty*(max(1.0,penalty)-1)))'

    # ── Feature caps ──────────────────────────────────────────────────────
    ML_caps['rho0'] = 40

    # ── Class-balance / training cuts ─────────────────────────────────────
    ML_balance['slope(training)']   = 'q=1'
    ML_balance['balance(training)'] = 'A=320'
    ML_balance['cuts(training)']    = xrho0 + '>6.5'

    # ── Prediction cuts (used when applying the trained model) ────────────
    ML_options['cuts(prediction)'] = xrho0 + '>6.5'

    # ── XGBoost hyper-parameters ──────────────────────────────────────────
    xgb_params['max_depth'] = 4
    xgb_params['n_estimators'] = 500    # reduce from 20000 for faster training
    xgb_params['nthread'] = -1          # use all CPU cores

    # ── Feature list ─────────────────────────────────────────────────────
    # Replace per-detector sSNR ratio with averaged mSNR ratio + energy feature
    if 'sSNR0/likelihood' in ML_list:
        ML_list.remove('sSNR0/likelihood')
    ML_list.append('mSNR/likelihood')
    ML_list.append('ecor/likelihood')

    # Add noise estimate feature
    ML_list.append('noise')

    # Remove veto feature not available in this dataset
    if 'Lveto2' in ML_list:
        ML_list.remove('Lveto2')

    # ── Diagnostic plot toggles ───────────────────────────────────────────
    ML_options.setdefault('rho0(mplot*d)', {})
    ML_options.setdefault('norm(mplot*d)', {})
    ML_options.setdefault('netcc0(mplot*d)', {})
    ML_options.setdefault('Qa(mplot*d)', {})
    ML_options.setdefault('Qp(mplot*d)', {})
    ML_options.setdefault('penalty(mplot*d)', {})

    ML_options['rho0(mplot*d)']['enable']    = True
    ML_options['norm(mplot*d)']['enable']    = True
    ML_options['netcc0(mplot*d)']['enable']  = True
    ML_options['Qa(mplot*d)']['enable']      = True
    ML_options['Qp(mplot*d)']['enable']      = True
    ML_options['penalty(mplot*d)']['enable'] = True
