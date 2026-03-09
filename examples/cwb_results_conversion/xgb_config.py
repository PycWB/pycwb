"""Custom XGBoost config override for the cwb_results_conversion example.

This file is loaded by ``build_features_from_parquet`` (and ``train()``) via
the ``--config`` flag.  It must define a single function with this exact
signature::

    def update_config(xgb_params, ML_list, ML_caps, ML_balance, ML_options):
        ...

The function receives the five dicts/lists already populated by the base
``xgb_config(search, nifo)`` call and mutates them in-place.
"""


def update_config(xgb_params, ML_list, ML_caps, ML_balance, ML_options):
    """Apply search-specific overrides on top of the base xgb_config values."""

    # rho0 expression used in cut strings
    xrho0 = 'sqrt(ecor/(1+penalty*(max(1.0,penalty)-1)))'

    # ── Feature caps ──────────────────────────────────────────────────────────
    ML_caps['rho0'] = 40

    # ── Class-balance / training cuts ─────────────────────────────────────────
    ML_balance['slope(training)']   = 'q=1'
    ML_balance['balance(training)'] = 'A=320'
    ML_balance['cuts(training)']    = xrho0 + '>6.5'

    # ── Prediction cuts ───────────────────────────────────────────────────────
    ML_options['cuts(prediction)'] = xrho0 + '>6.5'

    # ── XGBoost hyper-parameters ──────────────────────────────────────────────
    xgb_params['max_depth'] = 4

    # ── Feature list ─────────────────────────────────────────────────────────
    # Replace per-detector sSNR ratio with averaged mSNR ratio + energy feature
    ML_list.remove('sSNR0/likelihood')
    ML_list.append('mSNR/likelihood')
    ML_list.append('ecor/likelihood')

    # Add noise estimate feature
    ML_list.append('noise')

    # Remove veto feature not available in this dataset
    ML_list.remove('Lveto2')

    # ── Diagnostic plot toggles ───────────────────────────────────────────────
    ML_options['rho0(mplot*d)']['enable']    = True
    ML_options['norm(mplot*d)']['enable']    = True
    ML_options['netcc0(mplot*d)']['enable']  = True
    ML_options['Qa(mplot*d)']['enable']      = True
    ML_options['Qp(mplot*d)']['enable']      = True
    ML_options['penalty(mplot*d)']['enable'] = True
