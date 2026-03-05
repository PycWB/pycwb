"""
Extended utility functions for XGBoost training with pycwb JSON catalogs.
"""

import logging
import numpy as np

from .utils import getcapname

logger = logging.getLogger(__name__)


def get_balanced_tail(tpd, ML_caps, seed):
    """
    Balance high rho0 events (tail) by sampling to achieve equal number of signal and background events.
    
    This samples high rho0 SIM and BKG events to achieve required class balancing for (rho0 >= rho0_capvalue) events.
    The result is num(sim) = num(bkg) for high rho0 events.
    
    Parameters
    ----------
    tpd : pd.DataFrame
        Combined training DataFrame containing both signal and noise events.
    ML_caps : dict
        ML cap dictionary containing the rho0 cap value.
    seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    pd.DataFrame
        Balanced training DataFrame.
    """
    import pandas as pd
    
    print('\n -> start tail balance ...')
    
    # Get rho0 cap information
    rho0_capvalue = -1
    for feature, cap in ML_caps.items():
        if 'rho0' in feature:
            rho0_capname = getcapname(feature, cap)
            rho0_capvalue = cap
            break
    
    if rho0_capvalue < 0:
        raise ValueError('get_balanced_tail - error: rho0 cap not found in ML_caps')
    
    print(f'\n rho0_capname = {rho0_capname} - rho0_capvalue = {rho0_capvalue}')
    
    # Split into tail (high rho0) and bulk (low rho0)
    tail_sim = tpd[(tpd[rho0_capname] >= rho0_capvalue) & (tpd['classifier'] == 1)]
    tail_bkg = tpd[(tpd[rho0_capname] >= rho0_capvalue) & (tpd['classifier'] == 0)]
    bulk = tpd[tpd[rho0_capname] < rho0_capvalue]
    
    print(f'  Tail events: SIM={len(tail_sim)}, BKG={len(tail_bkg)}')
    
    # Balance tail events to have equal numbers
    n_tail = min(len(tail_sim), len(tail_bkg))
    
    if len(tail_sim) > n_tail:
        tail_sim = tail_sim.sample(n=n_tail, random_state=seed)
    if len(tail_bkg) > n_tail:
        tail_bkg = tail_bkg.sample(n=n_tail, random_state=seed)
    
    # Combine balanced tail with bulk
    tpd = pd.concat([bulk, tail_sim, tail_bkg], ignore_index=True)
    
    ncount = tpd[tpd['classifier'] == 0].shape[0]
    scount = tpd[tpd['classifier'] == 1].shape[0]
    print(f'  After balancing: SIM/BKG = {scount}/{ncount} = {float(scount)/float(ncount):.3f}')
    print('\n <- end tail balance')
    
    return tpd


def update_ML_list(ML_list, ML_defcaps, ML_caps):
    """
    Update ML_list based on changes to ML_caps.

    Mirrors the two-pass logic of the legacy ``cwb_xgboost.update_ML_list``:

    1. Remove every entry in ``ML_list`` that corresponds to any feature in
       ``ML_defcaps`` (whether capped or bare).
    2. Re-add entries for all features in the *updated* ``ML_caps``.

    Parameters
    ----------
    ML_list : list
        List of ML features to be updated in place.
    ML_defcaps : dict
        Default ML caps (deep-copied before user overrides are applied).
    ML_caps : dict
        Updated ML caps (after user ``update_config`` has run).
    """
    # Pass 1 – remove all entries that come from the default caps
    for cap_name, cap_value in ML_defcaps.items():
        if cap_value > 0:
            full_cap_name = getcapname(cap_name, cap_value)
        else:  # cap == 0 → bare feature name
            full_cap_name = cap_name
        if full_cap_name in ML_list:
            ML_list.remove(full_cap_name)
            print(f'  Removed feature: {full_cap_name}')

    # Pass 2 – add entries for the (possibly updated) caps
    for cap_name, cap_value in ML_caps.items():
        if cap_value > 0:
            new_name = getcapname(cap_name, cap_value)
        else:
            new_name = cap_name
        if new_name not in ML_list:
            ML_list.append(new_name)
            print(f'  Added feature: {new_name}')


def get_balanced_bulk(X_train, ML_caps, ML_balance, balance_type, dump=False, ofile_tag=""):
    """Balance the rho0 bulk (low-rho0 region) by applying per-bin sample weights to
    background events so that the SIM/BKG distributions are equalised.

    This is a pure-Python/NumPy migration of the legacy ``cwb_xgboost.get_balanced_bulk``
    function.  ROOT is **not** required.

    Parameters
    ----------
    X_train : pd.DataFrame
        Combined training set containing both SIM (classifier=1) and BKG
        (classifier=0) events.  Must already contain the capped rho0 column.
    ML_caps : dict
        Feature-cap dictionary that contains the rho0 cap value.
    ML_balance : dict
        Balance configuration dictionary with keys such as
        ``binning(training)``, ``nbins(training)``, ``slope(training)``,
        ``balance(training)``, ``smoothing(training)``.
    balance_type : str
        Either ``'training'`` or ``'tuning'``.
    dump : bool, optional
        If ``True``, save diagnostic plots alongside ``ofile_tag``.
    ofile_tag : str, optional
        Prefix for output plot file names (used when ``dump=True``).

    Returns
    -------
    X_train : pd.DataFrame
        DataFrame with a new ``weight1`` column containing per-event weights.
    weight : pd.Series
        Series of sample weights (same as ``X_train['weight1']``).
    """
    if balance_type not in ('tuning', 'training'):
        raise ValueError(
            f"get_balanced_bulk: unknown balance_type '{balance_type}'. "
            "Must be 'tuning' or 'training'."
        )

    binning = ML_balance[f'binning({balance_type})']
    nbins   = ML_balance[f'nbins({balance_type})']

    slope_str   = ML_balance[f'slope({balance_type})']
    balance_str = ML_balance[f'balance({balance_type})']
    smooth      = ML_balance[f'smoothing({balance_type})']

    if 'q=' not in slope_str:
        raise ValueError(
            f"get_balanced_bulk: 'slope({balance_type})' must have the form 'q=<value>', "
            f"got: '{slope_str}'"
        )
    q = float(slope_str.replace('q=', ''))

    if 'A=' not in balance_str:
        raise ValueError(
            f"get_balanced_bulk: 'balance({balance_type})' must have the form 'A=<value>', "
            f"got: '{balance_str}'"
        )
    A = float(balance_str.replace('A=', ''))

    # Locate capped rho0 column
    rho0_capvalue = -1
    rho0_capname  = None
    for feature, cap in ML_caps.items():
        if 'rho0' in feature:
            rho0_capname  = getcapname(feature, cap)
            rho0_capvalue = cap
            break
    if rho0_capvalue < 0:
        raise ValueError('get_balanced_bulk: rho0 not found in ML_caps; it must be defined.')

    logger.info(' rho0_capname = %s - rho0_capvalue = %s', rho0_capname, rho0_capvalue)
    print(f'\n -> start bulk balance ...\n rho0_capname = {rho0_capname} - rho0_capvalue = {rho0_capvalue}')

    X_train = X_train.copy()
    X_train['weight1'] = 1.0

    # Select reference distribution for bin edges
    if binning in ('bkg(percentiles)', 'linear'):
        classifier_ref = 0
    elif binning == 'sim(percentiles)':
        classifier_ref = 1
    else:
        raise ValueError(
            f"get_balanced_bulk: unsupported binning '{binning}'. "
            "Allowed: 'bkg(percentiles)', 'sim(percentiles)', 'linear'."
        )

    rho0_ref = X_train.loc[
        (X_train[rho0_capname] < rho0_capvalue) & (X_train['classifier'] == classifier_ref),
        rho0_capname,
    ].values

    if binning in ('bkg(percentiles)', 'sim(percentiles)'):
        rho0_sorted = np.sort(rho0_ref)
        perc = np.linspace(0, 100 * 0.99999, nbins + 1)
        bins = np.percentile(rho0_sorted, perc)
    else:  # linear
        bins = np.linspace(rho0_ref.min(), rho0_capvalue * 0.99999, nbins + 1)

    hist_b, _ = np.histogram(X_train.loc[X_train['classifier'] == 0, rho0_capname], bins=bins)
    hist_s, _ = np.histogram(X_train.loc[X_train['classifier'] == 1, rho0_capname], bins=bins)

    # Compute per-bin weights for BKG events  (W[i] balances SIM/BKG in bin i)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(hist_b > 0, hist_s / hist_b, 1.0)
    W = [ratio[i] * np.exp(np.log(A) * (1.0 - i / (nbins - 1)) ** q)
         for i in range(nbins)]

    if smooth:
        try:
            import statsmodels.api as sm
            lowess = sm.nonparametric.lowess(W, bins[:-1], frac=0.1)
            W = list(lowess[:, 1])
        except ImportError:
            logger.warning('statsmodels not available; skipping weight smoothing.')

    # Apply weights to BKG bulk events
    for i in range(nbins):
        mask = (
            (X_train[rho0_capname] >= bins[i]) &
            (X_train[rho0_capname] <= bins[i + 1]) &
            (X_train['classifier'] == 0)
        )
        X_train.loc[mask, 'weight1'] = W[i]

    # Optional diagnostic plots
    if dump and ofile_tag:
        try:
            from .plots import plot_balance_weight, plot_balance_hist, plot_balance_weight_hist
            plot_balance_weight(ofile_tag + '_balance_weight.png', bins, W)
            plot_balance_hist(ofile_tag + '_balance_hist.png', X_train, rho0_capname, bins)
            plot_balance_weight_hist(
                ofile_tag + '_balance_weighted_hist.png', X_train, rho0_capname, bins
            )
        except Exception as exc:
            logger.warning('Could not save bulk balance plots: %s', exc)

    weight = X_train['weight1']
    print('\n <- end bulk balance')
    return X_train, weight
