import copy
import logging

import numpy as np
import pandas as pd

from .utils import getcapname

logger = logging.getLogger(__name__)



def preprocess_events(events: pd.DataFrame, nifo: int, ML_options: dict, ML_caps: dict):
    """
    Preprocess the events DataFrame by extracting features and applying caps.
    Parameters
    ----------
    events : pd.DataFrame
        The DataFrame containing the events.
    nifo : int
        Number of interferometers.
    ML_options : dict
        Dictionary containing options for reading the events.
    ML_caps : dict
        Dictionary containing the caps for the features.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with additional features and caps applied.
    """
    # with open(catalog_file, 'rb') as f:
    #     catalog = orjson.loads(f.read())

    # catalog['events']

    # Convert catalog to DataFrame
    # events = pd.DataFrame(catalog['events'])
    # nifo = len(catalog['config']['ifo'])

    xvars=copy.deepcopy(ML_options['readfile(vars)'])
    # Extract per-IFO scalar columns from list columns only when they are not
    # already present as flat scalars (e.g. when loading from a flat Parquet
    # produced by uproot rather than the pycwb Catalog).
    _list_cols = ['rho', 'sSNR', 'duration', 'bandwidth', 'netcc', 'noise']
    for n in range(nifo):
        for col in _list_cols:
            dest = col + str(n)
            if dest in events.columns:
                continue  # already a flat scalar — skip the expensive apply
            if col not in events.columns:
                continue
            arr = events[col]
            # vectorised path for numpy/pandas arrays stored as fixed-length lists
            if hasattr(arr.iloc[0], '__len__'):
                events[dest] = arr.apply(lambda x, _n=n: x[_n] if len(x) > _n else None)
            # scalar column that was already named differently — nothing to do

    # add chunk = 1
    events['chunk'] = 0
    if 'ecor' in xvars and 'likelihood' in xvars:
        events['ecor/likelihood'] = events['ecor']/events['likelihood']
    if 'duration' in xvars and 'bandwidth' in xvars:
        events['duration0*bandwidth0'] = events['duration0']*events['bandwidth0']
        events['bandwidth0/duration0'] = events['bandwidth0']/events['duration0']
    if 'sSNR' in xvars and 'likelihood' in xvars:
        events['mSNR'] = np.minimum(events['sSNR0'],events['sSNR1'])
        for n in range(2,nifo): events['mSNR'] = np.minimum(events['mSNR'],events['sSNR'+str(n)])
        events['mSNR/likelihood'] = events['mSNR']/events['likelihood']
        for n in range(0,nifo): events['sSNR'+str(n)+'/likelihood'] = events['sSNR'+str(n)]/events['likelihood']

    if 'noise' in xvars:
        events['noise'] = 1/(events['noise0']*events['noise0'])
        for n in range(1,nifo): events['noise'] = events['noise']+1/(events['noise'+str(n)]*events['noise'+str(n)])
        events['noise'] = np.sqrt(1/events['noise'])

    # ── Lveto flat indices (not per-IFO; extract from list col if needed) ──────
    # Lveto[2] = Lveto2 feature used by blf/bhf/bld ML_list
    if 'Lveto' in events.columns:
        lveto_arr = events['Lveto']
        first = lveto_arr.iloc[0]
        n_lveto = len(first) if hasattr(first, '__len__') else 0
        for i in range(n_lveto):
            dest = f'Lveto{i}'
            if dest not in events.columns:
                events[dest] = lveto_arr.apply(lambda x, _i=i: float(x[_i]) if len(x) > _i else np.nan)

    # Qa = sqrt(Qveto[0])  — support both catalog column name ('qveto') and
    # flat-Parquet column name ('Qveto0') produced by tree_to_dataframe.
    if 'qveto' in events.columns:
        events['Qa'] = np.sqrt(events['qveto'].clip(lower=0))
    elif 'Qveto0' in events.columns:
        events['Qa'] = np.sqrt(events['Qveto0'].clip(lower=0))

    # check ML_options
    for option, value in ML_options.items():
        if(option=='Qp(index)'):
            # add Qp feature to xpd
            if (value!=1):
                print('\nError: ML_options(''Qp(index)'') must be 1, only one Qp feature is supported\n')
                exit(1)
            # qfactor = Qveto[1] (catalog name); Qveto1 = flat-Parquet name
            qfactor_col = 'qfactor' if 'qfactor' in events.columns else 'Qveto1'
            events['Qp'] = events[qfactor_col]/(2*np.sqrt(np.log10(np.minimum(200,events['ecor']))))
        if(option=='rho0(define)'):
            # select rho0 definition
            if (value!=0) and (value!=1):
                print('\nError: ML_options(''rho0(define)'') must be 0(std) or 1(new)\n')
                exit(1)
            if (value==0):  # standard -> rho0 = rho[0]
                # rho0_std may already be set (flat-Parquet path) or derived above
                src = 'rho0_std' if 'rho0_std' in events.columns else 'rho0'
                events['rho0'] = events[src]
            if (value==1):  # new -> rho0 = sqrt(ecor/(1+penalty*max(1,penalty)-1)
                events['rho0'] = np.sqrt(events['ecor']/(1+events['penalty']*(np.maximum(1,events['penalty'])-1)))

    for feature, cap in ML_caps.items():
        if(cap>0):
            feature_cap = getcapname(feature,cap)
            events[feature_cap] = events[feature]
            events.loc[events[feature_cap]>cap, feature_cap] = cap

    return events


def read_catalog_to_dataframe(catalog_file: str) -> tuple:
    """Read a Parquet catalog and return a pandas DataFrame suitable for XGBoost.

    Signal (injection) events receive ``classifier=1``; background events receive
    ``classifier=0``.  PyArrow list-type columns (``rho``, ``sSNR``, etc.) are
    left as Python lists so they are compatible with :func:`preprocess_events`.

    Parameters
    ----------
    catalog_file : str
        Path to the Parquet catalog file.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with all trigger columns plus a ``classifier`` column.
    nifo : int
        Number of interferometers read from the catalog config.
    config : dict
        Pipeline configuration dict from the catalog metadata.
    """
    from pycwb.modules.catalog import Catalog

    cat = Catalog.open(catalog_file)
    config = cat.config
    nifo = len(config.get("ifo", []))

    table = cat.triggers(deduplicate=True)
    df = table.to_pandas()

    # Classify events: injection struct present → signal (1), else background (0)
    if "injection" in df.columns:
        df["classifier"] = df["injection"].apply(lambda x: 0 if x is None else 1)
    else:
        df["classifier"] = 0

    logger.info(
        "Read %d triggers from %s (nifo=%d, bkg=%d, sig=%d)",
        len(df),
        catalog_file,
        nifo,
        (df["classifier"] == 0).sum(),
        (df["classifier"] == 1).sum(),
    )
    return df, nifo, config


def load_flat_parquet(bkg_path: str, sim_path: str) -> pd.DataFrame:
    """Load BKG and SIM flat-Parquet files and return a combined DataFrame.

    Background events receive ``classifier=0``; signal events receive
    ``classifier=1``.  Columns are the flat per-IFO scalars produced by
    ``tree_to_dataframe`` (e.g. ``rho0``, ``rho1``, ``Lveto0``, …).

    Parameters
    ----------
    bkg_path : str
        Path to the background Parquet file.
    sim_path : str
        Path to the signal/simulation Parquet file.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with a ``classifier`` column.
    """
    logger.info("Loading BKG : %s", bkg_path)
    bkg = pd.read_parquet(bkg_path)
    bkg["classifier"] = 0

    logger.info("Loading SIM : %s", sim_path)
    sim = pd.read_parquet(sim_path)
    sim["classifier"] = 1

    df = pd.concat([bkg, sim], ignore_index=True)
    logger.info(
        "Combined : %d events  (bkg=%d, sim=%d)",
        len(df), len(bkg), len(sim),
    )
    return df


def build_features_from_parquet(
    df: pd.DataFrame,
    nifo: int,
    search: str,
    config_file: str = None,
) -> tuple:
    """Orchestrate feature preprocessing for a flat-Parquet DataFrame.

    Calls :func:`~.config.xgb_config`, optionally applies user overrides from
    a *config_file*, then runs :func:`preprocess_events` and
    :func:`apply_training_cuts` — mirroring what :func:`~.training.train` does
    for pycwb-Catalog DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Combined BKG + SIM DataFrame with flat per-IFO columns
        (as returned by :func:`load_flat_parquet`).
    nifo : int
        Number of interferometers.
    search : str
        Search type: one of ``blf``, ``bhf``, ``bld``, ``bbh``, ``imbhb``.
    config_file : str, optional
        Path to a Python override file that defines an ``update_config``
        function with the signature::

            def update_config(xgb_params, ML_list, ML_caps, ML_balance, ML_options):
                ...

        The function mutates the five dicts/lists in-place after the base
        config has been loaded.  ``ML_list`` additions/removals that affect
        capped features are handled automatically via
        :func:`~.utils_extended.update_ML_list`.

    Returns
    -------
    events : pd.DataFrame
        Preprocessed DataFrame with all derived feature columns.
    xgb_params : dict
        XGBoost hyper-parameters from :func:`~.config.xgb_config`.
    ML_list : list[str]
        Feature names used for training.
    ML_caps : dict
        Feature cap values.
    ML_balance : dict
        Class-balance options.
    """
    import copy
    from .config import xgb_config
    from .utils_extended import update_ML_list

    logger.info("xgb_config  search=%s  nifo=%d", search, nifo)
    xgb_params, ML_list, ML_caps, ML_balance, ML_options = xgb_config(search, nifo)

    if config_file:
        from pycwb.utils.module import import_function_from_file
        ML_defcaps = copy.deepcopy(ML_caps)
        update_config = import_function_from_file(config_file, "update_config")
        update_config(xgb_params, ML_list, ML_caps, ML_balance, ML_options)
        update_ML_list(ML_list, ML_defcaps, ML_caps)
        logger.info("Applied user config from %s", config_file)

    logger.info("ML_list : %s", ML_list)
    logger.info("ML_caps : %s", ML_caps)

    logger.info("Running preprocess_events …")
    events = preprocess_events(df.copy(), nifo, ML_options, ML_caps)

    cuts = ML_balance.get("cuts(training)", "")
    if cuts:
        logger.info("Applying training cuts : %s", cuts)
        events = apply_training_cuts(events, cuts)

    logger.info("Events after preprocessing + cuts : %d", len(events))
    return events, xgb_params, ML_list, ML_caps, ML_balance


def apply_training_cuts(df: pd.DataFrame, cuts: str) -> pd.DataFrame:
    """Apply a simple rho0-based cut string (e.g. ``'rho0>6.5'``) to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame that must already contain the ``rho0`` column.
    cuts : str
        A pandas-compatible query string such as ``'rho0>6.5'``.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    if not cuts:
        return df
    before = len(df)
    df = df.query(cuts)
    logger.info("Training cuts '%s': %d → %d events", cuts, before, len(df))
    return df.reset_index(drop=True)
