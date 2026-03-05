"""XGBoost training pipeline for pycWB – Parquet/Arrow-native implementation.

This module is the full migration of the legacy ``cwb_xgboost_training.py`` to
the current pycWB codebase.  It reads background and signal events directly from
Parquet catalogs (produced by :class:`~pycwb.modules.catalog.Catalog`) instead
of ROOT tree files.

Typical usage::

    from pycwb.modules.cwb_xgboost.training import train

    train(
        bkg_catalog_file="run_bkg/catalog/catalog.parquet",
        sim_catalog_file="run_sim/catalog/catalog.parquet",
        model_file="models/bbh_model.pkl",
        search="bbh",
        config_file="user_xgboost_config.py",   # optional
        dump=True,
    )
"""

from __future__ import annotations

import copy
import logging
import os
import pickle
import time
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from pycwb.utils.module import import_function_from_file

from .config import xgb_config
from .read_data import apply_training_cuts, preprocess_events, read_catalog_to_dataframe
from .utils import getcapname
from .utils_extended import get_balanced_bulk, get_balanced_tail, update_ML_list
from .plots import (
    plot_hist_rho, plot_hist_mchirp, plot_hist_freq,
    plot_QaQp, mplot1d, mplot2d, plot_roc, plot_pr,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_and_preprocess(
    catalog_file: str,
    classifier_value: int,
    nifo: int,
    search: str,
    ML_caps: dict,
    ML_options: dict,
    frac: float = 1.0,
    seed: int = 150914,
    cuts: str = "",
) -> pd.DataFrame:
    """Read one catalog and return a preprocessed DataFrame.

    Parameters
    ----------
    catalog_file : str
        Path to the Parquet catalog.
    classifier_value : int
        0 for background, 1 for signal.  This value is assigned to the
        ``classifier`` column regardless of the ``injection`` field in the
        catalog, so you can pass a pure-background or pure-signal catalog.
    nifo : int
        Number of detectors.
    search : str
        Search type (``bbh``, ``imbhb``, ``blf``, ``bhf``, or ``bld``).
    ML_caps : dict
        Feature caps used in :func:`~.read_data.preprocess_events`.
    ML_options : dict
        Processing options used in :func:`~.read_data.preprocess_events`.
    frac : float
        Fraction of events to keep (random sub-sample, default 1.0 = all).
    seed : int
        Random seed for sub-sampling.
    cuts : str
        Optional pandas query string applied after pre-processing
        (e.g. ``"rho0 > 6.5"``).

    Returns
    -------
    pd.DataFrame
        Preprocessed, optionally sub-sampled DataFrame.
    """
    df, cat_nifo, _ = read_catalog_to_dataframe(catalog_file)
    # Override classifier so the caller controls bkg/sim labelling
    df["classifier"] = classifier_value

    if nifo == 0:
        nifo = cat_nifo

    df = preprocess_events(df, nifo, ML_options, ML_caps)

    if cuts:
        df = apply_training_cuts(df, cuts)

    if frac < 1.0:
        n = max(1, int(len(df) * frac))
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
        logger.info("Sub-sampled to %d events (frac=%.2f)", n, frac)

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train(
    bkg_catalog_file: str,
    sim_catalog_file: str,
    model_file: str,
    search: str,
    nifo: int = 0,
    config_file: Optional[str] = None,
    nfrac: float = 1.0,
    sfrac: float = 1.0,
    dump: bool = False,
    verbose: bool = False,
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier from Parquet catalogs and save the model.

    Parameters
    ----------
    bkg_catalog_file : str
        Path to the background Parquet catalog.
    sim_catalog_file : str
        Path to the signal/injection Parquet catalog.
    model_file : str
        Destination path for the serialised model (``*.pkl``).
    search : str
        Search type: one of ``bbh``, ``imbhb``, ``blf``, ``bhf``, ``bld``.
    nifo : int, optional
        Number of interferometers.  If 0 (default), read from the catalog config.
    config_file : str, optional
        Path to a user Python configuration file that defines
        ``update_config(xgb_params, ML_list, ML_caps, ML_balance, ML_options)``
        and ``add_ranking_statistics(events)`` to customise the defaults.
    nfrac : float
        Fraction of background events to use for training (default 1.0).
    sfrac : float
        Fraction of signal events to use for training (default 1.0).
    dump : bool
        Save diagnostic plots and feature-importance figures alongside the model.
    verbose : bool
        Pass ``verbose=True`` to XGBoost's ``fit()`` to print per-round metrics.

    Returns
    -------
    xgb.XGBClassifier
        Trained classifier (also persisted to *model_file*).
    """
    # ------------------------------------------------------------------
    # Resolve nifo from the background catalog if not provided
    # ------------------------------------------------------------------
    if nifo == 0:
        _, nifo, _ = read_catalog_to_dataframe(bkg_catalog_file)
        logger.info("Detected nifo = %d from catalog", nifo)

    # ------------------------------------------------------------------
    # Load default XGBoost configuration
    # ------------------------------------------------------------------
    xgb_params, ML_list, ML_caps, ML_balance, ML_options = xgb_config(search, nifo)
    seed = xgb_params["seed"]

    # ------------------------------------------------------------------
    # Apply user overrides (optional)
    # ------------------------------------------------------------------
    if config_file:
        ML_defcaps = copy.deepcopy(ML_caps)
        update_config = import_function_from_file(config_file, "update_config")
        update_config(xgb_params, ML_list, ML_caps, ML_balance, ML_options)
        update_ML_list(ML_list, ML_defcaps, ML_caps)
        logger.info("Applied user config from %s", config_file)

    print(
        f"\nXGBoost training config:"
        f"\n  search={search}, nifo={nifo}"
        f"\n  ML_list={ML_list}"
        f"\n  ML_caps={ML_caps}"
        f"\n  xgb_params={xgb_params}\n"
    )

    cuts = ML_balance.get(f"cuts(training)", "")

    # ------------------------------------------------------------------
    # Read and pre-process catalogs
    # ------------------------------------------------------------------
    logger.info("Reading background catalog: %s", bkg_catalog_file)
    npd = _load_and_preprocess(
        bkg_catalog_file, 0, nifo, search, ML_caps, ML_options,
        frac=nfrac, seed=seed, cuts=cuts,
    )
    logger.info("Background events after preprocessing: %d", len(npd))

    logger.info("Reading signal catalog: %s", sim_catalog_file)
    spd = _load_and_preprocess(
        sim_catalog_file, 1, nifo, search, ML_caps, ML_options,
        frac=sfrac, seed=seed, cuts=cuts,
    )
    logger.info("Signal events after preprocessing: %d", len(spd))

    # ------------------------------------------------------------------
    # Build the combined training set
    # ------------------------------------------------------------------
    # Identify the capped rho0 column for balancing
    rho0_capvalue = -1
    rho0_capname  = None
    for feature, cap in ML_caps.items():
        if "rho0" in feature:
            rho0_capname  = getcapname(feature, cap)
            rho0_capvalue = cap
            break
    if rho0_capvalue < 0:
        raise ValueError("rho0 cap not found in ML_caps; it is required for training.")

    model_dir  = os.path.dirname(model_file) or "."
    model_stem = os.path.splitext(os.path.basename(model_file))[0]
    ofile_tag  = os.path.join(model_dir, model_stem)

    # ------------------------------------------------------------------
    # Pre-training diagnostic plots (dump=True)
    # ------------------------------------------------------------------
    if dump:
        _dump_config(ofile_tag, xgb_params, ML_list, ML_caps, ML_balance, ML_options)
        import logging as _logging
        _logging.getLogger("matplotlib.font_manager").disabled = True
        try:
            plot_hist_rho(ofile_tag + "_bkg_sim_rho0_hist.png", npd, spd, '0')
        except Exception as _e:
            logger.warning("plot_hist_rho failed: %s", _e)
        if search in ('bbh', 'imbhb'):
            try:
                plot_hist_mchirp(ofile_tag + "_bkg_sim_chirp_hist.png", npd, spd)
            except Exception as _e:
                logger.warning("plot_hist_mchirp failed: %s", _e)
        try:
            plot_hist_freq(ofile_tag + "_bkg_sim_frequency0_hist.png", npd, spd)
        except Exception as _e:
            logger.warning("plot_hist_freq failed: %s", _e)
        if rho0_capname is not None:
            try:
                plot_QaQp(ofile_tag + "_bkg_sim_QaQp_plot.png", npd, spd,
                          rho_name=rho0_capname, rho_thr=rho0_capvalue)
            except Exception as _e:
                logger.warning("plot_QaQp failed: %s", _e)
        try:
            mplot1d(npd, spd, odir=os.path.join(model_dir, "mplot1d"),
                    utag=model_stem, goptions=ML_options)
        except Exception as _e:
            logger.warning("mplot1d failed: %s", _e)
        try:
            mplot2d(npd, spd, odir=os.path.join(model_dir, "mplot2d"),
                    utag=model_stem, goptions=ML_options)
        except Exception as _e:
            logger.warning("mplot2d failed: %s", _e)

    tpd = pd.concat([spd, npd], ignore_index=True)
    ncount = (tpd["classifier"] == 0).sum()
    scount = (tpd["classifier"] == 1).sum()
    print(f"\nMerged training set: SIM={scount}, BKG={ncount}, ratio={scount/ncount:.3f}")

    # Tail balance: equalise SIM and BKG in the high-rho0 tail
    if ML_balance.get("tail(training)", False):
        tpd = get_balanced_tail(tpd, ML_caps, seed)

    # Build full feature list including auxiliary columns needed for weighting
    ML_list_weight = list(ML_list)
    ML_list_weight.append("classifier")
    for extra in ("penalty", "ecor"):
        if extra not in ML_list_weight:
            ML_list_weight.append(extra)
    if ML_caps.get("Qa", -1) >= 0 and "Qa" not in ML_list_weight:
        ML_list_weight.append("Qa")
    if ML_caps.get("Qp", -1) >= 0 and "Qp" not in ML_list_weight:
        ML_list_weight.append("Qp")

    # Keep only columns that actually exist
    ML_list_weight = [c for c in ML_list_weight if c in tpd.columns]

    # ------------------------------------------------------------------
    # Train / eval split, bulk balance, training
    # ------------------------------------------------------------------
    X_train, X_eval, y_train, y_eval = train_test_split(
        tpd[ML_list_weight], tpd[["classifier"]],
        test_size=0.10, random_state=seed,
    )
    print(
        f"\nTrain/eval split: X_train={X_train.shape[0]}, X_eval={X_eval.shape[0]}"
    )

    if ML_balance.get("bulk(training)", False):
        X_train, weight = get_balanced_bulk(
            X_train, ML_caps, ML_balance, "training", dump, ofile_tag
        )
    else:
        X_train["weight1"] = 1.0
        weight = X_train["weight1"]

    # Use only the feature columns for training
    X_train_feat = X_train[ML_list]
    X_eval_feat  = X_eval[ML_list]

    # ------------------------------------------------------------------
    # Fit XGBoost
    # ------------------------------------------------------------------
    # XGBoost 2.x: eval_metric and early_stopping_rounds must be constructor
    # parameters; use_label_encoder was removed in XGBoost 2.x.
    _train_params = dict(xgb_params)
    _train_params.pop("use_label_encoder", None)
    _train_params.setdefault("eval_metric", ["logloss", "auc", "aucpr"])
    _train_params.setdefault("early_stopping_rounds", 50)
    XGB_clf = xgb.XGBClassifier(**_train_params)
    print("\nStart XGBoost training …\n")
    start = time.time()
    XGB_clf.fit(
        X_train_feat, y_train,
        sample_weight=weight,
        eval_set=[(X_eval_feat, y_eval)],
        verbose=verbose,
    )
    elapsed = time.time() - start
    print(f"\nTraining done. Elapsed time: {elapsed:.1f} s")
    print(f"  Best score:      {XGB_clf.best_score:.5f}")
    print(f"  Best iteration:  {XGB_clf.best_iteration}")

    # ------------------------------------------------------------------
    # Log feature importances
    # ------------------------------------------------------------------
    if verbose:
        ml_list_trained = XGB_clf.get_booster().feature_names
        print("\nFeature importances (gain):")
        print(XGB_clf.get_booster().get_score(importance_type="gain"))

    # ------------------------------------------------------------------
    # Persist model and optional plots
    # ------------------------------------------------------------------
    os.makedirs(model_dir, exist_ok=True)
    # Use XGBoost native format when the extension is .ubj or .json;
    # fall back to pickle for .pkl / legacy paths.
    _ext = model_file.rsplit(".", 1)[-1].lower()
    if _ext in ("ubj", "json"):
        XGB_clf.save_model(model_file)
    else:
        with open(model_file, "wb") as fh:
            pickle.dump(XGB_clf, fh)
    print(f"\nModel saved to {model_file}")

    if dump:
        _dump_plots(XGB_clf, ofile_tag, X_eval_feat, y_eval)
        _dump_output(XGB_clf, ofile_tag)

    return XGB_clf


# ---------------------------------------------------------------------------
# Flat-Parquet helpers (complement to train() which uses pycwb Catalog)
# ---------------------------------------------------------------------------

def make_train_test_split(
    events: pd.DataFrame,
    ML_list: list,
    ML_caps: dict,
    seed: int,
    test_size: float = 0.10,
):
    """Build a stratified train/test split from a preprocessed DataFrame.

    Mirrors the logic inside :func:`train` but operates on an already-
    preprocessed DataFrame (e.g. one produced by
    :func:`~.read_data.build_features_from_parquet`).

    Parameters
    ----------
    events : pd.DataFrame
        Preprocessed event DataFrame containing at least the columns in
        *ML_list* plus ``classifier``.
    ML_list : list[str]
        Feature columns to use for training.
    ML_caps : dict
        Cap dict from :func:`~.config.xgb_config` — used to decide which
        auxiliary columns (``Qa``, ``Qp``) to include.
    seed : int
        Random seed (use ``xgb_params["seed"]``).
    test_size : float
        Fraction of events held out for evaluation (default 0.10).

    Returns
    -------
    X_train, X_test : pd.DataFrame
        Feature matrices as DataFrames (column names preserved).
    y_train, y_test : pd.Series
    """
    aux = ["classifier", "penalty", "ecor"]
    if ML_caps.get("Qa", -1) >= 0:
        aux.append("Qa")
    if ML_caps.get("Qp", -1) >= 0:
        aux.append("Qp")
    keep = [c for c in (ML_list + aux) if c in events.columns]

    clean = events[keep].dropna()
    logger.info(
        "Events after NaN drop : %d  (dropped %d)",
        len(clean), len(events) - len(clean),
    )

    X = clean[ML_list]
    y = clean["classifier"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )
    logger.info(
        "Train: %d  (bkg=%d, sim=%d)  |  Test: %d  (bkg=%d, sim=%d)",
        len(X_train), (y_train == 0).sum(), (y_train == 1).sum(),
        len(X_test),  (y_test  == 0).sum(), (y_test  == 1).sum(),
    )
    return X_train, X_test, y_train, y_test


def train_classifier(
    X_train,
    y_train,
    X_test,
    y_test,
    xgb_params: dict,
    verbose: bool = False,
    nthread: int = -1,
    device: Optional[str] = None,
) -> xgb.XGBClassifier:
    """Fit an XGBoost classifier with early stopping.

    Compatible with XGBoost 2.x: removes ``use_label_encoder``, moves
    ``eval_metric`` / ``early_stopping_rounds`` to the constructor, and
    converts DataFrames to numpy to avoid dtype-compatibility issues.

    .. note::
        The default config sets ``nthread=1`` for safety in cluster environments
        where many jobs run concurrently (HTCondor/SLURM).  When training
        locally, pass ``nthread=-1`` (the default here) to use all CPU cores,
        or an explicit count such as ``nthread=8``.

    Parameters
    ----------
    X_train, X_test : array-like or pd.DataFrame
        Feature matrices.  Column names are preserved when DataFrames are
        passed.
    y_train, y_test : array-like
        Binary labels (0 = background, 1 = signal).
    xgb_params : dict
        Hyper-parameters from :func:`~.config.xgb_config` (or overrides).
        ``nthread`` and ``device`` keys are overridden by the explicit
        *nthread* / *device* arguments below.
    verbose : bool
        Print per-round metrics every 100 rounds.
    nthread : int
        Number of CPU threads for XGBoost's parallel histogram building and
        scoring.  ``-1`` (default) resolves to ``os.cpu_count()`` at runtime,
        overriding any ``nthread`` value present in *xgb_params*.  Pass
        ``None`` to keep the value from *xgb_params* unchanged.
    device : str or None
        XGBoost 2.x device string: ``"cpu"`` (default) or ``"cuda"`` / ``"cuda:0"``
        for GPU acceleration.  ``None`` leaves the *xgb_params* value unchanged.
        When ``device`` is set to a CUDA device, ``tree_method`` is forced to
        ``"hist"`` (the only GPU-compatible method in XGBoost 2.x).

    Returns
    -------
    xgb.XGBClassifier
        Trained classifier with ``feature_names`` set.
    """
    import os as _os

    params = dict(xgb_params)
    params.setdefault("eval_metric", ["logloss", "auc", "aucpr"])
    params.setdefault("early_stopping_rounds", 50)
    params.pop("use_label_encoder", None)  # removed in XGBoost 2.x

    # ------------------------------------------------------------------
    # Thread count — the config defaults to nthread=1 for cluster safety;
    # override here to use all available CPUs when training locally.
    # ------------------------------------------------------------------
    if nthread is not None:
        if nthread == -1:
            cpu_count = _os.cpu_count() or 1
            resolved_nthread = min(cpu_count, 8) if cpu_count > 8 else cpu_count
        else:
            resolved_nthread = int(nthread)
        params["nthread"] = resolved_nthread
        logger.info("nthread overridden → %d", resolved_nthread)

    # ------------------------------------------------------------------
    # Device (GPU support, XGBoost 2.x)
    # ------------------------------------------------------------------
    if device is not None:
        params["device"] = device
        if device.startswith("cuda"):
            params["tree_method"] = "hist"  # required for GPU in XGBoost 2.x
        logger.info("device = %s", device)

    feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None

    # Convert to numpy to avoid XGBoost 2.x pandas dtype-compatibility issues
    X_tr = np.asarray(X_train, dtype=np.float32)
    X_te = np.asarray(X_test,  dtype=np.float32)
    y_tr = np.asarray(y_train)
    y_te = np.asarray(y_test)

    logger.info(
        "XGBoost config: tree_method=%s  nthread=%s  device=%s  n_estimators=%s",
        params.get("tree_method"), params.get("nthread"), params.get("device", "cpu"),
        params.get("n_estimators"),
    )
    clf = xgb.XGBClassifier(**params)
    logger.info("Starting XGBoost training …")
    t0 = time.time()
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        verbose=10,
    )
    if feature_names:
        clf.get_booster().feature_names = feature_names
    logger.info("Training done in %.1f s", time.time() - t0)
    logger.info("Best iteration : %d", clf.best_iteration)
    logger.info("Best score     : %.5f", clf.best_score)
    return clf


# ---------------------------------------------------------------------------
# Optional diagnostic plots
# ---------------------------------------------------------------------------

def _dump_plots(XGB_clf, ofile_tag: str, X_eval, y_eval) -> None:
    """Save feature-importance, ROC, and PR plots (requires matplotlib)."""
    import logging as _logging
    _logging.getLogger("matplotlib.font_manager").disabled = True

    # Feature importance
    try:
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        rcParams["text.usetex"] = False
        fig, ax = plt.subplots(figsize=(18, 12))
        xgb.plot_importance(XGB_clf, max_num_features=50, height=0.8, ax=ax)
        plt.tight_layout()
        ofile = ofile_tag + "_importance.png"
        plt.savefig(ofile)
        plt.close()
        print(f"Saved feature importance plot: {ofile}")
    except Exception as exc:
        logger.warning("Could not save feature importance plot: %s", exc)

    # ROC and PR curves — delegate to plots.py for consistent style and correct
    # handling of DataFrame y_eval (single-column) vs 1-D array.
    try:
        plot_roc(ofile_tag + "_roc.png", XGB_clf, X_eval, y_eval)
    except Exception as exc:
        logger.warning("Could not save ROC plot: %s", exc)

    try:
        plot_pr(ofile_tag + "_pr.png", XGB_clf, X_eval, y_eval)
    except Exception as exc:
        logger.warning("Could not save PR plot: %s", exc)


def _dump_config(
    ofile_tag: str,
    xgb_params: dict,
    ML_list: list,
    ML_caps: dict,
    ML_balance: dict,
    ML_options: dict,
) -> None:
    """Write a human-readable ``.cfg`` file with all XGBoost training parameters.

    Parameters
    ----------
    ofile_tag : str
        Base path (no extension); the file ``ofile_tag + ".cfg"`` is created.
    xgb_params : dict
        XGBoost hyper-parameters passed to :class:`xgb.XGBClassifier`.
    ML_list : list[str]
        Feature list used for training.
    ML_caps : dict
        Feature cap dictionary.
    ML_balance : dict
        Balance / cuts configuration.
    ML_options : dict
        Per-feature options (plot ranges, enable flags …).
    """
    ofile = ofile_tag + ".cfg"
    lines = [
        "# XGBoost training configuration",
        "# Generated by pycwb.modules.cwb_xgboost.training",
        "",
        "[xgb_params]",
    ]
    for k, v in sorted(xgb_params.items()):
        lines.append(f"  {k} = {v!r}")
    lines += ["", "[ML_list]"]
    for item in ML_list:
        lines.append(f"  {item!r}")
    lines += ["", "[ML_caps]"]
    for k, v in sorted(ML_caps.items()):
        lines.append(f"  {k!r} = {v!r}")
    lines += ["", "[ML_balance]"]
    for k, v in sorted(ML_balance.items()):
        lines.append(f"  {k!r} = {v!r}")
    lines += ["", "[ML_options]"]
    for k, v in sorted(ML_options.items()):
        lines.append(f"  {k!r} = {v!r}")
    lines.append("")

    try:
        with open(ofile, "w") as fh:
            fh.write("\n".join(lines))
        print(f"Config saved to {ofile}")
    except Exception as exc:
        logger.warning("Could not write config file %s: %s", ofile, exc)


def _dump_output(XGB_clf: "xgb.XGBClassifier", ofile_tag: str) -> None:
    """Write a ``.out`` file with tree statistics and all five feature-importance types.

    The five importance types reported are the standard XGBoost types:

    * **weight**      – number of times a feature is used to split across all trees.
    * **gain**        – average training-loss reduction gained when a feature is used.
    * **cover**       – average number of samples affected by splits on that feature.
    * **total_gain**  – total (summed) gain across all splits on that feature.
    * **total_cover** – total (summed) cover across all splits on that feature.

    Parameters
    ----------
    XGB_clf : xgb.XGBClassifier
        Trained classifier.
    ofile_tag : str
        Base path (no extension); the file ``ofile_tag + ".out"`` is created.
    """
    ofile = ofile_tag + ".out"
    booster = XGB_clf.get_booster()

    # --- tree statistics ---------------------------------------------------
    try:
        n_trees = booster.num_boosted_rounds()
    except Exception:
        # best_iteration is 0-based; add 1 for count
        n_trees = getattr(XGB_clf, "best_iteration", 0) + 1

    lines = [
        "# XGBoost training output",
        "# Generated by pycwb.modules.cwb_xgboost.training",
        "",
        "[tree_statistics]",
        f"  num_trees          = {n_trees}",
        f"  best_iteration     = {getattr(XGB_clf, 'best_iteration', 'N/A')}",
        f"  best_score         = {getattr(XGB_clf, 'best_score', 'N/A')}",
        f"  feature_names      = {booster.feature_names}",
        "",
    ]

    # --- feature importances (all five types) ------------------------------
    importance_types = ["weight", "gain", "cover", "total_gain", "total_cover"]
    for itype in importance_types:
        try:
            scores = booster.get_score(importance_type=itype)
        except Exception as exc:
            lines.append(f"[feature_importance_{itype}]")
            lines.append(f"  # could not compute: {exc}")
            lines.append("")
            continue

        # Sort descending by score value
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        lines.append(f"[feature_importance_{itype}]")
        for feat, val in sorted_scores:
            lines.append(f"  {feat:<30s} = {val:.6g}")
        lines.append("")

    try:
        with open(ofile, "w") as fh:
            fh.write("\n".join(lines))
        print(f"Output saved to {ofile}")
    except Exception as exc:
        logger.warning("Could not write output file %s: %s", ofile, exc)
