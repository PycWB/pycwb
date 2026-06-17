"""XGBoost training step — workflow-compatible wrapper.

Reads pre-filtered plain parquet catalogs (produced by
:func:`~.random_filter.random_filter_parquet`) and runs the full XGBoost
training pipeline, reusing internal functions from
:mod:`pycwb.modules.cwb_xgboost`.

Workflow action
---------------
``postprocess.train_xgboost.train_xgboost``

Parameters (via YAML ``args``)
------------------------------
work_dir : str
    Base directory; relative paths are resolved against this.
bkg_catalog : str
    Path to the background parquet catalog (plain parquet).  Deprecated in
    favor of ``bkg_catalogs`` for new workflows.
sim_catalog : str
    Path to the signal/injection parquet catalog (plain parquet).  Deprecated
    in favor of ``sim_catalogs`` for new workflows.
bkg_catalogs : list[str], optional
    Background catalogs to concatenate for training.
sim_catalogs : list[str], optional
    Simulation catalogs to concatenate for training.
model_file : str
    Destination for the trained model (``.ubj`` recommended).
search : str
    Search type: ``blf``, ``bhf``, ``bld``, ``bbh``, ``imbhb``.
nifo : int, default 0
    Number of interferometers (0 = auto-detect from columns).
config_file : str, optional
    Path to a Python file defining ``update_config(...)``.
dump : bool, default False
    Save diagnostic plots alongside the model.
verbose : bool, default False
    Print per-round XGBoost training metrics.

Returns
-------
dict
    ``{"model_file": str, "auc": float}``
"""

from __future__ import annotations

import copy
import logging
import os
import pickle
import time
from typing import Optional

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from pycwb.post_production.action_spec import action_spec
from pycwb.modules.postprocess.lag_filters import nonzero_lag_mask

logger = logging.getLogger(__name__)


@action_spec(
    outputs=['model_file'],
    inputs=['bkg_catalog', 'sim_catalog', 'bkg_catalogs', 'sim_catalogs', 'config_file'],
    description='Train XGBoost classifier from BKG + SIM catalogs',
    help=(
        "New workflows should pass bkg_catalogs and sim_catalogs lists. "
        "Keep SIM cleaning in an explicit upstream filter_real_simulation "
        "action; matched_outer_file remains supported for old workflows."
    ),
)
def train_xgboost(
    work_dir: str,
    bkg_catalog: Optional[str] = None,
    sim_catalog: Optional[str] = None,
    model_file: str = "models/xgb_model.ubj",
    bkg_catalogs: Optional[list[str]] = None,
    sim_catalogs: Optional[list[str]] = None,
    search: str = "blf",
    nifo: int = 0,
    config_file: Optional[str] = None,
    dump: bool = False,
    verbose: bool = False,
    **kwargs,
) -> dict:
    """Train an XGBoost classifier from pre-filtered plain Parquet catalogs."""
    # ── resolve paths ────────────────────────────────────────────────────
    def _resolve(relpath: str) -> str:
        if os.path.isabs(relpath):
            return relpath
        return os.path.join(work_dir, relpath)

    bkg_files = _catalog_files(bkg_catalogs, bkg_catalog, "bkg_catalogs")
    sim_files = _catalog_files(sim_catalogs, sim_catalog, "sim_catalogs")
    bkg_paths = [_resolve(path) for path in bkg_files]
    sim_paths = [_resolve(path) for path in sim_files]
    model_path = _resolve(model_file)
    config_path = _resolve(config_file) if config_file else None

    logger.info("Training XGBoost  search=%s  nifo=%d", search, nifo)
    logger.info("  BKG : %s", ", ".join(bkg_paths))
    logger.info("  SIM : %s", ", ".join(sim_paths))
    logger.info("  model : %s", model_path)

    # ── imports (lazy, avoid circular issues) ────────────────────────────
    from pycwb.modules.cwb_xgboost.config import xgb_config
    from pycwb.modules.cwb_xgboost.read_data import (
        preprocess_events,
        apply_training_cuts,
    )
    from pycwb.modules.cwb_xgboost.utils import getcapname
    from pycwb.modules.cwb_xgboost.utils_extended import (
        get_balanced_tail,
        get_balanced_bulk,
        update_ML_list,
    )
    from pycwb.utils.module import import_function_from_file

    # ── read filtered parquets ───────────────────────────────────────────
    bdf = _read_and_concat(bkg_paths, "BKG")
    sdf = _read_and_concat(sim_paths, "SIM")
    if "lag_idx" in bdf.columns:
        n_before = len(bdf)
        bdf = bdf[nonzero_lag_mask(bdf)].reset_index(drop=True)
        logger.info("Filtered BKG lag 0 for training: %d -> %d rows", n_before, len(bdf))
    bdf["classifier"] = 0
    sdf["classifier"] = 1
    logger.info("BKG rows: %d  |  SIM rows: %d", len(bdf), len(sdf))

    # ── filter SIM to clean matched injections only ──────────────────────
    # First, if matched_outer_file provided, keep only triggers with a
    # matching simulation (sim_sim_idx.notna()).  This removes noise
    # triggers from the SIM catalog.
    matched_outer_file = kwargs.get("matched_outer_file")
    if matched_outer_file:
        outer_path = _resolve(matched_outer_file)
        if os.path.exists(outer_path):
            outer_df = pd.read_parquet(outer_path)
            # Build a set of trigger IDs that are matched to a simulation
            matched_ids = set(outer_df.loc[outer_df["sim_sim_idx"].notna(), "id"].dropna().values)
            n_before = len(sdf)
            sdf = sdf[sdf["id"].isin(matched_ids)].reset_index(drop=True)
            logger.info(
                "  Filtered SIM via matched_outer: %d → %d rows (removed %d noise triggers)",
                n_before, len(sdf), n_before - len(sdf),
            )
        else:
            logger.warning("matched_outer_file not found: %s", outer_path)
    # Then remove vetoed/across-segment rows
    sdf = _filter_clean_matches(sdf)

    # ── auto-detect nifo & map Catalog column names ──────────────────────
    if nifo == 0:
        rho_cols = [c for c in bdf.columns if c.startswith("rho") and c[3:].isdigit()]
        nifo = len(rho_cols) or 2
        logger.info("Auto-detected nifo = %d from columns", nifo)
    bdf = _map_catalog_columns(bdf, nifo)
    sdf = _map_catalog_columns(sdf, nifo)

    # ── load config ──────────────────────────────────────────────────────
    xgb_params, ML_list, ML_caps, ML_balance, ML_options = xgb_config(search, nifo)
    seed = xgb_params["seed"]

    if config_path:
        ML_defcaps = copy.deepcopy(ML_caps)
        update_config_fn = import_function_from_file(config_path, "update_config")
        update_config_fn(xgb_params, ML_list, ML_caps, ML_balance, ML_options)
        update_ML_list(ML_list, ML_defcaps, ML_caps)
        logger.info("Applied user config from %s", config_path)

    print(
        f"\nXGBoost training config:"
        f"\n  search={search}, nifo={nifo}"
        f"\n  ML_list={ML_list}"
        f"\n  ML_caps={ML_caps}"
        f"\n  xgb_params={xgb_params}\n"
    )

    cuts = ML_balance.get("cuts(training)", "")

    # ── preprocess ───────────────────────────────────────────────────────
    bdf = preprocess_events(bdf, nifo, ML_options, ML_caps)
    sdf = preprocess_events(sdf, nifo, ML_options, ML_caps)
    if cuts:
        bdf = apply_training_cuts(bdf, cuts)
        sdf = apply_training_cuts(sdf, cuts)
    logger.info("After preprocess: BKG=%d  SIM=%d", len(bdf), len(sdf))

    # ── merge & balance ──────────────────────────────────────────────────
    tpd = pd.concat([sdf, bdf], ignore_index=True)
    ncount = (tpd["classifier"] == 0).sum()
    scount = (tpd["classifier"] == 1).sum()
    print(f"\nMerged training set: SIM={scount}, BKG={ncount}, ratio={scount/max(ncount, 1):.3f}")

    if ML_balance.get("tail(training)", False):
        tpd = get_balanced_tail(tpd, ML_caps, seed)

    # Build full feature list including auxiliary columns
    ML_list_weight = list(ML_list) + ["classifier"]
    for extra in ("penalty", "ecor"):
        if extra not in ML_list_weight:
            ML_list_weight.append(extra)
    if ML_caps.get("Qa", -1) >= 0 and "Qa" not in ML_list_weight:
        ML_list_weight.append("Qa")
    if ML_caps.get("Qp", -1) >= 0 and "Qp" not in ML_list_weight:
        ML_list_weight.append("Qp")
    ML_list_weight = [c for c in ML_list_weight if c in tpd.columns]

    # ── split ────────────────────────────────────────────────────────────
    X_all = tpd[ML_list_weight]
    y_all = tpd[["classifier"]]
    X_train, X_eval, y_train, y_eval = train_test_split(
        X_all, y_all, test_size=0.10, random_state=seed,
    )
    print(f"Train/eval split: X_train={X_train.shape[0]}, X_eval={X_eval.shape[0]}")

    # ── bulk balance ─────────────────────────────────────────────────────
    model_dir = os.path.dirname(model_path) or "."
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    ofile_tag = os.path.join(model_dir, model_stem)

    if ML_balance.get("bulk(training)", False):
        X_train, weight = get_balanced_bulk(
            X_train, ML_caps, ML_balance, "training", dump, ofile_tag
        )
    else:
        X_train["weight1"] = 1.0
        weight = X_train["weight1"]

    X_train_feat = X_train[ML_list]
    X_eval_feat = X_eval[ML_list]

    # ── fit XGBoost ──────────────────────────────────────────────────────
    _train_params = dict(xgb_params)
    _train_params.pop("use_label_encoder", None)
    _train_params.setdefault("eval_metric", ["logloss", "auc", "aucpr"])
    _train_params.setdefault("early_stopping_rounds", 50)
    XGB_clf = xgb.XGBClassifier(**_train_params)

    print("\nStart XGBoost training ...\n")
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

    # ── save model ───────────────────────────────────────────────────────
    os.makedirs(model_dir, exist_ok=True)
    _ext = model_path.rsplit(".", 1)[-1].lower()
    if _ext in ("ubj", "json"):
        XGB_clf.save_model(model_path)
    else:
        with open(model_path, "wb") as fh:
            pickle.dump(XGB_clf, fh)
    size_kb = os.path.getsize(model_path) / 1024
    print(f"Model saved -> {model_path}  ({size_kb:.1f} KB)")

    auc = float(getattr(XGB_clf, "best_score", 0.0))
    return {"model_file": model_path, "auc": auc}


def _catalog_files(
    catalogs: Optional[list[str]],
    legacy_catalog: Optional[str],
    name: str,
) -> list[str]:
    """Return normalized catalog paths from list or legacy single argument."""
    files: list[str] = []
    if catalogs:
        files.extend(str(path) for path in catalogs)
    if legacy_catalog:
        files.append(str(legacy_catalog))
    if not files:
        raise ValueError(f"{name} or legacy single-catalog argument is required")
    return files


def _read_and_concat(paths: list[str], label: str) -> pd.DataFrame:
    """Read one or more parquet catalogs and concatenate them."""
    frames = []
    for path in paths:
        df = pd.read_parquet(path)
        logger.info("  %s rows from %s: %d", label, path, len(df))
        frames.append(df)
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Column-name mapping: Catalog schema → preprocess_events expected names
# ---------------------------------------------------------------------------

# Mapping from Catalog column names to what preprocess_events expects.
# Catalog stores per-IFO scalars like duration_H1, duration_L1;
# preprocess_events expects flat scalars like duration0, duration1.
_CATALOG_RENAME_MAP = {
    "coherent_energy": "ecor",
    "q_veto": "qveto",
    "q_factor": "qfactor",
}

# Per-IFO columns: Catalog suffix → preprocess index
# e.g. duration_H1 → duration0, duration_L1 → duration1
_IFO_SUFFIX_TO_INDEX = {
    "H1": 0, "H2": 0, "L1": 1, "V1": 2, "K1": 3,
    "G1": 4, "I1": 5, "T1": 5,
}

# Columns that exist as per-IFO scalars in Catalog (need flattening to N-indexed names)
# Maps Catalog base name → preprocess_events expected base name
_PER_IFO_COLS_MAP = {
    "duration": "duration",
    "bandwidth": "bandwidth",
    "central_freq": "frequency",
    "frequency": "frequency",
    "noise_rms": "noise",
    "noise": "noise",
    "hrss": "hrss",
    "data_energy": "data_energy",
    "signal_energy": "signal_energy",
    "cross_energy": "cross_energy",
    "null_energy": "null_energy",
    "residual_energy": "residual_energy",
    "time_lag": "time_lag",
    "segment_lag": "segment_lag",
    "sample_rate": "sample_rate",
}


def _map_catalog_columns(df: pd.DataFrame, nifo: int) -> pd.DataFrame:
    """Rename Catalog column names to match preprocess_events expectations.

    - ``coherent_energy`` → ``ecor``
    - ``q_veto`` → ``qveto``, ``q_factor`` → ``qfactor``
    - Per-IFO scalars: ``duration_H1`` → ``duration0``, etc.
    - ``net_cc`` is already a scalar (not per-IFO) — keep as-is.
    - ``rho`` is already a scalar — keep as-is.
    """
    df = df.copy()

    # Simple renames
    df.rename(columns=_CATALOG_RENAME_MAP, inplace=True)

    # Per-IFO column flattening: duration_H1 → duration0, bandwidth_L1 → bandwidth1, etc.
    # Also applies base-name remapping: noise_rms_H1 → noise0, central_freq_H1 → frequency0
    for col in list(df.columns):
        for suffix, idx in _IFO_SUFFIX_TO_INDEX.items():
            if col.endswith(f"_{suffix}"):
                base = col[:-(len(suffix) + 1)]  # strip _H1 → get "noise_rms"
                # Look up the target base name
                target_base = _PER_IFO_COLS_MAP.get(base, base)
                new_name = f"{target_base}{idx}"  # noise0
                if new_name not in df.columns:
                    df.rename(columns={col: new_name}, inplace=True)
                break  # only match first suffix

    # sSNR may need to be derived from per-IFO signal/noise
    # If no sSNR0/sSNR1 but we have signal_energy and data_energy, create sSNR
    if "sSNR0" not in df.columns and "sSNR1" not in df.columns:
        for i in range(nifo):
            se_col = f"signal_energy{i}"
            de_col = f"data_energy{i}"
            if se_col in df.columns and de_col in df.columns:
                df[f"sSNR{i}"] = df[se_col] / df[de_col].replace(0, 1.0)

    # netcc: if net_cc exists but netcc doesn't, rename
    if "netcc" not in df.columns and "netcc0" not in df.columns:
        if "net_cc" in df.columns:
            df["netcc0"] = df["net_cc"]
        # also create netcc1 as copy if only one net_cc
        if "netcc1" not in df.columns and "netcc0" in df.columns:
            df["netcc1"] = df["netcc0"]

    # norm: derive from coherent_energy_norm if not present
    if "norm" not in df.columns and "coherent_energy_norm" in df.columns:
        df["norm"] = df["coherent_energy_norm"]

    logger.info("  Column mapping: %d columns after rename", len(df.columns))
    return df


def _filter_clean_matches(sdf: pd.DataFrame) -> pd.DataFrame:
    """Filter SIM DataFrame to keep only clean simulation matches.

    Removes rows flagged as vetoed (``sim_vetoed_cat0``, ``sim_vetoed_cat2``)
    or crossing segment boundaries (``sim_across_segments``).  If none of
    these columns exist, returns the DataFrame unchanged (assumes it's not
    a matched_right file).
    """
    match_cols = ["sim_vetoed_cat0", "sim_vetoed_cat2", "sim_across_segments"]
    has_match_cols = any(c in sdf.columns for c in match_cols)
    if not has_match_cols:
        return sdf  # not a matched_right file, nothing to filter

    n_before = len(sdf)
    mask = pd.Series(True, index=sdf.index)
    for col in match_cols:
        if col in sdf.columns:
            mask = mask & (~sdf[col].fillna(False).astype(bool))

    sdf = sdf[mask].reset_index(drop=True)
    n_removed = n_before - len(sdf)
    if n_removed > 0:
        logger.info(
            "  Filtered SIM matched: removed %d vetoed/across-segment rows, %d remaining",
            n_removed, len(sdf),
        )
    return sdf
