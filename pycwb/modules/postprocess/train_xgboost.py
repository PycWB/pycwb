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
dump_training_review : bool, default False
    Save human-readable training settings and training output files for review.
training_settings_file : str, optional
    Destination for the settings dump. Defaults to ``<model stem>.cfg`` when
    ``dump`` or ``dump_training_review`` is enabled.
training_output_file : str, optional
    Destination for the output dump. Defaults to ``<model stem>.out`` when
    ``dump`` or ``dump_training_review`` is enabled.
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
    outputs=['model_file', 'training_settings_file', 'training_output_file'],
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
    dump_training_review: bool = False,
    training_settings_file: Optional[str] = None,
    training_output_file: Optional[str] = None,
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
    settings_path = _resolve(training_settings_file) if training_settings_file else None
    output_path = _resolve(training_output_file) if training_output_file else None
    dump_review = bool(dump or dump_training_review or settings_path or output_path)

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
    from pycwb.modules.cwb_xgboost.training import (
        dump_training_settings,
        dump_training_output,
    )
    from pycwb.utils.module import import_function_from_file

    # ── load config before reading so Parquet can project only needed columns ──
    if nifo == 0:
        nifo = _detect_nifo_from_schema(bkg_paths + sim_paths)
        logger.info("Auto-detected nifo = %d from parquet schema", nifo)

    xgb_params, ML_list, ML_caps, ML_balance, ML_options = xgb_config(search, nifo)
    seed = xgb_params["seed"]

    if config_path:
        ML_defcaps = copy.deepcopy(ML_caps)
        update_config_fn = import_function_from_file(config_path, "update_config")
        update_config_fn(xgb_params, ML_list, ML_caps, ML_balance, ML_options)
        update_ML_list(ML_list, ML_defcaps, ML_caps)
        logger.info("Applied user config from %s", config_path)

    read_columns = _xgb_required_input_columns(
        bkg_paths + sim_paths,
        nifo=nifo,
        ML_options=ML_options,
        ML_caps=ML_caps,
        ML_balance=ML_balance,
        ML_list=ML_list,
        include_training_metadata=True,
    )

    # ── read filtered parquets ───────────────────────────────────────────
    bdf = _read_and_concat(bkg_paths, "BKG", columns=read_columns)
    sdf = _read_and_concat(sim_paths, "SIM", columns=read_columns)
    bkg_rows_input = len(bdf)
    sim_rows_input = len(sdf)
    if "lag_idx" in bdf.columns:
        n_before = len(bdf)
        bdf = bdf[nonzero_lag_mask(bdf)].reset_index(drop=True)
        logger.info("Filtered BKG lag 0 for training: %d -> %d rows", n_before, len(bdf))
    bkg_rows_after_lag_filter = len(bdf)
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
    sim_rows_after_cleaning = len(sdf)

    # ── map Catalog column names ─────────────────────────────────────────
    bdf = _map_catalog_columns(bdf, nifo)
    sdf = _map_catalog_columns(sdf, nifo)

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
    bkg_rows_after_preprocess = len(bdf)
    sim_rows_after_preprocess = len(sdf)

    training_columns = _training_working_columns(ML_list, ML_caps, bdf, sdf)
    bdf = _compact_training_frame(bdf, training_columns)
    sdf = _compact_training_frame(sdf, training_columns)

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

    settings_dump_path = None
    output_dump_path = None
    if dump_review:
        settings_path = settings_path or (ofile_tag + ".cfg")
        settings_dump_path = dump_training_settings(
            ofile_tag,
            xgb_params,
            ML_list,
            ML_caps,
            ML_balance,
            ML_options,
            output_file=settings_path,
            metadata={
                "search": search,
                "nifo": nifo,
                "model_file": model_path,
                "config_file": config_path,
                "bkg_catalogs": bkg_paths,
                "sim_catalogs": sim_paths,
                "bkg_rows_input": bkg_rows_input,
                "sim_rows_input": sim_rows_input,
                "bkg_rows_after_lag_filter": bkg_rows_after_lag_filter,
                "sim_rows_after_cleaning": sim_rows_after_cleaning,
                "bkg_rows_after_preprocess": bkg_rows_after_preprocess,
                "sim_rows_after_preprocess": sim_rows_after_preprocess,
                "train_rows": int(len(X_train_feat)),
                "eval_rows": int(len(X_eval_feat)),
                "dump": dump,
                "dump_training_review": dump_training_review,
            },
        )

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
    if dump_review:
        output_path = output_path or (ofile_tag + ".out")
        output_dump_path = dump_training_output(
            XGB_clf,
            ofile_tag,
            output_file=output_path,
            training_summary={
                "elapsed_seconds": float(elapsed),
                "model_file": model_path,
                "model_size_kb": float(size_kb),
                "auc": auc,
                "best_iteration": getattr(XGB_clf, "best_iteration", None),
                "best_score": getattr(XGB_clf, "best_score", None),
                "train_rows": int(len(X_train_feat)),
                "eval_rows": int(len(X_eval_feat)),
            },
        )

    result = {"model_file": model_path, "auc": auc}
    if settings_dump_path:
        result["training_settings_file"] = settings_dump_path
    if output_dump_path:
        result["training_output_file"] = output_dump_path
    return result


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


def _read_and_concat(paths: list[str], label: str, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Read one or more parquet catalogs and concatenate them."""
    frames = []
    for path in paths:
        read_columns = _existing_columns(path, columns) if columns is not None else None
        df = pd.read_parquet(path, columns=read_columns)
        logger.info(
            "  %s rows from %s: %d%s",
            label,
            path,
            len(df),
            "" if read_columns is None else f" ({len(read_columns)} projected columns)",
        )
        frames.append(df)
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    return pd.concat(frames, ignore_index=True)


def _detect_nifo_from_schema(paths: list[str]) -> int:
    columns = _schema_columns(paths)
    rho_cols = [c for c in columns if c.startswith("rho") and c[3:].isdigit()]
    if rho_cols:
        return max(int(c[3:]) for c in rho_cols) + 1
    suffixes = [suffix for suffix in _IFO_SUFFIX_TO_INDEX if any(c.endswith(f"_{suffix}") for c in columns)]
    if suffixes:
        return max(_IFO_SUFFIX_TO_INDEX[suffix] for suffix in suffixes) + 1
    return 2


def _xgb_required_input_columns(
    paths: list[str],
    nifo: int,
    ML_options: dict,
    ML_caps: dict,
    ML_balance: dict,
    ML_list: list[str],
    include_training_metadata: bool = False,
    extra_columns: Optional[list[str]] = None,
) -> list[str]:
    """Return raw Parquet columns needed to build configured XGB features."""
    available = _schema_columns(paths)
    keep: set[str] = set()

    def add_existing(*names: str) -> None:
        keep.update(name for name in names if name in available)

    def add_prefixes(*prefixes: str) -> None:
        keep.update(col for col in available if any(col == prefix or col.startswith(prefix) for prefix in prefixes))

    add_existing("id", "job_id", "lag_idx", "lag", "trial_idx", "gps_time", "ifar")
    add_prefixes("time_lag_", "segment_lag_", "segment_shift_", "shift_")

    xvars = set(ML_options.get("readfile(vars)", []))
    feature_text = " ".join(list(ML_list) + list(ML_caps) + [str(ML_balance.get("cuts(training)", "")), str(ML_options.get("cuts(prediction)", ""))])

    if "rho" in xvars or "rho0" in feature_text:
        add_existing("rho", "rho0", "rho0_std")
        add_prefixes("rho")

    if "norm" in xvars or "norm" in feature_text:
        add_existing("norm", "coherent_energy_norm")

    if "netcc" in xvars or "netcc" in feature_text:
        add_existing("net_cc", "netcc", "netcc0", "netcc1")

    if "Qveto" in xvars or "Qa" in feature_text or "Qp" in feature_text:
        add_existing("q_veto", "q_factor", "qveto", "qfactor", "Qveto", "Qveto0", "Qveto1")
        add_prefixes("Qveto")

    if "Lveto" in feature_text:
        add_existing("Lveto", "Lveto2")
        add_prefixes("Lveto")

    if "ecor" in xvars or "ecor" in feature_text or "rho0" in feature_text or "Qp" in feature_text:
        add_existing("coherent_energy", "ecor")

    if "penalty" in xvars or "penalty" in feature_text or "rho0" in feature_text:
        add_existing("penalty")

    if "likelihood" in xvars or "likelihood" in feature_text:
        add_existing("likelihood")

    if "sSNR" in xvars or "sSNR" in feature_text or "mSNR" in feature_text:
        add_existing("sSNR")
        add_prefixes("sSNR")
        for ifo in _IFO_SUFFIX_TO_INDEX:
            add_existing(f"signal_energy_{ifo}", f"data_energy_{ifo}")
        for idx in range(nifo):
            add_existing(f"signal_energy{idx}", f"data_energy{idx}")

    if "noise" in xvars or "noise" in feature_text:
        add_existing("noise")
        add_prefixes("noise")
        for ifo in _IFO_SUFFIX_TO_INDEX:
            add_existing(f"noise_rms_{ifo}")
        for idx in range(nifo):
            add_existing(f"noise{idx}")

    if "duration" in xvars or "duration" in feature_text:
        add_existing("duration")
        add_prefixes("duration")
    if "bandwidth" in xvars or "bandwidth" in feature_text:
        add_existing("bandwidth")
        add_prefixes("bandwidth")
    if "frequency" in xvars or "frequency" in feature_text:
        add_existing("frequency")
        add_prefixes("frequency", "central_freq")
    if "chirp" in xvars or "chirp" in feature_text:
        add_existing("chirp")
        add_prefixes("chirp")

    if include_training_metadata:
        add_existing("sim_sim_idx", "sim_vetoed_cat0", "sim_vetoed_cat1", "sim_vetoed_cat2", "sim_across_segments")
    if extra_columns:
        add_existing(*extra_columns)

    return [col for col in sorted(keep) if col in available]


def _schema_columns(paths: list[str]) -> set[str]:
    import pyarrow.parquet as pq

    columns: set[str] = set()
    for path in paths:
        columns.update(pq.read_schema(path).names)
    return columns


def _existing_columns(path: str, columns: Optional[list[str]]) -> Optional[list[str]]:
    if columns is None:
        return None
    import pyarrow.parquet as pq

    available = set(pq.read_schema(path).names)
    return [col for col in columns if col in available]


def _training_working_columns(
    ML_list: list[str],
    ML_caps: dict,
    bdf: pd.DataFrame,
    sdf: pd.DataFrame,
) -> list[str]:
    columns = list(ML_list) + ["classifier", "penalty", "ecor"]
    if ML_caps.get("Qa", -1) >= 0:
        columns.append("Qa")
    if ML_caps.get("Qp", -1) >= 0:
        columns.append("Qp")
    available = set(bdf.columns) | set(sdf.columns)
    return list(dict.fromkeys(col for col in columns if col in available))


def _compact_training_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    compact = df[[col for col in columns if col in df.columns]].copy()
    if "classifier" in compact.columns:
        compact["classifier"] = compact["classifier"].astype("int8", copy=False)
    for col in compact.columns:
        if col == "classifier":
            continue
        if pd.api.types.is_numeric_dtype(compact[col]):
            compact[col] = pd.to_numeric(compact[col], errors="coerce").astype("float32", copy=False)
    return compact


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
