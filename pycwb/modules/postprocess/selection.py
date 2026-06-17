"""Selection actions for post-production workflows.

These actions make the data split stage explicit in the workflow:

``trigger_selection``
    Select or split jobs, materialize matching trigger catalogs, and return
    per-partition live-time summaries.

``filter_real_simulation``
    Keep only recovered, real simulation triggers from a matched SIM table
    before they are used for training.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from pycwb.post_production.action_spec import action_spec
from pycwb.modules.postprocess.lag_filters import nonzero_lag_mask, try_unshifted_job_ids_from_catalog

logger = logging.getLogger(__name__)


@action_spec(
    outputs=["outputs"],
    inputs=["catalog_file", "progress_file"],
    display_name="Select triggers",
    description="Select or split jobs and triggers with matching live time",
    help=(
        "Use selection mode to produce one selected subset. Use split mode to "
        "produce disjoint named partitions, for example train and far BKG "
        "sets. The result stores jobs/triggers as files and returns small "
        "metadata such as live time and counts in the workflow context."
    ),
    args_schema={
        "returns": "List containing any of: jobs, triggers, livetime.",
        "selection": "Single subset options, usually fraction and seed.",
        "split": "Named split options with fractions, for example train/far.",
        "outputs": "Output paths, either flat for selection or nested by split name.",
    },
)
def trigger_selection(
    work_dir: str,
    catalog_file: str,
    progress_file: str,
    returns: Optional[list[str]] = None,
    selection: Optional[dict[str, Any]] = None,
    split: Optional[dict[str, Any]] = None,
    outputs: Optional[dict[str, Any]] = None,
    fraction: float = 1.0,
    exclude_zero_lag: bool = True,
    seed: int = 150914,
    job_filter: Optional[dict[str, Any]] = None,
    trigger_filter: Optional[dict[str, Any]] = None,
    **kwargs,
) -> dict:
    """Select or split triggers by whole jobs.

    Parameters
    ----------
    work_dir : str
        Base directory for relative paths.
    catalog_file : str
        Trigger catalog parquet.
    progress_file : str
        Progress parquet containing ``job_id`` and ``livetime``.
    returns : list[str], optional
        Values to return/materialize: ``jobs``, ``triggers``, ``livetime``.
        Defaults to all three.
    selection : dict, optional
        Single-subset options.  Supports ``fraction``, ``seed`` and
        ``exclude_zero_lag``.
    split : dict, optional
        Disjoint split options.  Supports ``fractions`` (mapping or list),
        ``seed`` and ``exclude_zero_lag``.
    outputs : dict, optional
        Output paths.  For split mode this is nested by partition name.

    Returns
    -------
    dict
        Structured result keyed by partition name for split mode, with
        ``livetime`` summaries nested under each partition.
    """
    returns = list(returns or ["jobs", "triggers", "livetime"])
    outputs = outputs or {}

    unshifted_job_ids = try_unshifted_job_ids_from_catalog(_resolve(work_dir, catalog_file))

    progress = pd.read_parquet(_resolve(work_dir, progress_file))
    progress = _filter_progress(
        progress,
        exclude_zero_lag=exclude_zero_lag,
        job_filter=job_filter,
        unshifted_job_ids=unshifted_job_ids,
    )
    job_lt = progress.groupby("job_id")["livetime"].sum()
    if job_lt.empty:
        raise ValueError("No jobs remain after progress filtering")

    if split:
        split_exclude_zero_lag = split.get("exclude_zero_lag", exclude_zero_lag)
        if split_exclude_zero_lag != exclude_zero_lag:
            progress = pd.read_parquet(_resolve(work_dir, progress_file))
            progress = _filter_progress(
                progress,
                exclude_zero_lag=split_exclude_zero_lag,
                job_filter=job_filter,
                unshifted_job_ids=unshifted_job_ids,
            )
            job_lt = progress.groupby("job_id")["livetime"].sum()
        split_seed = int(split.get("seed", seed))
        fractions = split.get("fractions")
        if fractions is None:
            raise ValueError("split.fractions is required in split mode")

        split_by = str(split.get("by", "livetime"))
        if split_by in {"interval", "interval_livetime"}:
            progress = _add_interval_columns(
                progress,
                _shift_by_job_from_catalog(work_dir, catalog_file),
            )
            interval_lt = progress.groupby("_interval_key")["livetime"].sum()
            if interval_lt.empty:
                raise ValueError("No intervals remain after progress filtering")
            partitions = _split_intervals_by_livetime(interval_lt, fractions, split_seed)
            total_summary = _livetime_summary(interval_lt, n_triggers=None)
            total_summary["n_intervals"] = int(len(interval_lt))
            total_summary["n_jobs"] = int(progress["job_id"].nunique())
            result = {
                "mode": "split",
                "split_by": split_by,
                "total": total_summary,
            }
            catalog = None
            if "triggers" in returns or _outputs_request_triggers(outputs):
                catalog = _read_catalog(work_dir, catalog_file, trigger_filter)
            for name, interval_keys in partitions.items():
                part_outputs = outputs.get(name, {}) if isinstance(outputs.get(name), dict) else {}
                result[name] = _materialize_interval_partition(
                    work_dir=work_dir,
                    name=name,
                    interval_keys=interval_keys,
                    interval_lt=interval_lt,
                    progress=progress,
                    catalog=catalog,
                    outputs=part_outputs,
                    returns=returns,
                    exclude_zero_lag=split_exclude_zero_lag,
                    unshifted_job_ids=unshifted_job_ids,
                )
            return result

        partitions = _split_jobs_by_livetime(job_lt, fractions, split_seed)
        result = {
            "mode": "split",
            "split_by": split_by,
            "total": _livetime_summary(job_lt, n_triggers=None),
        }
        catalog = None
        if "triggers" in returns or _outputs_request_triggers(outputs):
            catalog = _read_catalog(work_dir, catalog_file, trigger_filter)
        for name, job_ids in partitions.items():
            part_outputs = outputs.get(name, {}) if isinstance(outputs.get(name), dict) else {}
            result[name] = _materialize_partition(
                work_dir=work_dir,
                name=name,
                job_ids=job_ids,
                job_lt=job_lt,
                catalog=catalog,
                outputs=part_outputs,
                returns=returns,
                exclude_zero_lag=split_exclude_zero_lag,
                unshifted_job_ids=unshifted_job_ids,
            )
        return result

    selection = selection or {}
    sel_fraction = float(selection.get("fraction", fraction))
    sel_seed = int(selection.get("seed", seed))
    sel_job_ids = _select_jobs_by_livetime(job_lt, sel_fraction, sel_seed)
    catalog = None
    if "triggers" in returns or _outputs_request_triggers(outputs):
        catalog = _read_catalog(work_dir, catalog_file, trigger_filter)
    selected = _materialize_partition(
        work_dir=work_dir,
        name="selected",
        job_ids=sel_job_ids,
        job_lt=job_lt,
        catalog=catalog,
        outputs=outputs,
        returns=returns,
        exclude_zero_lag=bool(selection.get("exclude_zero_lag", exclude_zero_lag)),
        unshifted_job_ids=unshifted_job_ids,
    )
    selected["mode"] = "selection"
    selected["total"] = _livetime_summary(job_lt, n_triggers=None)
    return selected


@action_spec(
    outputs=["output_file"],
    inputs=["matched_file", "sim_catalog"],
    display_name="Filter real SIM",
    description="Keep recovered, non-vetoed simulation triggers for training",
    help=(
        "Use this before SIM data enters classifier training. The action reads "
        "a matched SIM table, keeps rows with both simulation truth and a "
        "matched trigger, optionally removes vetoed/across-segment injections, "
        "and writes a clean parquet catalog."
    ),
)
def filter_real_simulation(
    work_dir: str,
    matched_file: str,
    output_file: str,
    sim_catalog: Optional[str] = None,
    require_recovered: bool = True,
    exclude_vetoed: bool = True,
    veto_columns: Optional[list[str]] = None,
    output_schema: str = "matched",
    **kwargs,
) -> dict:
    """Filter a matched simulation table to real recovered injection triggers."""
    matched_path = _resolve(work_dir, matched_file)
    out_path = _resolve(work_dir, output_file)
    df = pd.read_parquet(matched_path)
    n_before = len(df)

    sim_col = _first_existing(df, ["sim_sim_idx", "sim_idx", "simulation_id", "sim_id"])
    if sim_col is None:
        sim_candidates = [c for c in df.columns if c.startswith("sim_") and c.endswith("idx")]
        sim_col = sim_candidates[0] if sim_candidates else None
    if sim_col is None:
        raise KeyError("Could not identify simulation index column in matched table")

    mask = df[sim_col].notna()
    if require_recovered:
        trigger_col = _first_existing(df, ["id", "trigger_id", "event_id", "rho"])
        if trigger_col is None:
            raise KeyError("Could not identify trigger column in matched table")
        mask = mask & df[trigger_col].notna()

    veto_columns = veto_columns or [
        "sim_vetoed_cat0",
        "sim_vetoed_cat1",
        "sim_vetoed_cat2",
        "sim_across_segments",
    ]
    if exclude_vetoed:
        for col in veto_columns:
            if col in df.columns:
                mask = mask & (~df[col].fillna(False).astype(bool))

    clean = df[mask].reset_index(drop=True)

    if output_schema == "raw" and sim_catalog:
        id_col = _first_existing(clean, ["id", "trigger_id", "event_id"])
        if id_col is None:
            raise KeyError("output_schema='raw' requires an id-like trigger column")
        raw = pd.read_parquet(_resolve(work_dir, sim_catalog))
        if id_col not in raw.columns:
            raise KeyError(f"Raw SIM catalog does not contain '{id_col}'")
        clean_ids = set(clean[id_col].dropna().values)
        clean = raw[raw[id_col].isin(clean_ids)].reset_index(drop=True)
    elif output_schema != "matched":
        raise ValueError("output_schema must be 'matched' or 'raw'")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    clean.to_parquet(out_path, index=False)
    logger.info(
        "Filtered real SIM: %d -> %d rows written to %s",
        n_before, len(clean), out_path,
    )
    return {
        "triggers_file": output_file,
        "n_before": int(n_before),
        "n_after": int(len(clean)),
        "n_removed": int(n_before - len(clean)),
        "matched_file": matched_file,
        "output_schema": output_schema,
    }


def _resolve(work_dir: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(work_dir, path)


def _filter_progress(
    progress: pd.DataFrame,
    exclude_zero_lag: bool,
    job_filter: Optional[dict[str, Any]],
    unshifted_job_ids: Optional[set[int]] = None,
) -> pd.DataFrame:
    df = progress.copy()
    if "status" in df.columns:
        df = df[df["status"] == "completed"]
    if exclude_zero_lag:
        df = df[nonzero_lag_mask(df, unshifted_job_ids=unshifted_job_ids)]
    if job_filter:
        query = job_filter.get("query") or job_filter.get("where")
        if query:
            df = df.query(query)
        include_jobs = job_filter.get("include_jobs")
        if include_jobs is not None:
            df = df[df["job_id"].isin(include_jobs)]
        exclude_jobs = job_filter.get("exclude_jobs")
        if exclude_jobs is not None:
            df = df[~df["job_id"].isin(exclude_jobs)]
    return df.reset_index(drop=True)


def _read_catalog(
    work_dir: str,
    catalog_file: str,
    trigger_filter: Optional[dict[str, Any]],
) -> pd.DataFrame:
    df = pd.read_parquet(_resolve(work_dir, catalog_file))
    if trigger_filter:
        query = trigger_filter.get("query") or trigger_filter.get("where")
        if query:
            df = df.query(query)
    return df.reset_index(drop=True)


def _shift_by_job_from_catalog(work_dir: str, catalog_file: str) -> dict[int, tuple[float, ...]]:
    from pycwb.modules.catalog.catalog import Catalog

    catalog = Catalog.open(_resolve(work_dir, catalog_file))
    shift_by_job: dict[int, tuple[float, ...]] = {}
    for job in catalog.jobs:
        shift = job.get("shift")
        shift_by_job[int(job["index"])] = tuple(float(value) for value in (shift or []))
    if not shift_by_job:
        raise ValueError("Interval split requires Catalog job metadata with shift values")
    return shift_by_job


def _add_interval_columns(
    progress: pd.DataFrame,
    shift_by_job: dict[int, tuple[float, ...]],
) -> pd.DataFrame:
    if "job_id" not in progress.columns or "lag_idx" not in progress.columns:
        raise KeyError("Interval split requires progress columns 'job_id' and 'lag_idx'")

    df = progress.copy()
    df["_shift_tuple"] = df["job_id"].map(lambda job_id: shift_by_job.get(int(job_id)))
    missing = df[df["_shift_tuple"].isna()]["job_id"].drop_duplicates().tolist()
    if missing:
        raise KeyError(f"Catalog job metadata missing shift for job IDs: {missing[:10]}")

    lag_idx = pd.to_numeric(df["lag_idx"], errors="raise").astype(int)
    df["_interval_key"] = [
        (shift, int(lag))
        for shift, lag in zip(df["_shift_tuple"], lag_idx)
    ]
    df["_shift_key"] = df["_shift_tuple"].map(_shift_key)
    return df


def _select_jobs_by_livetime(job_lt: pd.Series, fraction: float, seed: int) -> list[int]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(job_lt.index.to_numpy()))
    if fraction >= 1:
        return sorted(int(j) for j in shuffled)
    target = float(job_lt.sum()) * fraction
    selected: list[int] = []
    total = 0.0
    for jid in shuffled:
        selected.append(int(jid))
        total += float(job_lt.loc[jid])
        if total >= target:
            break
    return sorted(selected)


def _split_jobs_by_livetime(
    job_lt: pd.Series,
    fractions: dict[str, float] | list[float],
    seed: int,
) -> dict[str, list[int]]:
    if isinstance(fractions, dict):
        items = [(str(k), float(v)) for k, v in fractions.items()]
    else:
        items = [(f"part{i}", float(v)) for i, v in enumerate(fractions)]
    total_fraction = sum(v for _, v in items)
    if total_fraction <= 0 or total_fraction > 1.000001:
        raise ValueError("split fractions must sum to a value in (0, 1]")

    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(job_lt.index.to_numpy()))
    total_lt = float(job_lt.sum())
    partitions: dict[str, list[int]] = {name: [] for name, _ in items}
    cursor = 0
    for idx, (name, frac) in enumerate(items):
        target = total_lt * frac
        acc = 0.0
        is_last_requested = idx == len(items) - 1 and abs(total_fraction - 1.0) < 1e-9
        while cursor < len(shuffled) and (acc < target or (is_last_requested and cursor < len(shuffled))):
            jid = shuffled[cursor]
            partitions[name].append(int(jid))
            acc += float(job_lt.loc[jid])
            cursor += 1
            if is_last_requested:
                continue
            if acc >= target:
                break
    return {name: sorted(ids) for name, ids in partitions.items()}


def _split_intervals_by_livetime(
    interval_lt: pd.Series,
    fractions: dict[str, float] | list[float],
    seed: int,
) -> dict[str, list[tuple[tuple[float, ...], int]]]:
    if isinstance(fractions, dict):
        items = [(str(k), float(v)) for k, v in fractions.items()]
    else:
        items = [(f"part{i}", float(v)) for i, v in enumerate(fractions)]
    total_fraction = sum(v for _, v in items)
    if total_fraction <= 0 or total_fraction > 1.000001:
        raise ValueError("split fractions must sum to a value in (0, 1]")

    rng = np.random.default_rng(seed)
    shuffled = list(interval_lt.index)
    rng.shuffle(shuffled)
    livetime_by_interval = interval_lt.to_dict()
    total_lt = float(interval_lt.sum())
    partitions: dict[str, list[tuple[tuple[float, ...], int]]] = {name: [] for name, _ in items}
    cursor = 0
    for idx, (name, frac) in enumerate(items):
        target = total_lt * frac
        acc = 0.0
        is_last_requested = idx == len(items) - 1 and abs(total_fraction - 1.0) < 1e-9
        while cursor < len(shuffled) and (acc < target or (is_last_requested and cursor < len(shuffled))):
            interval_key = shuffled[cursor]
            partitions[name].append(interval_key)
            acc += float(livetime_by_interval[interval_key])
            cursor += 1
            if is_last_requested:
                continue
            if acc >= target:
                break
    return {
        name: sorted(intervals, key=_interval_sort_key)
        for name, intervals in partitions.items()
    }


def _materialize_partition(
    work_dir: str,
    name: str,
    job_ids: list[int],
    job_lt: pd.Series,
    catalog: Optional[pd.DataFrame],
    outputs: dict[str, Any],
    returns: list[str],
    exclude_zero_lag: bool,
    unshifted_job_ids: Optional[set[int]],
) -> dict:
    result: dict[str, Any] = {}
    n_triggers: Optional[int] = None

    if "jobs" in returns:
        result["job_ids"] = list(job_ids)
    jobs_file = outputs.get("jobs_file") or outputs.get("job_ids_file")
    if jobs_file:
        jobs_path = _resolve(work_dir, jobs_file)
        os.makedirs(os.path.dirname(jobs_path) or ".", exist_ok=True)
        with open(jobs_path, "w") as f:
            for jid in job_ids:
                f.write(f"{jid}\n")
        result["jobs_file"] = jobs_file

    triggers_file = outputs.get("triggers_file") or outputs.get("catalog_file")
    if catalog is not None:
        triggers = catalog[catalog["job_id"].isin(job_ids)].reset_index(drop=True)
        if exclude_zero_lag:
            triggers = triggers[nonzero_lag_mask(triggers, unshifted_job_ids=unshifted_job_ids)].reset_index(drop=True)
        n_triggers = len(triggers)
        if triggers_file:
            triggers_path = _resolve(work_dir, triggers_file)
            os.makedirs(os.path.dirname(triggers_path) or ".", exist_ok=True)
            triggers.to_parquet(triggers_path, index=False)
            result["triggers_file"] = triggers_file
        elif "triggers" in returns:
            result["triggers"] = triggers

    if "livetime" in returns:
        result["livetime"] = _livetime_summary(job_lt.loc[job_ids], n_triggers=n_triggers)

    logger.info(
        "%s: %d jobs, %.0f s live time%s",
        name, len(job_ids), float(job_lt.loc[job_ids].sum()),
        "" if n_triggers is None else f", {n_triggers} triggers",
    )
    return result


def _materialize_interval_partition(
    work_dir: str,
    name: str,
    interval_keys: list[tuple[tuple[float, ...], int]],
    interval_lt: pd.Series,
    progress: pd.DataFrame,
    catalog: Optional[pd.DataFrame],
    outputs: dict[str, Any],
    returns: list[str],
    exclude_zero_lag: bool,
    unshifted_job_ids: Optional[set[int]],
) -> dict:
    result: dict[str, Any] = {}
    interval_set = set(interval_keys)
    selected_progress = progress[progress["_interval_key"].isin(interval_set)].reset_index(drop=True)
    job_ids = sorted(int(job_id) for job_id in selected_progress["job_id"].drop_duplicates())
    n_triggers: Optional[int] = None

    if "jobs" in returns:
        result["job_ids"] = job_ids
    jobs_file = outputs.get("jobs_file") or outputs.get("job_ids_file")
    if jobs_file:
        jobs_path = _resolve(work_dir, jobs_file)
        os.makedirs(os.path.dirname(jobs_path) or ".", exist_ok=True)
        with open(jobs_path, "w") as f:
            for jid in job_ids:
                f.write(f"{jid}\n")
        result["jobs_file"] = jobs_file

    progress_file = outputs.get("progress_file")
    if progress_file:
        progress_path = _resolve(work_dir, progress_file)
        os.makedirs(os.path.dirname(progress_path) or ".", exist_ok=True)
        _progress_output(selected_progress).to_parquet(progress_path, index=False)
        result["progress_file"] = progress_file
    elif "progress" in returns:
        result["progress"] = _progress_output(selected_progress)

    intervals = _intervals_output(selected_progress, interval_keys)
    intervals_file = outputs.get("intervals_file")
    if intervals_file:
        intervals_path = _resolve(work_dir, intervals_file)
        os.makedirs(os.path.dirname(intervals_path) or ".", exist_ok=True)
        intervals.to_parquet(intervals_path, index=False)
        result["intervals_file"] = intervals_file
    intervals_csv_file = outputs.get("intervals_csv_file")
    if intervals_csv_file:
        intervals_csv_path = _resolve(work_dir, intervals_csv_file)
        os.makedirs(os.path.dirname(intervals_csv_path) or ".", exist_ok=True)
        intervals.to_csv(intervals_csv_path, index=False)
        result["intervals_csv_file"] = intervals_csv_file
    if "intervals" in returns:
        result["intervals"] = intervals

    triggers_file = outputs.get("triggers_file") or outputs.get("catalog_file")
    if catalog is not None:
        triggers = _catalog_rows_for_progress(catalog, selected_progress)
        if exclude_zero_lag:
            triggers = triggers[nonzero_lag_mask(triggers, unshifted_job_ids=unshifted_job_ids)].reset_index(drop=True)
        n_triggers = len(triggers)
        if triggers_file:
            triggers_path = _resolve(work_dir, triggers_file)
            os.makedirs(os.path.dirname(triggers_path) or ".", exist_ok=True)
            triggers.to_parquet(triggers_path, index=False)
            result["triggers_file"] = triggers_file
        elif "triggers" in returns:
            result["triggers"] = triggers

    if "livetime" in returns:
        livetime_summary = _livetime_summary(_interval_livetime_values(interval_lt, interval_keys), n_triggers=n_triggers)
        livetime_summary["n_intervals"] = int(len(interval_keys))
        livetime_summary["n_jobs"] = int(len(job_ids))
        result["livetime"] = livetime_summary

    logger.info(
        "%s: %d intervals, %d jobs, %.0f s live time%s",
        name, len(interval_keys), len(job_ids), float(_interval_livetime_values(interval_lt, interval_keys).sum()),
        "" if n_triggers is None else f", {n_triggers} triggers",
    )
    return result


def _livetime_summary(job_lt: pd.Series, n_triggers: Optional[int]) -> dict:
    seconds = float(job_lt.sum())
    summary = {
        "seconds": seconds,
        "days": seconds / 86400.0,
        "years": seconds / 31557600.0,
        "n_jobs": int(len(job_lt)),
    }
    if n_triggers is not None:
        summary["n_triggers"] = int(n_triggers)
    return summary


def _outputs_request_triggers(outputs: dict[str, Any]) -> bool:
    if not outputs:
        return False
    if "triggers_file" in outputs or "catalog_file" in outputs:
        return True
    return any(isinstance(v, dict) and _outputs_request_triggers(v) for v in outputs.values())


def _catalog_rows_for_progress(catalog: pd.DataFrame, progress: pd.DataFrame) -> pd.DataFrame:
    match_cols = ["job_id", "lag_idx"]
    if "trial_idx" in catalog.columns and "trial_idx" in progress.columns:
        match_cols.insert(1, "trial_idx")
    missing = [col for col in match_cols if col not in catalog.columns or col not in progress.columns]
    if missing:
        raise KeyError(f"Cannot materialize interval triggers; missing columns: {missing}")

    keys = pd.MultiIndex.from_frame(progress[match_cols].drop_duplicates())
    catalog_keys = pd.MultiIndex.from_frame(catalog[match_cols])
    return catalog[catalog_keys.isin(keys)].reset_index(drop=True)


def _progress_output(progress: pd.DataFrame) -> pd.DataFrame:
    df = progress.drop(columns=["_interval_key", "_shift_tuple"], errors="ignore").copy()
    return _add_shift_columns(df, progress["_shift_tuple"])


def _intervals_output(
    progress: pd.DataFrame,
    interval_keys: list[tuple[tuple[float, ...], int]],
) -> pd.DataFrame:
    grouped = progress.groupby("_interval_key").agg(
        livetime=("livetime", "sum"),
        n_rows=("job_id", "size"),
        n_jobs=("job_id", "nunique"),
    )
    grouped_records = grouped.to_dict(orient="index")
    rows = []
    for shift, lag_idx in interval_keys:
        summary = grouped_records[(shift, lag_idx)]
        row = {
            "shift_key": _shift_key(shift),
            "lag_idx": int(lag_idx),
            "livetime": float(summary["livetime"]),
            "n_rows": int(summary["n_rows"]),
            "n_jobs": int(summary["n_jobs"]),
        }
        for idx, value in enumerate(shift):
            row[f"shift_{idx}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def _interval_livetime_values(
    interval_lt: pd.Series,
    interval_keys: list[tuple[tuple[float, ...], int]],
) -> pd.Series:
    livetime_by_interval = interval_lt.to_dict()
    return pd.Series([float(livetime_by_interval[key]) for key in interval_keys], dtype=float)


def _add_shift_columns(df: pd.DataFrame, shifts: pd.Series) -> pd.DataFrame:
    df["_shift_key"] = shifts.map(_shift_key).values
    max_len = max((len(shift) for shift in shifts), default=0)
    for idx in range(max_len):
        df[f"_shift_{idx}"] = [
            float(shift[idx]) if idx < len(shift) else np.nan
            for shift in shifts
        ]
    return df.rename(columns={"_shift_key": "shift_key", **{f"_shift_{idx}": f"shift_{idx}" for idx in range(max_len)}})


def _shift_key(shift: tuple[float, ...]) -> str:
    return ",".join(f"{value:.12g}" for value in shift)


def _interval_sort_key(interval_key: tuple[tuple[float, ...], int]) -> tuple:
    shift, lag_idx = interval_key
    return tuple(shift), int(lag_idx)


def _first_existing(df: pd.DataFrame, columns: list[str]) -> Optional[str]:
    for col in columns:
        if col in df.columns:
            return col
    return None
