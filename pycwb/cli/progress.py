"""CLI module for displaying run progress from catalog and progress Parquet files."""

from __future__ import annotations

import glob
import logging
import os
import re
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)

# Set to True by --timeit; controls whether _tlog() prints timing to stderr.
_TIMEIT: bool = False


def _tlog(msg: str) -> None:
    """Print a timing message to stderr when --timeit is active."""
    if _TIMEIT:
        print(f"[TIMER] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# ANSI helpers (degrade gracefully on non-TTY / Windows terminals)
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.name != "nt"

_RESET = "\033[0m"
_BOLD = "\033[1m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_DIM = "\033[2m"


def _c(code: str, text: str) -> str:
    """Wrap *text* in the given ANSI escape *code* when colour output is enabled."""
    if _USE_COLOR:
        return f"{code}{text}{_RESET}"
    return text


def _progress_bar(done: int, total: int, width: int = 30) -> str:
    """Return an ASCII progress bar like ``[████░░░░░░]``."""
    if total <= 0:
        ratio = 0.0
    else:
        ratio = min(done / total, 1.0)
    filled = round(ratio * width)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    pct = f"{ratio * 100:5.1f}%"
    return f"[{bar}] {pct}"


def _fmt_duration(seconds: float) -> str:
    """Human-friendly duration: ``17.2 days``, ``3.5 hours``, ``23.4 min``, or ``45.0s``."""
    if seconds <= 0:
        return "0s"
    if seconds >= 86400:
        return f"{seconds / 86400:.1f} days"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} hours"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f}s"


def _fmt_timestamp(ts: float) -> str:
    """Format a Unix timestamp as a human-readable UTC string, or ``'—'`` for zero/None."""
    if not ts:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts))


def _frac_sort_key(path: str) -> tuple:
    """Numeric sort key for fraction filenames like ``progress_65-128.parquet``."""
    match = re.search(r"(?:progress|catalog)_(\d+)-(\d+)\.parquet", os.path.basename(path))
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (10**18, 10**18)


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def init_parser(parser: Any) -> None:
    """
    Initialize argument parser for the ``pycwb progress`` command.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The sub-parser to configure with progress-command arguments.
    """
    parser.add_argument(
        "--work-dir",
        "-d",
        metavar="work_dir",
        type=str,
        default=".",
        help="working directory (default: current directory)",
    )
    parser.add_argument(
        "--catalog-dir",
        metavar="catalog_dir",
        type=str,
        default="catalog",
        help="sub-directory containing the catalog/progress files (default: catalog)",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        metavar="job_range",
        type=str,
        default=None,
        help='restrict display to these jobs, e.g. "1-5,7"',
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="show per-job breakdown table (hidden by default)",
    )
    parser.add_argument(
        "--timeit",
        "-t",
        action="store_true",
        default=False,
        help="print per-step wall-clock timings to stderr for performance profiling",
    )
    parser.add_argument(
        "--label",
        "-l",
        metavar="label",
        type=str,
        default=None,
        help="use progress.{label}.parquet / catalog.{label}.parquet as the merged files",
    )


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _expected_trials(ws: Any) -> int:
    """Return the number of unique trial indices expected for a WaveSegment.

    Derived from ``ws.injections`` so the denominator is correct even before
    all trials have written any progress rows.
    """
    if ws.injections:
        return len({inj.get("trial_idx", 0) for inj in ws.injections})
    return 1


def _parse_job_range(spec: str) -> set[int]:
    """Parse a job range specification like ``"1-5,7,9-12"`` into a set."""
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        elif part:
            result.add(int(part))
    return result


def _load_jobs_from_catalog(
    catalog_file: str,
    jobs: dict[int, Any],
    lag_file_nlag_cache: dict[str, int],
) -> None:
    """Load jobs from a single catalog file into *jobs* in-place.

    *lag_file_nlag_cache* is a caller-managed dict mapping lag_file paths to
    their row count, shared across calls to avoid re-reading the same file
    from slow network storage multiple times.
    """
    from dacite import from_dict, Config as DaciteConfig
    from pycwb.modules.catalog import Catalog
    from pycwb.types.job import WaveSegment

    try:
        _t_read = time.perf_counter()
        cat = Catalog.open(catalog_file)
        _tlog(f"catalog read   {os.path.basename(catalog_file)}  {time.perf_counter() - _t_read:.3f}s")

        _t_proc = time.perf_counter()
        for jd in cat.jobs:
            ws = from_dict(WaveSegment, jd, config=DaciteConfig(cast=[tuple]))
            if ws.index not in jobs:
                # All jobs in a run often share the same lag_file on (slow) network
                # storage.  Read each unique path at most once.
                if ws.lag_file is not None and ws.lag_file in lag_file_nlag_cache:
                    _n_lag = lag_file_nlag_cache[ws.lag_file]
                else:
                    _n_lag = ws.n_lag  # may read lag_file from disk
                    if ws.lag_file is not None:
                        lag_file_nlag_cache[ws.lag_file] = _n_lag
                _n_trials = _expected_trials(ws)
                ws._expected_lags = _n_lag * _n_trials  # type: ignore[attr-defined]
                jobs[ws.index] = ws
        _tlog(f"catalog process {os.path.basename(catalog_file)}  {len(cat.jobs)} jobs  {time.perf_counter() - _t_proc:.3f}s")
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not load jobs from %s: %s", catalog_file, exc)


def _load_progress_table(path: str) -> Any | None:
    """Read a progress Parquet file; return None on any error."""
    import pyarrow.parquet as pq
    from pycwb.modules.catalog.catalog import PROGRESS_SCHEMA

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        return pq.read_table(path, schema=PROGRESS_SCHEMA)
    except Exception:  # schema mismatch — retry without enforcing schema
        try:
            return pq.read_table(path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not read progress file %s: %s", path, exc)
            return None


def _summarise_table(table: Any) -> dict:
    """
    Return summary stats dict from a progress table.

    Uses vectorised pyarrow ``group_by`` so that only the small per-job
    aggregate result (one row per job) is ever converted to Python objects,
    rather than converting every individual lag row.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    if len(table) == 0:
        return {"by_job": {}, "total_rows": 0, "total_triggers": 0, "total_livetime": 0.0, "latest_ts": 0.0}

    schema_names = {f.name for f in table.schema}

    # Build is_skip boolean column from the status field.
    if "status" in schema_names:
        is_skip = pc.equal(table.column("status"), "skipped_segTHR")
    else:
        is_skip = pa.chunked_array([pa.repeat(pa.scalar(False), len(table))])

    # Helper: fetch a column as the given type, filling nulls with 0.
    def _num_col(name: str, dtype: pa.DataType) -> Any:
        if name in schema_names:
            return pc.fill_null(pc.cast(table.column(name), dtype), pa.scalar(0, dtype))
        return pa.chunked_array([pa.repeat(pa.scalar(0, dtype), len(table))])

    n_triggers_arr = _num_col("n_triggers", pa.int64())
    livetime_arr = _num_col("livetime", pa.float64())
    timestamp_arr = _num_col("timestamp", pa.float64())

    # Group by job_id and aggregate — stays entirely in Arrow memory.
    # agg_tbl has one row per unique job_id (~64 rows), so the final
    # to_pylist() call is cheap.
    agg_tbl = (
        pa.table(
            {
                "job_id": table.column("job_id"),
                "lags_skip": pc.cast(is_skip, pa.int64()),
                "lags_done": pc.cast(pc.invert(is_skip), pa.int64()),
                "n_triggers": n_triggers_arr,
                "livetime": livetime_arr,
                "timestamp": timestamp_arr,
            }
        )
        .group_by("job_id")
        .aggregate(
            [
                ("lags_skip", "sum"),
                ("lags_done", "sum"),
                ("n_triggers", "sum"),
                ("livetime", "sum"),
                ("timestamp", "max"),
            ]
        )
    )

    by_job: dict[int, dict] = {}
    for row in agg_tbl.to_pylist():
        by_job[row["job_id"]] = {
            "lags_done": int(row["lags_done_sum"] or 0),
            "lags_skip": int(row["lags_skip_sum"] or 0),
            "n_triggers": int(row["n_triggers_sum"] or 0),
            "livetime": float(row["livetime_sum"] or 0.0),
            # trials is no longer collected per-row; kept as empty set for API compatibility.
            "trials": set(),
            "latest_ts": float(row["timestamp_max"] or 0.0),
        }

    return {
        "by_job": by_job,
        "total_rows": len(table),
        "total_triggers": int(pc.sum(n_triggers_arr).as_py() or 0),
        "total_livetime": float(pc.sum(livetime_arr).as_py() or 0.0),
        "latest_ts": float(pc.max(timestamp_arr).as_py() or 0.0),
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

_SEP_THICK = "═" * 72
_SEP_THIN = "─" * 72


def _aggregate_fraction_summaries(frac_summaries: dict) -> "dict | None":
    """Merge all fraction summary dicts into one combined summary."""
    if not frac_summaries:
        return None
    combined: dict[int, dict] = {}
    total_triggers = 0
    total_livetime = 0.0
    total_rows = 0
    latest_ts = 0.0
    for s in frac_summaries.values():
        total_triggers += s["total_triggers"]
        total_livetime += s["total_livetime"]
        total_rows += s["total_rows"]
        if s["latest_ts"] > latest_ts:
            latest_ts = s["latest_ts"]
        for jid, jstats in s["by_job"].items():
            if jid not in combined:
                combined[jid] = {
                    "lags_done": 0,
                    "lags_skip": 0,
                    "n_triggers": 0,
                    "livetime": 0.0,
                    "trials": set(),
                    "latest_ts": 0.0,
                }
            e = combined[jid]
            e["lags_done"] += jstats["lags_done"]
            e["lags_skip"] += jstats["lags_skip"]
            e["n_triggers"] += jstats["n_triggers"]
            e["livetime"] += jstats["livetime"]
            e["trials"] |= jstats["trials"]
            if jstats["latest_ts"] > e["latest_ts"]:
                e["latest_ts"] = jstats["latest_ts"]
    return {
        "by_job": combined,
        "total_rows": total_rows,
        "total_triggers": total_triggers,
        "total_livetime": total_livetime,
        "latest_ts": latest_ts,
    }


def _print_fraction_header() -> None:
    """Print the FRACTION PROGRESS FILES section heading."""
    print(_c(_BOLD, "\nFRACTION PROGRESS FILES"))
    print(_SEP_THIN)


def _print_fraction_catalog(
    label: str,
    progress_file: str | None,
    s_jid: int,
    e_jid: int,
    job_expected_lags: dict[int, int],
    summaries: dict[str, dict],
) -> None:
    """Print one progress row for the fraction covering jobs *s_jid*–*e_jid*.

    If *progress_file* is None the fraction is shown as "not started".
    Updates *summaries* in-place when a readable progress file is found.
    """
    total_lags = sum(lags for jid, lags in job_expected_lags.items() if s_jid <= jid <= e_jid)

    if progress_file is None:
        bar = _progress_bar(0, total_lags if total_lags else 1)
        if s_jid == e_jid:
            display_label = f"(not started \u2014 job {s_jid})"
            task_range = f"job {s_jid}"
        else:
            display_label = f"(not started \u2014 jobs {s_jid}\u2013{e_jid})"
            task_range = f"jobs {s_jid}\u2013{e_jid}"
        unit_info = f"    0/{total_lags:<5} tasks" if total_lags else "    0 tasks"
        print(
            f"  {_c(_DIM, display_label):<50}  {task_range:<15}  {unit_info}  {bar}  {_c(_DIM, '0')} triggers  livetime 0s"
        )
        return

    table = _load_progress_table(progress_file)
    if table is None:
        print(f"  {label:<40}  {_c(_RED, 'unreadable / empty')}")
        return

    s = _summarise_table(table)
    summaries[progress_file] = s

    done_lags = sum(v["lags_done"] + v["lags_skip"] for v in s["by_job"].values())

    if total_lags == 0:
        # no catalog metadata for this range — fall back to progress rows
        total_lags = done_lags

    bar = _progress_bar(done_lags, total_lags if total_lags else done_lags)
    if s_jid == e_jid:
        task_range = f"job {s_jid}"
    else:
        task_range = f"jobs {s_jid}\u2013{e_jid}"
    n_trigs = s["total_triggers"]
    livetime = _fmt_duration(s["total_livetime"])
    unit_info = f"{done_lags:>5}/{total_lags:<5} tasks" if total_lags else f"{done_lags:>5} tasks"

    print(
        f"  {_c(_CYAN, label):<50}  "
        f"{task_range:<15}  {unit_info}  {bar}  "
        f"{_c(_GREEN, str(n_trigs))} triggers  livetime {livetime}"
    )


def _print_fraction_block(
    frac_progress_files: list[str], jobs_by_index: dict, frac_catalog_files: list[str] | None = None
) -> dict[str, dict]:
    """Print the per-fraction summary, loading jobs one catalog at a time.

    For each fraction catalog file (sorted numerically), jobs are loaded
    immediately via :func:`_load_jobs_from_catalog` and the corresponding
    progress row is printed right away, so the user sees output line-by-line
    rather than waiting for all catalog files to be read first.

    *jobs_by_index* is populated in-place as catalogs are loaded; callers
    that need the complete map after this function returns can read it back.
    """
    _print_fraction_header()

    sorted_catalogs = sorted(frac_catalog_files or [], key=_frac_sort_key)
    sorted_progress = sorted(frac_progress_files, key=_frac_sort_key)

    # Build a lookup from (s_jid, e_jid) → progress file path (filename scan only, no I/O).
    progress_by_key: dict[tuple[int, int], str] = {}
    for pf in sorted_progress:
        m = re.search(r"progress_(\d+)-(\d+)\.parquet", os.path.basename(pf))
        if m:
            progress_by_key[(int(m.group(1)), int(m.group(2)))] = pf

    if not sorted_catalogs and not sorted_progress:
        print(_c(_DIM, "  (none found)"))
        return {}

    lag_file_nlag_cache: dict[str, int] = {}
    # Running map of job_id → expected lag count, built up as catalogs are loaded.
    job_expected_lags: dict[int, int] = {}
    summaries: dict[str, dict] = {}
    seen_keys: set[tuple[int, int]] = set()

    _t_loop = time.perf_counter()
    for cf in sorted_catalogs:
        m = re.search(r"catalog_(\d+)-(\d+)\.parquet", os.path.basename(cf))
        s_jid = int(m.group(1)) if m else 10**18
        e_jid = int(m.group(2)) if m else 10**18
        key = (s_jid, e_jid)
        seen_keys.add(key)

        # Load jobs from this catalog immediately, then print the row.
        _load_jobs_from_catalog(cf, jobs_by_index, lag_file_nlag_cache)
        _t_idx = time.perf_counter()
        for jid, ws in jobs_by_index.items():
            if jid not in job_expected_lags:
                job_expected_lags[jid] = (
                    ws._expected_lags  # type: ignore[attr-defined]
                    if hasattr(ws, "_expected_lags")
                    else ws.n_lag * _expected_trials(ws)
                )
        _tlog(f"expected_lags index  {os.path.basename(cf)}  {len(job_expected_lags)} jobs  {time.perf_counter() - _t_idx:.3f}s")

        label = f"progress_{s_jid}-{e_jid}.parquet"
        _print_fraction_catalog(label, progress_by_key.get(key), s_jid, e_jid, job_expected_lags, summaries)
    _tlog(f"fraction loop  {len(sorted_catalogs)} catalogs  {time.perf_counter() - _t_loop:.3f}s total")

    # Handle any progress files that have no corresponding catalog fraction.
    for (ps, pe), pf in sorted(progress_by_key.items()):
        if (ps, pe) not in seen_keys:
            label = os.path.basename(pf)
            _print_fraction_catalog(label, pf, ps, pe, job_expected_lags, summaries)

    return summaries


def _print_main_block(main_pf: str, jobs_by_index: dict, filter_jobs: set[int] | None) -> dict | None:
    """Print the main merged progress block and return its summary dict."""
    label = os.path.basename(main_pf)
    print(_c(_BOLD, f"MAIN PROGRESS ({label})"))
    print(_SEP_THIN)

    table = _load_progress_table(main_pf)
    if table is None:
        print(_c(_DIM, "  (no merged progress.parquet found — run 'pycwb merge' first)"))
        print()
        return None

    s = _summarise_table(table)

    total_lags_all = 0
    done_lags_all = 0
    for jid, jstats in s["by_job"].items():
        ws = jobs_by_index.get(jid)
        done = jstats["lags_done"] + jstats["lags_skip"]
        done_lags_all += done
        if ws is not None:
            total_lags_all += ws._expected_lags if hasattr(ws, "_expected_lags") else ws.n_lag * _expected_trials(ws)  # type: ignore[attr-defined]
        else:
            total_lags_all += done

    bar = _progress_bar(done_lags_all, total_lags_all if total_lags_all else done_lags_all)
    lag_str = f"{done_lags_all}/{total_lags_all}" if total_lags_all else str(done_lags_all)
    print(
        f"  Total {lag_str} lags  {bar}\n"
        f"  Triggers : {_c(_GREEN, str(s['total_triggers']))}\n"
        f"  Livetime : {_fmt_duration(s['total_livetime'])}\n"
        f"  Last update: {_fmt_timestamp(s['latest_ts'])}"
    )
    print()
    return s


def _print_job_table(s: dict, jobs_by_index: dict, filter_jobs: set[int] | None) -> None:
    """Print the per-job breakdown table."""
    print(_c(_BOLD, "PER-JOB BREAKDOWN"))
    print(_SEP_THIN)
    header = (
        f"  {'Job':>4}  {'Done':>5}/{'Total':<5}  {'Skip':>4}  "
        f"{'Progress':<38}  {'Triggers':>8}  {'Livetime':>12}  Last updated"
    )
    print(_c(_DIM, header))
    print(_c(_DIM, "  " + "─" * 100))

    warned_no_meta = False
    for jid in sorted(s["by_job"].keys()):
        if filter_jobs is not None and jid not in filter_jobs:
            continue
        jstats = s["by_job"][jid]
        done = jstats["lags_done"]
        skip = jstats["lags_skip"]
        total_done = done + skip
        ws = jobs_by_index.get(jid)
        if ws is not None:
            total = ws._expected_lags if hasattr(ws, "_expected_lags") else ws.n_lag * _expected_trials(ws)  # type: ignore[attr-defined]
        else:
            total = total_done
            warned_no_meta = True

        bar = _progress_bar(total_done, total if total else total_done)
        trigs = jstats["n_triggers"]
        lt = _fmt_duration(jstats["livetime"])
        ts = _fmt_timestamp(jstats["latest_ts"])

        # Color the job id based on completeness
        if total > 0 and total_done >= total:
            jid_str = _c(_GREEN, f"{jid:>4}")
        elif total_done > 0:
            jid_str = _c(_YELLOW, f"{jid:>4}")
        else:
            jid_str = _c(_RED, f"{jid:>4}")

        print(f"  {jid_str}  {done:>5}/{total:<5}  {skip:>4}s  {bar}  {trigs:>8}  {lt:>12}  {ts}")

    print()
    if warned_no_meta:
        print(_c(_YELLOW, "  ⚠ Some jobs have no catalog metadata — totals may be underestimated."))
        print()


def _print_consistency_check(frac_summaries: dict, main_s: dict | None) -> None:
    """Compare fraction totals to the main merged file and flag discrepancies."""
    print(_c(_BOLD, "CONSISTENCY CHECK"))
    print(_SEP_THIN)

    issues = []

    if not frac_summaries:
        print(_c(_DIM, "  No fraction files to check against."))
        print()
        return

    # Aggregate lags from fractions
    frac_by_job: dict[int, int] = {}
    for pf, s in frac_summaries.items():
        for jid, jstats in s["by_job"].items():
            frac_by_job[jid] = frac_by_job.get(jid, 0) + jstats["lags_done"] + jstats["lags_skip"]

    frac_total_rows = sum(s["total_rows"] for s in frac_summaries.values())

    if main_s is None:
        print(_c(_YELLOW, f"  ⚠ No main progress.parquet — fractions total {frac_total_rows} lag rows."))
        print()
        return

    main_total_rows = main_s["total_rows"]

    # Per-job lag count comparison
    for jid in sorted(set(list(frac_by_job.keys()) + list(main_s["by_job"].keys()))):
        frac_n = frac_by_job.get(jid, 0)
        main_stats = main_s["by_job"].get(jid)
        main_n = (main_stats["lags_done"] + main_stats["lags_skip"]) if main_stats else 0
        if frac_n != main_n:
            issues.append(f"  Job {jid}: fractions report {frac_n} lags, main has {main_n} lags")

    if frac_total_rows != main_total_rows:
        issues.append(
            f"  Row count mismatch: fractions total {frac_total_rows}, main progress.parquet has {main_total_rows}"
        )

    if issues:
        print(_c(_YELLOW, "  ⚠ Inconsistencies detected:"))
        for iss in issues:
            print(_c(_YELLOW, iss))
    else:
        print(_c(_GREEN, f"  ✓ All {len(frac_summaries)} fraction file(s) are consistent with main progress.parquet"))

    print()


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------


def command(args: Any) -> None:
    """
    Display run progress across all fraction and main progress files.

    Reads catalog and progress Parquet files from *args.catalog_dir* inside
    *args.work_dir* and prints a formatted report to stdout.  The report
    includes per-fraction progress bars, an overall summary from the merged
    ``progress.parquet``, an optional per-job breakdown table (``--verbose``),
    and a consistency check comparing fraction totals to the merged file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.  Expected attributes:
        ``work_dir``, ``catalog_dir``, ``jobs``, ``verbose``, ``label``.
    """
    global _TIMEIT
    _TIMEIT = args.timeit

    work_dir = os.path.abspath(args.work_dir)
    catalog_dir = os.path.join(work_dir, args.catalog_dir)
    filter_jobs = _parse_job_range(args.jobs) if args.jobs else None

    print()
    print(_c(_BOLD, _SEP_THICK))
    print(_c(_BOLD, "  pycWB Progress Report"))
    print(_c(_DIM, f"  catalog dir: {catalog_dir}"))
    print(_c(_BOLD, _SEP_THICK))
    print()

    # ── Discover files ──────────────────────────────────────────────────
    frac_catalog_files = sorted(glob.glob(os.path.join(catalog_dir, "catalog_*.parquet")), key=_frac_sort_key)
    frac_progress_files = sorted(glob.glob(os.path.join(catalog_dir, "progress_*.parquet")), key=_frac_sort_key)
    label = args.label
    if label:
        main_progress_file = os.path.join(catalog_dir, f"progress.{label}.parquet")
        main_catalog_file = os.path.join(catalog_dir, f"catalog.{label}.parquet")
    else:
        main_progress_file = os.path.join(catalog_dir, "progress.parquet")
        main_catalog_file = os.path.join(catalog_dir, "catalog.parquet")

    # ── Build job metadata map ──────────────────────────────────────────
    jobs_by_index: dict = {}

    if filter_jobs is not None:
        print(_c(_DIM, f"  Showing jobs: {sorted(filter_jobs)}\n"))

    # ── Render fraction block (loads job metadata catalog-by-catalog) ───
    frac_summaries = _print_fraction_block(frac_progress_files, jobs_by_index, frac_catalog_files)

    # ── Fall back to main catalog if no fraction catalogs were found ────
    if not jobs_by_index and os.path.exists(main_catalog_file):
        try:
            _t0 = time.perf_counter()
            _load_jobs_from_catalog(main_catalog_file, jobs_by_index, {})
            _tlog(f"load_jobs  main catalog → {len(jobs_by_index)} jobs  {time.perf_counter() - _t0:.3f}s")
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not load job metadata from main catalog: %s", exc)
    main_s = _print_main_block(main_progress_file, jobs_by_index, filter_jobs)

    if args.verbose:
        if main_s is not None:
            _print_job_table(main_s, jobs_by_index, filter_jobs)
        else:
            combined_s = _aggregate_fraction_summaries(frac_summaries)
            if combined_s:
                print(_c(_BOLD, "PER-JOB BREAKDOWN (aggregated from fractions)"))
                print(_SEP_THIN)
                _print_job_table(combined_s, jobs_by_index, filter_jobs)

    _print_consistency_check(frac_summaries, main_s)
