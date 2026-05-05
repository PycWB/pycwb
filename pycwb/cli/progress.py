"""CLI module for displaying run progress from catalog and progress Parquet files."""

import glob
import os
import re
import sys
import time


# ---------------------------------------------------------------------------
# ANSI helpers (degrade gracefully on non-TTY / Windows terminals)
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.name != "nt"

_RESET = "\033[0m"
_BOLD  = "\033[1m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED   = "\033[91m"
_CYAN  = "\033[96m"
_DIM   = "\033[2m"


def _c(code: str, text: str) -> str:
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
    empty  = width - filled
    bar = "█" * filled + "░" * empty
    pct = f"{ratio * 100:5.1f}%"
    return f"[{bar}] {pct}"


def _fmt_duration(seconds: float) -> str:
    """Human-friendly duration: ``1h 23m 45s``."""
    if seconds <= 0:
        return "0s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s:.1f}s")
    return " ".join(parts)


def _fmt_timestamp(ts: float) -> str:
    if not ts:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts))


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def init_parser(parser):
    """Initialize argument parser for the ``pycwb progress`` command."""
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
        "--label",
        "-l",
        metavar="label",
        type=str,
        default=None,
        help='use progress.{label}.parquet / catalog.{label}.parquet as the merged files',
    )


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _expected_trials(ws) -> int:
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


def _load_jobs_from_catalogs(catalog_files: list[str]) -> dict[int, object]:
    """Return {job_index: WaveSegment} for all unique jobs found in *catalog_files*."""
    from dacite import from_dict
    from pycwb.modules.catalog import Catalog
    from pycwb.types.job import WaveSegment

    jobs: dict[int, WaveSegment] = {}
    for cf in catalog_files:
        try:
            cat = Catalog.open(cf)
            for jd in cat.jobs:
                ws = from_dict(WaveSegment, jd)
                if ws.index not in jobs:
                    jobs[ws.index] = ws
        except Exception:
            pass
    return jobs


def _load_progress_table(path: str) -> "pa.Table | None":
    """Read a progress Parquet file; return None on any error."""
    import pyarrow.parquet as pq
    from pycwb.modules.catalog.catalog import PROGRESS_SCHEMA

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        return pq.read_table(path, schema=PROGRESS_SCHEMA)
    except Exception:
        try:
            return pq.read_table(path)
        except Exception:
            return None


def _summarise_table(table: "pa.Table") -> dict:
    """Return summary stats dict from a progress table."""
    import pyarrow.compute as pc

    rows = table.to_pylist()
    by_job: dict[int, dict] = {}
    total_triggers = 0
    total_livetime  = 0.0
    latest_ts = 0.0

    for r in rows:
        jid = r["job_id"]
        if jid not in by_job:
            by_job[jid] = {
                "lags_done":  0,
                "lags_skip":  0,
                "n_triggers": 0,
                "livetime":   0.0,
                "trials":     set(),
                "latest_ts":  0.0,
            }
        entry = by_job[jid]
        if r.get("status", "completed") == "skipped_segTHR":
            entry["lags_skip"] += 1
        else:
            entry["lags_done"] += 1
        entry["n_triggers"] += r.get("n_triggers", 0) or 0
        entry["livetime"]   += r.get("livetime", 0.0) or 0.0
        entry["trials"].add(r.get("trial_idx", 0))
        ts = r.get("timestamp", 0.0) or 0.0
        if ts > entry["latest_ts"]:
            entry["latest_ts"] = ts
        if ts > latest_ts:
            latest_ts = ts
        total_triggers += r.get("n_triggers", 0) or 0
        total_livetime  += r.get("livetime", 0.0) or 0.0

    return {
        "by_job":         by_job,
        "total_rows":     len(rows),
        "total_triggers": total_triggers,
        "total_livetime": total_livetime,
        "latest_ts":      latest_ts,
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

_SEP_THICK = "═" * 72
_SEP_THIN  = "─" * 72


def _aggregate_fraction_summaries(frac_summaries: dict) -> "dict | None":
    """Merge all fraction summary dicts into one combined summary."""
    if not frac_summaries:
        return None
    combined: dict[int, dict] = {}
    total_triggers = 0
    total_livetime  = 0.0
    total_rows = 0
    latest_ts = 0.0
    for s in frac_summaries.values():
        total_triggers += s["total_triggers"]
        total_livetime  += s["total_livetime"]
        total_rows      += s["total_rows"]
        if s["latest_ts"] > latest_ts:
            latest_ts = s["latest_ts"]
        for jid, jstats in s["by_job"].items():
            if jid not in combined:
                combined[jid] = {
                    "lags_done":  0, "lags_skip":  0,
                    "n_triggers": 0, "livetime":   0.0,
                    "trials":     set(), "latest_ts": 0.0,
                }
            e = combined[jid]
            e["lags_done"]  += jstats["lags_done"]
            e["lags_skip"]  += jstats["lags_skip"]
            e["n_triggers"] += jstats["n_triggers"]
            e["livetime"]   += jstats["livetime"]
            e["trials"]     |= jstats["trials"]
            if jstats["latest_ts"] > e["latest_ts"]:
                e["latest_ts"] = jstats["latest_ts"]
    return {
        "by_job":         combined,
        "total_rows":     total_rows,
        "total_triggers": total_triggers,
        "total_livetime": total_livetime,
        "latest_ts":      latest_ts,
    }


def _print_fraction_block(frac_files: list[str], jobs_by_index: dict) -> dict[str, dict]:
    """Print the per-fraction summary and return {filename: summary_dict}."""
    print(_c(_BOLD, "\nFRACTION PROGRESS FILES"))
    print(_SEP_THIN)

    if not frac_files:
        print(_c(_DIM, "  (none found)"))
        return {}

    summaries: dict[str, dict] = {}
    for pf in sorted(frac_files):
        table = _load_progress_table(pf)
        label = os.path.basename(pf)
        if table is None:
            print(f"  {label:<40}  {_c(_RED, 'unreadable / empty')}")
            continue

        s = _summarise_table(table)
        summaries[pf] = s

        # Total lags expected from jobs referenced in this fraction's progress
        total_lags = 0
        for jid, jstats in s["by_job"].items():
            ws = jobs_by_index.get(jid)
            if ws is not None:
                n_trials = _expected_trials(ws)
                total_lags += ws.n_lag * n_trials
            else:
                total_lags += (jstats["lags_done"] + jstats["lags_skip"])  # best effort

        done_lags = sum(v["lags_done"] + v["lags_skip"] for v in s["by_job"].values())
        bar = _progress_bar(done_lags, total_lags if total_lags else done_lags)
        # Derive displayed range from the filename (encodes the worker/task range)
        m = re.match(r'progress_(\d+)-(\d+)\.parquet', label)
        if m:
            task_range = f"jobs {m.group(1)}\u2013{m.group(2)}"
        else:
            job_ids = sorted(s["by_job"].keys())
            task_range = f"jobs {job_ids[0]}\u2013{job_ids[-1]}" if len(job_ids) > 1 else f"job {job_ids[0]}"
        n_trigs  = s["total_triggers"]
        livetime = _fmt_duration(s["total_livetime"])

        unit_info = f"{done_lags:>5}/{total_lags:<5} tasks" if total_lags else f"{done_lags:>5} tasks"

        print(
            f"  {_c(_CYAN, label):<50}  "
            f"{task_range:<15}  {unit_info}  {bar}  "
            f"{_c(_GREEN, str(n_trigs))} triggers  livetime {livetime}"
        )

    print()
    return summaries


def _print_main_block(main_pf: str, jobs_by_index: dict, filter_jobs: "set[int] | None") -> "dict | None":
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
    done_lags_all  = 0
    for jid, jstats in s["by_job"].items():
        ws = jobs_by_index.get(jid)
        done = jstats["lags_done"] + jstats["lags_skip"]
        done_lags_all += done
        if ws is not None:
            n_trials = _expected_trials(ws)
            total_lags_all += ws.n_lag * n_trials
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


def _print_job_table(s: dict, jobs_by_index: dict, filter_jobs: "set[int] | None"):
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
        done  = jstats["lags_done"]
        skip  = jstats["lags_skip"]
        total_done = done + skip
        ws = jobs_by_index.get(jid)
        if ws is not None:
            n_trials = _expected_trials(ws)
            total = ws.n_lag * n_trials
        else:
            total = total_done
            warned_no_meta = True

        bar  = _progress_bar(total_done, total if total else total_done)
        trigs = jstats["n_triggers"]
        lt   = _fmt_duration(jstats["livetime"])
        ts   = _fmt_timestamp(jstats["latest_ts"])

        # Color the job id based on completeness
        if total > 0 and total_done >= total:
            jid_str = _c(_GREEN, f"{jid:>4}")
        elif total_done > 0:
            jid_str = _c(_YELLOW, f"{jid:>4}")
        else:
            jid_str = _c(_RED, f"{jid:>4}")

        print(
            f"  {jid_str}  {done:>5}/{total:<5}  {skip:>4}s  "
            f"{bar}  {trigs:>8}  {lt:>12}  {ts}"
        )

    print()
    if warned_no_meta:
        print(_c(_YELLOW, "  ⚠ Some jobs have no catalog metadata — totals may be underestimated."))
        print()


def _print_consistency_check(frac_summaries: dict, main_s: "dict | None"):
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
            issues.append(
                f"  Job {jid}: fractions report {frac_n} lags, main has {main_n} lags"
            )

    if frac_total_rows != main_total_rows:
        issues.append(
            f"  Row count mismatch: fractions total {frac_total_rows}, "
            f"main progress.parquet has {main_total_rows}"
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

def command(args):
    """Display run progress across all fraction and main progress files."""
    work_dir   = os.path.abspath(args.work_dir)
    catalog_dir = os.path.join(work_dir, args.catalog_dir)
    filter_jobs = _parse_job_range(args.jobs) if args.jobs else None

    print()
    print(_c(_BOLD, _SEP_THICK))
    print(_c(_BOLD, f"  pycWB Progress Report"))
    print(_c(_DIM,  f"  catalog dir: {catalog_dir}"))
    print(_c(_BOLD, _SEP_THICK))
    print()

    # ── Discover files ──────────────────────────────────────────────────
    frac_catalog_files  = sorted(glob.glob(os.path.join(catalog_dir, "catalog_*.parquet")))
    frac_progress_files = sorted(glob.glob(os.path.join(catalog_dir, "progress_*.parquet")))
    label = args.label
    if label:
        main_progress_file  = os.path.join(catalog_dir, f"progress.{label}.parquet")
        main_catalog_file   = os.path.join(catalog_dir, f"catalog.{label}.parquet")
    else:
        main_progress_file  = os.path.join(catalog_dir, "progress.parquet")
        main_catalog_file   = os.path.join(catalog_dir, "catalog.parquet")

    # ── Build job metadata map ──────────────────────────────────────────
    jobs_by_index: dict = {}
    if frac_catalog_files:
        try:
            jobs_by_index = _load_jobs_from_catalogs(frac_catalog_files)
        except Exception as exc:
            print(_c(_YELLOW, f"  ⚠ Could not load job metadata from catalogs: {exc}"))
    if not jobs_by_index and os.path.exists(main_catalog_file):
        try:
            jobs_by_index = _load_jobs_from_catalogs([main_catalog_file])
        except Exception:
            pass

    if filter_jobs is not None:
        print(_c(_DIM, f"  Showing jobs: {sorted(filter_jobs)}\n"))

    # ── Render blocks ───────────────────────────────────────────────────
    frac_summaries = _print_fraction_block(frac_progress_files, jobs_by_index)
    main_s         = _print_main_block(main_progress_file, jobs_by_index, filter_jobs)

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
