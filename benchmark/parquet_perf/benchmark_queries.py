"""
tests/parquet_perf/benchmark_queries.py
=========================================
Benchmark common user-facing queries against the InjectionParams Parquet files
produced by write_parquet.py.

Each benchmark function receives a file path (str) and returns the number of
matching rows plus the wall-clock time in milliseconds.  A summary table is
printed at the end.

Queries exercised
-----------------
1. **Count by approximant** – full scan to count WNB / SG / BBH rows.
2. **WNB only** – filter ``injection.approximant == 'WNB'``.
3. **SG only** – filter ``injection.approximant == 'SGE'``.
4. **BBH only** – filter ``injection.approximant in BBH_APPROXIMANTS``.
5. **BBH heavy** – BBH with total mass (mass1 + mass2) > 60 M☉
   (extracted from the JSON ``parameters`` blob via DuckDB ``json_extract``).
6. **BBH spin aligned** – |spin1z| > 0.5 (from JSON blob).
7. **High hrss** – ``injection.hrss > 1e-22`` (typed column → predicate pushdown).
8. **Specific GPS window** – triggers in a 1-day GPS window.
9. **SG high-Q** – SG with Q == 100 (from JSON blob).
10. **WNB low-frequency** – WNB with frequency < 100 Hz (from JSON blob).
11. **Round-trip consistency** – deserialise every row back to InjectionParams.

Two backends are benchmarked where available:
* **PyArrow** – ``pyarrow.parquet.read_table`` with ``filters=`` push-down.
* **DuckDB** – SQL over Parquet (very fast for struct/JSON columns).

Usage
-----
    # from workspace root (generates + writes first if files are missing)
    python tests/parquet_perf/benchmark_queries.py

    # already have parquet/ populated
    python tests/parquet_perf/benchmark_queries.py --parquet-dir tests/parquet_perf/parquet

    # single file
    python tests/parquet_perf/benchmark_queries.py --file tests/parquet_perf/parquet/params_10k.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pycwb.types.trigger import InjectionParams

# ---------------------------------------------------------------------------
# DuckDB import (optional)
# ---------------------------------------------------------------------------
try:
    import duckdb
    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False

_BBH_APPROXIMANTS = frozenset([
    "IMRPhenomTPHM", "IMRPhenomXPHM", "SEOBNRv4PHM",
    "NRSur7dq4", "IMRPhenomD",
])

# GPS window: one sidereal day centred around a reference time
_GPS_REF   = 1187008882.4   # GW170817
_GPS_HALF  = 43082.0         # half-day in seconds


# ===========================================================================
# PyArrow benchmarks
# ===========================================================================

def _pa_time(fn: Callable[[], pa.Table]) -> tuple[int, float]:
    """Run *fn*, return (nrows, elapsed_ms)."""
    t0 = time.perf_counter()
    tbl = fn()
    elapsed = (time.perf_counter() - t0) * 1e3
    return len(tbl), elapsed


def pa_count_by_approximant(path: str) -> dict[str, tuple[int, float]]:
    """Count rows per approximant (full scan, no push-down for struct columns)."""
    results = {}
    for approx in ("WNB", "SGE", "IMRPhenomTPHM"):
        label = f"pa_count_{approx[:3]}"
        t0 = time.perf_counter()
        tbl = pq.read_table(path, columns=["injection"])
        col = tbl.column("injection")
        approx_arr = pc.struct_field(col, "approximant")
        mask = pa.compute.equal(approx_arr, approx)
        n = int(pa.compute.sum(mask.cast(pa.int32())).as_py())
        elapsed = (time.perf_counter() - t0) * 1e3
        results[label] = (n, elapsed)
    return results


def pa_wnb_only(path: str) -> tuple[int, float]:
    t0 = time.perf_counter()
    tbl = pq.read_table(path, columns=["injection"])
    col = tbl.column("injection")
    mask = pa.compute.equal(pc.struct_field(col, "approximant"), "WNB")
    filtered = tbl.filter(mask)
    return len(filtered), (time.perf_counter() - t0) * 1e3


def pa_sg_only(path: str) -> tuple[int, float]:
    t0 = time.perf_counter()
    tbl = pq.read_table(path, columns=["injection"])
    col = tbl.column("injection")
    mask = pa.compute.equal(pc.struct_field(col, "approximant"), "SGE")
    filtered = tbl.filter(mask)
    return len(filtered), (time.perf_counter() - t0) * 1e3


def pa_bbh_only(path: str) -> tuple[int, float]:
    t0 = time.perf_counter()
    tbl = pq.read_table(path, columns=["injection"])
    col = tbl.column("injection")
    approx_arr = pc.struct_field(col, "approximant")
    mask = pa.compute.is_in(approx_arr, value_set=pa.array(list(_BBH_APPROXIMANTS)))
    filtered = tbl.filter(mask)
    return len(filtered), (time.perf_counter() - t0) * 1e3


def pa_high_hrss(path: str, threshold: float = 1e-22) -> tuple[int, float]:
    """Filter on typed ``injection.hrss`` (float32 → predicate possible)."""
    t0 = time.perf_counter()
    tbl = pq.read_table(path, columns=["injection"])
    col = tbl.column("injection")
    hrss_arr = pc.struct_field(col, "hrss").cast(pa.float64())
    mask = pa.compute.greater(hrss_arr, threshold)
    filtered = tbl.filter(mask)
    return len(filtered), (time.perf_counter() - t0) * 1e3


def pa_gps_window(path: str) -> tuple[int, float]:
    t0 = time.perf_counter()
    tbl = pq.read_table(path, columns=["injection"])
    col = tbl.column("injection")
    gps = pc.struct_field(col, "gps_time")
    lo  = pa.compute.greater_equal(gps, _GPS_REF - _GPS_HALF)
    hi  = pa.compute.less_equal(gps,    _GPS_REF + _GPS_HALF)
    filtered = tbl.filter(pa.compute.and_(lo, hi))
    return len(filtered), (time.perf_counter() - t0) * 1e3


def pa_roundtrip(path: str) -> tuple[int, float]:
    """Deserialise every struct row back to InjectionParams objects."""
    t0 = time.perf_counter()
    tbl  = pq.read_table(path, columns=["injection"])
    col  = tbl.column("injection")
    objs = []
    for row in col.to_pylist():
        inj = InjectionParams(**row)
        objs.append(inj)
    return len(objs), (time.perf_counter() - t0) * 1e3


# ===========================================================================
# DuckDB benchmarks
# ===========================================================================

def _duck(con: "duckdb.DuckDBPyConnection", sql: str) -> tuple[int, float]:
    t0 = time.perf_counter()
    rel = con.execute(sql)
    rows = rel.fetchall()
    elapsed = (time.perf_counter() - t0) * 1e3
    # rows is a list of tuples; sum the first column if it's a count
    n = rows[0][0] if rows and isinstance(rows[0][0], int) else len(rows)
    return n, elapsed


def run_duckdb_benchmarks(path: str) -> list[tuple[str, int, float]]:
    """Return list of (label, n_rows, ms) for all DuckDB queries."""
    if not _HAS_DUCKDB:
        return []

    con = duckdb.connect(":memory:")
    # register the parquet file as a view
    con.execute(f"CREATE VIEW triggers AS SELECT * FROM read_parquet('{path}')")

    results: list[tuple[str, int, float]] = []

    def run(label: str, sql: str) -> None:
        n, ms = _duck(con, sql)
        results.append((label, n, ms))

    run("duck_count_WNB",
        "SELECT COUNT(*) FROM triggers WHERE injection.approximant = 'WNB'")

    run("duck_count_SGE",
        "SELECT COUNT(*) FROM triggers WHERE injection.approximant = 'SGE'")

    run("duck_count_BBH",
        "SELECT COUNT(*) FROM triggers "
        "WHERE injection.approximant NOT IN ('WNB', 'SGE')")

    run("duck_high_hrss",
        "SELECT COUNT(*) FROM triggers WHERE injection.hrss > 1e-22")

    run("duck_gps_window",
        f"SELECT COUNT(*) FROM triggers "
        f"WHERE injection.gps_time BETWEEN {_GPS_REF - _GPS_HALF} "
        f"                              AND {_GPS_REF + _GPS_HALF}")

    run("duck_bbh_heavy_mass",
        "SELECT COUNT(*) FROM triggers "
        "WHERE injection.approximant NOT IN ('WNB', 'SGE') "
        "  AND (CAST(json_extract(injection.parameters, '$.mass1') AS FLOAT) "
        "     + CAST(json_extract(injection.parameters, '$.mass2') AS FLOAT)) > 60")

    run("duck_bbh_spin_aligned",
        "SELECT COUNT(*) FROM triggers "
        "WHERE injection.approximant NOT IN ('WNB', 'SGE') "
        "  AND ABS(CAST(json_extract(injection.parameters, '$.spin1z') AS FLOAT)) > 0.5")

    run("duck_sg_high_Q",
        "SELECT COUNT(*) FROM triggers "
        "WHERE injection.approximant = 'SGE' "
        "  AND CAST(json_extract(injection.parameters, '$.Q') AS FLOAT) = 100.0")

    run("duck_wnb_low_freq",
        "SELECT COUNT(*) FROM triggers "
        "WHERE injection.approximant = 'WNB' "
        "  AND CAST(json_extract(injection.parameters, '$.frequency') AS FLOAT) < 100")

    run("duck_wnb_short_duration",
        "SELECT COUNT(*) FROM triggers "
        "WHERE injection.approximant = 'WNB' "
        "  AND CAST(json_extract(injection.parameters, '$.duration') AS FLOAT) < 0.01")

    # Aggregation: mean hrss per approximant group
    run("duck_mean_hrss_per_approx",
        "SELECT COUNT(*) FROM ("
        "  SELECT injection.approximant, AVG(injection.hrss) AS mean_hrss "
        "  FROM triggers GROUP BY injection.approximant"
        ") sub")

    con.close()
    return results


# ===========================================================================
# PyArrow JSON-extract helpers (for parameters blob queries without DuckDB)
# ===========================================================================

def _pa_json_extract_float(col: pa.ChunkedArray, key: str) -> pa.Array:
    """Extract a float from the ``parameters`` JSON string column."""
    params = pc.struct_field(col, "parameters")
    # json_extract is not built into pyarrow – parse Python-side
    values = []
    for blob in params.to_pylist():
        try:
            d = json.loads(blob)
            values.append(d.get(key))
        except (ValueError, TypeError):
            values.append(None)
    return pa.array(values, type=pa.float64())


def pa_bbh_heavy_mass(path: str) -> tuple[int, float]:
    t0 = time.perf_counter()
    tbl = pq.read_table(path, columns=["injection"])
    col = tbl.column("injection")
    approx_arr = pc.struct_field(col, "approximant")
    is_bbh = pa.compute.is_in(approx_arr, value_set=pa.array(list(_BBH_APPROXIMANTS)))
    bbh_tbl = tbl.filter(is_bbh)
    bbh_col = bbh_tbl.column("injection")
    m1 = _pa_json_extract_float(bbh_col, "mass1")
    m2 = _pa_json_extract_float(bbh_col, "mass2")
    total = pa.compute.add(m1, m2)
    mask  = pa.compute.greater(total, 60.0)
    n = int(pa.compute.sum(mask.cast(pa.int32())).as_py())
    return n, (time.perf_counter() - t0) * 1e3


def pa_bbh_spin(path: str, threshold: float = 0.5) -> tuple[int, float]:
    t0 = time.perf_counter()
    tbl = pq.read_table(path, columns=["injection"])
    col = tbl.column("injection")
    is_bbh = pa.compute.is_in(pc.struct_field(col, "approximant"),
                               value_set=pa.array(list(_BBH_APPROXIMANTS)))
    bbh_col = tbl.filter(is_bbh).column("injection")
    s1z = _pa_json_extract_float(bbh_col, "spin1z")
    mask = pa.compute.greater(pa.compute.abs(s1z), threshold)
    n = int(pa.compute.sum(mask.cast(pa.int32())).as_py())
    return n, (time.perf_counter() - t0) * 1e3


def pa_sg_high_q(path: str) -> tuple[int, float]:
    t0 = time.perf_counter()
    tbl = pq.read_table(path, columns=["injection"])
    col = tbl.column("injection")
    is_sg  = pa.compute.equal(pc.struct_field(col, "approximant"), "SGE")
    sg_col = tbl.filter(is_sg).column("injection")
    q_arr  = _pa_json_extract_float(sg_col, "Q")
    mask   = pa.compute.equal(q_arr, 100.0)
    n = int(pa.compute.sum(mask.cast(pa.int32())).as_py())
    return n, (time.perf_counter() - t0) * 1e3


def pa_wnb_low_freq(path: str) -> tuple[int, float]:
    t0 = time.perf_counter()
    tbl = pq.read_table(path, columns=["injection"])
    col = tbl.column("injection")
    is_wnb  = pa.compute.equal(pc.struct_field(col, "approximant"), "WNB")
    wnb_col = tbl.filter(is_wnb).column("injection")
    freq    = _pa_json_extract_float(wnb_col, "frequency")
    mask    = pa.compute.less(freq, 100.0)
    n = int(pa.compute.sum(mask.cast(pa.int32())).as_py())
    return n, (time.perf_counter() - t0) * 1e3


# ===========================================================================
# Runner
# ===========================================================================

def run_pyarrow_benchmarks(path: str) -> list[tuple[str, int, float]]:
    results: list[tuple[str, int, float]] = []

    def record(label: str, fn: Callable[[], tuple[int, float]]) -> None:
        n, ms = fn()
        results.append((label, n, ms))

    record("pa_wnb_only",       lambda: pa_wnb_only(path))
    record("pa_sg_only",        lambda: pa_sg_only(path))
    record("pa_bbh_only",       lambda: pa_bbh_only(path))
    record("pa_high_hrss",      lambda: pa_high_hrss(path))
    record("pa_gps_window",     lambda: pa_gps_window(path))
    record("pa_bbh_heavy_mass", lambda: pa_bbh_heavy_mass(path))
    record("pa_bbh_spin",       lambda: pa_bbh_spin(path))
    record("pa_sg_high_q",      lambda: pa_sg_high_q(path))
    record("pa_wnb_low_freq",   lambda: pa_wnb_low_freq(path))
    record("pa_roundtrip",      lambda: pa_roundtrip(path))

    return results


def benchmark_file(path: str) -> None:
    n_rows = pq.read_metadata(path).num_rows
    size_mb = os.path.getsize(path) / 1e6
    print(f"\n{'='*70}")
    print(f"  File : {os.path.basename(path)}")
    print(f"  Rows : {n_rows:,}   Size : {size_mb:.2f} MB")
    print(f"{'='*70}")

    # --- PyArrow ---
    pa_results = run_pyarrow_benchmarks(path)
    print(f"\n  [PyArrow]")
    print(f"  {'Query':<30} {'Rows':>8} {'ms':>10}")
    print(f"  {'─'*30} {'─'*8} {'─'*10}")
    for label, n, ms in pa_results:
        print(f"  {label:<30} {n:>8,} {ms:>10.1f}")

    # --- DuckDB ---
    if _HAS_DUCKDB:
        duck_results = run_duckdb_benchmarks(path)
        print(f"\n  [DuckDB]")
        print(f"  {'Query':<35} {'Rows':>8} {'ms':>10}")
        print(f"  {'─'*35} {'─'*8} {'─'*10}")
        for label, n, ms in duck_results:
            print(f"  {label:<35} {n:>8,} {ms:>10.1f}")
    else:
        print("\n  [DuckDB] not available — install with: pip install duckdb")


# ===========================================================================
# CLI
# ===========================================================================

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Parquet queries over InjectionParams data.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--parquet-dir",
        default=os.path.join(_HERE, "parquet"),
        help="Directory containing params_*.parquet files.",
    )
    group.add_argument(
        "--file",
        help="Benchmark a single .parquet file instead of a whole directory.",
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Run generate_params.py + write_parquet.py first if files are missing.",
    )
    args = parser.parse_args(argv)

    if args.file:
        paths = [args.file]
    else:
        parquet_dir = args.parquet_dir
        if not os.path.isdir(parquet_dir) or not any(
            f.endswith(".parquet") for f in os.listdir(parquet_dir)
        ):
            if args.generate:
                print("No Parquet files found – generating now …")
                from generate_params import main as gen_main
                from write_parquet  import main as write_main
                gen_main([])
                write_main([])
            else:
                sys.exit(
                    f"No .parquet files found in {parquet_dir}.\n"
                    "Run generate_params.py and write_parquet.py first, "
                    "or pass --generate."
                )
        paths = sorted(
            os.path.join(parquet_dir, f)
            for f in os.listdir(parquet_dir)
            if f.endswith(".parquet")
        )

    for p in paths:
        benchmark_file(p)

    print(f"\n{'─'*70}")
    print("Done.")


if __name__ == "__main__":
    main()
