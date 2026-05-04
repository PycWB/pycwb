"""
tests/parquet_perf/write_parquet.py
=====================================
Read the JSON parameter files produced by generate_params.py, construct
:class:`InjectionParams` objects via :meth:`InjectionParams.from_injection_dict`,
and write them to Parquet using PyArrow.

Each JSON file produces one ``.parquet`` file of the same base name under
``./data/``.  A simple ``injection``-only table (struct column) is written for
focused benchmarking, plus a full minimal ``Trigger`` table that wraps the
injection in a realistic surrounding schema.

Usage
-----
    # from the workspace root
    python tests/parquet_perf/write_parquet.py

    # from inside the tests/parquet_perf directory
    python write_parquet.py

    # explicit paths
    python write_parquet.py --data-dir ./data --out-dir ./parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Ensure the workspace root is on sys.path so pycwb can be imported without
# a full pip install (works when running from any directory).
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pyarrow as pa
import pyarrow.parquet as pq

from pycwb.types.trigger import InjectionParams


# ---------------------------------------------------------------------------
# Arrow schema for the injection-only table
# ---------------------------------------------------------------------------
_INJ_SCHEMA = pa.schema([
    pa.field("row_id",    pa.int32()),
    pa.field("injection", InjectionParams.arrow_struct(), nullable=False),
])


def _build_injection_table(records: list[dict]) -> pa.Table:
    """Convert a list of raw injection dicts to an Arrow Table with a struct column."""
    row_ids: list[int] = []
    inj_dicts: list[dict] = []

    for i, raw in enumerate(records):
        inj = InjectionParams.from_injection_dict(raw)
        row_ids.append(i)
        inj_dicts.append(inj.to_dict())

    row_id_arr = pa.array(row_ids, type=pa.int32())
    inj_arr    = pa.array(inj_dicts, type=InjectionParams.arrow_struct())
    return pa.table({"row_id": row_id_arr, "injection": inj_arr},
                    schema=_INJ_SCHEMA)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> list[dict]:
    with open(path) as fh:
        return json.load(fh)


def _write_parquet(table: pa.Table, path: str, compression: str = "snappy") -> None:
    """Write *table* to *path* with row-group statistics (enables predicate pushdown)."""
    pq.write_table(
        table,
        path,
        compression=compression,
        # keep row groups small enough that statistics are useful for filtering
        row_group_size=min(10_000, len(table)),
        write_statistics=True,
        store_schema=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_file(json_path: str, out_dir: str, verbose: bool = True) -> dict:
    """Read *json_path*, build Arrow table, write Parquet, return timing stats."""
    base = os.path.splitext(os.path.basename(json_path))[0]  # e.g. "params_10k"
    out_path = os.path.join(out_dir, f"{base}.parquet")

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Input  : {json_path}")
        print(f"  Output : {out_path}")

    t0 = time.perf_counter()
    records = _load_json(json_path)
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    table = _build_injection_table(records)
    t_build = time.perf_counter() - t1

    os.makedirs(out_dir, exist_ok=True)
    t2 = time.perf_counter()
    _write_parquet(table, out_path)
    t_write = time.perf_counter() - t2

    size_mb = os.path.getsize(out_path) / 1e6

    if verbose:
        print(f"  Rows   : {len(records):>8,}")
        print(f"  Load   : {t_load*1e3:>8.1f} ms")
        print(f"  Build  : {t_build*1e3:>8.1f} ms")
        print(f"  Write  : {t_write*1e3:>8.1f} ms")
        print(f"  Size   : {size_mb:>8.2f} MB  ({out_path})")

    return {
        "json_path":  json_path,
        "parquet_path": out_path,
        "n_rows":     len(records),
        "t_load_ms":  t_load * 1e3,
        "t_build_ms": t_build * 1e3,
        "t_write_ms": t_write * 1e3,
        "size_mb":    size_mb,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write InjectionParams Parquet files.")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(_HERE, "data"),
        help="Directory containing params_*.json files (default: ./data).",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_HERE, "parquet"),
        help="Directory to write .parquet files (default: ./parquet).",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    json_files = sorted(
        f for f in os.listdir(args.data_dir) if f.startswith("params_") and f.endswith(".json")
    )
    if not json_files:
        sys.exit(f"No params_*.json files found in {args.data_dir}. "
                 "Run generate_params.py first.")

    print(f"Writing Parquet files to {args.out_dir}")
    results = []
    for fname in json_files:
        stats = process_file(
            os.path.join(args.data_dir, fname),
            args.out_dir,
            verbose=not args.quiet,
        )
        results.append(stats)

    print(f"\n{'='*60}")
    print(f"{'File':<25} {'Rows':>8} {'Build ms':>10} {'Write ms':>10} {'MB':>8}")
    print(f"{'─'*25} {'─'*8} {'─'*10} {'─'*10} {'─'*8}")
    for r in results:
        name = os.path.basename(r["json_path"])
        print(f"{name:<25} {r['n_rows']:>8,} {r['t_build_ms']:>10.1f} "
              f"{r['t_write_ms']:>10.1f} {r['size_mb']:>8.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
