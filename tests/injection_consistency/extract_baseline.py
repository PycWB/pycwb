#!/usr/bin/env python3
"""extract_baseline.py — Extract core physics values from a pycWB catalog Parquet
into a human-readable, git-diffable ``baseline.json``.

Usage
-----
    cd tests/injection_consistency/
    python extract_baseline.py

The script reads ``catalog/catalog.parquet`` (the existing reference run) and
writes ``baseline.json``.  The JSON file is committed to the repository as the
CI consistency reference; the binary Parquet is NOT committed.

To regenerate the baseline after intentional pipeline output changes, re-run
this script and commit the updated ``baseline.json``.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths (relative to this script's directory)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
CATALOG_PARQUET = HERE / "catalog" / "catalog.parquet"
BASELINE_JSON = HERE / "baseline.json"


# ---------------------------------------------------------------------------
# Core physics columns to extract
# ---------------------------------------------------------------------------
# Grouped by physics domain.  Per-IFO columns are discovered dynamically
# from the ``ifo_list`` stored in the Parquet metadata / row data.
# ---------------------------------------------------------------------------

BOOKKEEPING_COLS = [
    "id", "job_id", "lag_idx", "trial_idx", "cluster_id",
    "event_index", "n_detectors", "hybrid",
]

SNR_CORR_COLS = [
    "rho", "rho_alt", "net_cc", "sky_cc", "subnet_cc", "subnet_cc2",
]

ENERGY_COLS = [
    "likelihood", "coherent_energy", "coherent_energy_norm",
    "net_energy_disb", "net_null", "net_energy", "like_sky", "energy_sky",
]

QUALITY_COLS = [
    "network_sensitivity", "network_alignment_factor", "network_index",
    "packet_norm", "penalty", "cluster_union_size", "strain",
]

PIXEL_COLS = [
    "n_pixels_total", "n_pixels_positive", "n_pixels_core", "sky_size",
]

SKY_COLS = [
    "phi", "theta", "ra", "dec", "phi_det", "theta_det",
    "psi", "iota", "sky_error_regions",
]

CHIRP_COLS = [
    "mchirp", "mchirp_err", "chirp_ellip", "chirp_pfrac", "chirp_efrac", "ebbh",
]

POSTPROC_COLS = ["ifar", "q_veto", "q_factor"]

GPS_TIME_COL = "gps_time"

# Per-IFO fields: {field}_{ifo} — built dynamically
PER_IFO_F64_FIELDS = ["time", "segment_start", "event_start", "event_stop"]
PER_IFO_F32_FIELDS = [
    "left_edge", "right_edge", "duration", "time_lag", "segment_lag",
    "central_freq", "freq_low", "freq_high", "bandwidth", "sample_rate",
    "hrss", "noise_rms", "data_energy", "signal_energy",
    "cross_energy", "null_energy", "residual_energy", "fp", "fx",
]
PER_IFO_FIELDS = PER_IFO_F64_FIELDS + PER_IFO_F32_FIELDS

# Injection struct sub-fields
INJECTION_FIELDS = [
    "name", "hrss", "target_snr", "ra", "dec", "gps_time", "pol",
    "approximant", "snr_sq", "rec_snr_sq", "overlap_snr",
    "d_eff", "fp", "fx", "time", "hrss_det", "parameters",
]

ALL_SCALAR_COLS = (
    BOOKKEEPING_COLS + SNR_CORR_COLS + ENERGY_COLS + QUALITY_COLS +
    PIXEL_COLS + SKY_COLS + CHIRP_COLS + POSTPROC_COLS + [GPS_TIME_COL]
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialise_value(val: Any) -> Any:
    """Convert numpy / Arrow types to JSON-serialisable Python types."""
    if val is None:
        return None
    # bool / np.bool_ MUST come before int — np.bool_ is a subclass of int
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, np.ndarray):
        lst = val.tolist()
        return [
            None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))
            else _serialise_value(v)
            for v in lst
        ]
    if isinstance(val, (list, tuple)):
        return [_serialise_value(v) for v in val]
    if isinstance(val, dict):
        return {str(k): _serialise_value(v) for k, v in val.items()}
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    # Fallback: try direct conversion
    try:
        return float(val) if isinstance(val, (int, float)) else str(val)
    except (TypeError, ValueError):
        return str(val)


def extract_baseline(parquet_path: str | Path) -> dict:
    """Read a pycWB catalog Parquet and return a baseline dict ready for JSON."""
    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()

    # --- Metadata ---
    schema_meta = table.schema.metadata or {}
    pycwb_version = (
        schema_meta.get(b"pycwb_version", b"unknown").decode()
        if schema_meta else "unknown"
    )

    ifo_list: list[str] = []
    if "ifo_list" in df.columns and len(df) > 0:
        first_ifo = df["ifo_list"].iloc[0]
        if hasattr(first_ifo, "tolist"):
            ifo_list = [str(x) for x in first_ifo.tolist()]
        elif isinstance(first_ifo, list):
            ifo_list = [str(x) for x in first_ifo]

    metadata = {
        "source": os.path.basename(str(parquet_path)),
        "config": "config/user_parameters.yaml",
        "n_triggers": len(df),
        "ifo_list": ifo_list,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "pycwb_version": pycwb_version,
    }

    # --- Events ---
    events: list[dict[str, Any]] = []
    for idx in range(len(df)):
        event: dict[str, Any] = {}

        for col in ALL_SCALAR_COLS:
            if col in df.columns:
                event[col] = _serialise_value(df[col].iloc[idx])

        for ifo in ifo_list:
            for field in PER_IFO_FIELDS:
                col_name = f"{field}_{ifo}"
                if col_name in df.columns:
                    event[col_name] = _serialise_value(df[col_name].iloc[idx])

        # Injection struct — flatten sub-fields
        if "injection" in df.columns:
            inj_val = df["injection"].iloc[idx]
            if inj_val is not None and not (
                isinstance(inj_val, float) and np.isnan(inj_val)
            ):
                if isinstance(inj_val, dict):
                    for field in INJECTION_FIELDS:
                        key = f"injection_{field}"
                        event[key] = _serialise_value(inj_val.get(field))
                else:
                    event["injection"] = _serialise_value(inj_val)

        events.append(event)

    return {"metadata": metadata, "events": events}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CATALOG_PARQUET.exists():
        print(f"ERROR: Catalog parquet not found at {CATALOG_PARQUET}", file=sys.stderr)
        print(
            "Run the pipeline first to generate the catalog, then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Reading: {CATALOG_PARQUET}")
    baseline = extract_baseline(CATALOG_PARQUET)

    print(f"  Triggers: {baseline['metadata']['n_triggers']}")
    print(f"  IFOs:     {baseline['metadata']['ifo_list']}")
    print(f"  Version:  {baseline['metadata']['pycwb_version']}")

    with open(BASELINE_JSON, "w") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"Written: {BASELINE_JSON}")
    print("Done. Commit baseline.json to the repository for CI consistency testing.")


if __name__ == "__main__":
    main()
