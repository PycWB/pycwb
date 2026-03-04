"""
convert_root_to_parquet.py — Convert cWB waveburst ROOT files to flat Parquet.

Reads BKG and/or SIM ROOT files via uproot, extracts XGBoost-required branches
as flat scalar columns, and saves Snappy-compressed Parquet files.

Key details:
  - Per-IFO branches (rho, netcc, sSNR, …) → rho0, rho1, …  (capped at nifo)
  - Non-IFO branches (Lveto, Qveto)         → Lveto0, Lveto1, … / Qveto0, …
    (all elements extracted from root array, indices independent of IFO count)

Usage:
    python convert_root_to_parquet.py \\
        --bkg wave_O4_K17_C00_LH_BurstLF_BKG_run1.M2.root \\
        --sim wave_O4_K17_C00_LH_BurstLF_SIM_Training_Set2_run1.M1.root \\
        --nifo 2 \\
        --bkg-out bkg_xgb.parquet \\
        --sim-out sim_xgb.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys

# ── use local pyBurst source ───────────────────────────────────────────────────
sys.path.insert(0, "/Users/yumengxu/Project/Physics/cwb/pyBurst")

import numpy as np
import pandas as pd
import uproot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── branch configuration ───────────────────────────────────────────────────────

# One scalar value per event
SCALAR_BRANCHES = [
    "likelihood",   # network likelihood
    "ecor",         # coherent energy
    "norm",         # energy normalisation
    "penalty",      # penalty factor
]

# Array branches: per-IFO → colN for n in range(nifo)
ARRAY_BRANCHES_PER_IFO = [
    "rho",          # per-IFO SNR  (rho[0] = rho0_std for XGBoost)
    "netcc",        # per-IFO network correlation coefficient
    "sSNR",         # per-IFO signal SNR
    "noise",        # per-IFO noise RMS
    "duration",     # per-IFO duration
    "bandwidth",    # per-IFO bandwidth
    "frequency",    # per-IFO central frequency
    "chirp",        # chirp estimators (chirp[1], chirp[3] for bbh)
]

# Array branches whose indices are NOT tied to IFO count — extract ALL elements
ARRAY_BRANCHES_ALL = [
    "Qveto",        # Qveto[0] = Qa², Qveto[1] used for Qp
    "Lveto",        # Lveto[2] = Lveto2 feature for blf/bhf/bld
]

GPS_BRANCH = "gps"


# ─────────────────────────────────────────────────────────────────────────────
def _unpack_array(arr: np.ndarray, n_cols: int, name: str) -> dict[str, np.ndarray]:
    """Unpack a 1-D object array or 2-D fixed array into {col0: …, col1: …, …}."""
    out: dict[str, np.ndarray] = {}
    if arr.ndim == 1 and arr.dtype == object:
        for i in range(n_cols):
            out[f"{name}{i}"] = np.array(
                [float(row[i]) if len(row) > i else np.nan for row in arr],
                dtype=np.float32,
            )
    elif arr.ndim == 2:
        for i in range(min(n_cols, arr.shape[1])):
            out[f"{name}{i}"] = arr[:, i].astype(np.float32)
    return out


def tree_to_dataframe(root_file: str, tree_name: str,
                      nifo: int, classifier: int) -> pd.DataFrame:
    """Read a waveburst TTree and return a flat XGBoost-ready DataFrame.

    Parameters
    ----------
    root_file  : path to the ROOT file
    tree_name  : TTree name (usually "waveburst")
    nifo       : number of detectors — per-IFO columns are capped at this
    classifier : 0 = background, 1 = signal
    """
    all_requested = SCALAR_BRANCHES + ARRAY_BRANCHES_PER_IFO + ARRAY_BRANCHES_ALL + [GPS_BRANCH]

    log.info("Opening %s", root_file)
    with uproot.open(f"{root_file}:{tree_name}") as tree:
        available = set(tree.keys())
        to_read   = [b for b in all_requested if b in available]
        missing   = [b for b in all_requested if b not in available]
        if missing:
            log.warning("Branches not found (skipped): %s", missing)
        raw = tree.arrays(to_read, library="np")

    n_events = len(raw[to_read[0]])
    log.info("  %d events found", n_events)
    rows: dict[str, np.ndarray] = {}

    # ── scalar branches ────────────────────────────────────────────────────────
    for branch in SCALAR_BRANCHES:
        if branch in raw:
            rows[branch] = raw[branch].astype(np.float32)

    # ── GPS reference time ─────────────────────────────────────────────────────
    if GPS_BRANCH in raw:
        gps_arr = raw[GPS_BRANCH]
        if gps_arr.ndim == 1 and gps_arr.dtype == object:
            rows["gps_time"] = np.array(
                [float(g[0]) if len(g) > 0 else np.nan for g in gps_arr],
                dtype=np.float64,
            )
        elif gps_arr.ndim == 2:
            rows["gps_time"] = gps_arr[:, 0].astype(np.float64)
        else:
            rows["gps_time"] = gps_arr.astype(np.float64)

    # ── per-IFO array branches → colN for n in range(nifo) ────────────────────
    for branch in ARRAY_BRANCHES_PER_IFO:
        if branch not in raw:
            continue
        rows.update(_unpack_array(raw[branch], nifo, branch))
    # rho0_std alias expected by preprocess_events when rho0(define)==0
    if "rho0" in rows:
        rows["rho0_std"] = rows["rho0"].copy()

    # ── non-IFO array branches → colN for all available elements ──────────────
    for branch in ARRAY_BRANCHES_ALL:
        if branch not in raw:
            continue
        arr = raw[branch]
        # auto-detect actual element count from the data
        if arr.ndim == 1 and arr.dtype == object:
            n_cols = int(max(len(r) for r in arr[:100]))
        elif arr.ndim == 2:
            n_cols = arr.shape[1]
        else:
            n_cols = 0
        log.info("  %s: %d elements per event", branch, n_cols)
        rows.update(_unpack_array(arr, n_cols, branch))

    # ── classifier label ───────────────────────────────────────────────────────
    rows["classifier"] = np.full(n_events, classifier, dtype=np.int8)

    df = pd.DataFrame(rows)
    log.info("  DataFrame: %d rows × %d columns", len(df), len(df.columns))
    return df


# ─────────────────────────────────────────────────────────────────────────────
def convert(root_file: str, out_parquet: str, nifo: int, classifier: int) -> None:
    df = tree_to_dataframe(root_file, "waveburst", nifo, classifier)
    df.to_parquet(out_parquet, index=False, compression="snappy")
    size_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    log.info("Saved → %s  (%.1f MB in-memory, snappy-compressed)", out_parquet, size_mb)
    log.info("Columns: %s", list(df.columns))


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Convert cWB ROOT → Parquet for XGBoost")
    parser.add_argument("--bkg",     default="wave_O4_K17_C00_LH_BurstLF_BKG_run1.M2.root",
                        help="BKG ROOT file (classifier=0)")
    parser.add_argument("--sim",     default="wave_O4_K17_C00_LH_BurstLF_SIM_Training_Set2_run1.M1.root",
                        help="SIM/TrainingSet ROOT file (classifier=1)")
    parser.add_argument("--nifo",    type=int, default=2,
                        help="Number of detectors")
    parser.add_argument("--bkg-out", default="bkg_xgb.parquet",
                        help="Output file for BKG")
    parser.add_argument("--sim-out", default="sim_xgb.parquet",
                        help="Output file for SIM")
    parser.add_argument("--skip-bkg", action="store_true",
                        help="Skip BKG conversion")
    parser.add_argument("--skip-sim", action="store_true",
                        help="Skip SIM conversion")
    args = parser.parse_args()

    if not args.skip_bkg:
        log.info("=== Converting BKG ===")
        convert(args.bkg, args.bkg_out, args.nifo, classifier=0)

    if not args.skip_sim:
        log.info("=== Converting SIM ===")
        convert(args.sim, args.sim_out, args.nifo, classifier=1)

    log.info("Done.")


if __name__ == "__main__":
    main()
