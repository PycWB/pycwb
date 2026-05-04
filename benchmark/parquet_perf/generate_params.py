"""
tests/parquet_perf/generate_params.py
======================================
Generate synthetic injection parameter dicts for WNB, SG, and BBH waveforms
Three sizes: 10k, 50k, 100k, and 1M records, saved as JSON for downstream use.

Usage
-----
    python generate_params.py                  # writes all three sizes (+ 1M)
    python generate_params.py --size 10000     # single size
    python generate_params.py --out-dir ./data # custom output directory

The resulting JSON files are read by write_parquet.py which creates
InjectionParams objects and serialises them to Parquet.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Per-approximant generators
# ---------------------------------------------------------------------------

_WNB_FREQ_RANGE   = (24.0, 1696.0)   # Hz
_WNB_BW_RANGE     = (10.0, 300.0)    # Hz
_WNB_DUR_LOG_RANGE = (math.log10(0.0001), math.log10(0.5))  # log10 seconds
_WNB_HRSS_MIN     = 5e-23
_WNB_HRSS_STEPS   = 3

_SG_CONFIGS = [
    # (frequency_Hz, Q, name_prefix)
    (36,   3.0,   "SGE_Q3"),
    (70,   3.0,   "SGE_Q3"),
    (235,  3.0,   "SGE_Q3"),
    (849,  3.0,   "SGE_Q3"),
    (1615, 3.0,   "SGE_Q3"),
    (70,   9.0,   "SGE_Q9"),
    (100,  9.0,   "SGE_Q9"),
    (235,  9.0,   "SGE_Q9"),
    (361,  9.0,   "SGE_Q9"),
    (36,   9.0,   "SGE_Q9"),
    (48,   9.0,   "SGE_Q9"),
    (153,  9.0,   "SGE_Q9"),
    (554,  9.0,   "SGE_Q9"),
    (849,  9.0,   "SGE_Q9"),
    (1304, 9.0,   "SGE_Q9"),
    (1615, 9.0,   "SGE_Q9"),
    (48,   100.0, "SGE_Q100"),
    (70,   100.0, "SGE_Q100"),
    (235,  100.0, "SGE_Q100"),
    (849,  100.0, "SGE_Q100"),
    (1304, 100.0, "SGE_Q100"),
    (1615, 100.0, "SGE_Q100"),
]

_SG_HRSS_MIN   = 5e-23
_SG_HRSS_STEPS = 7

# BBH mass / spin ranges matching typical O3 searches
_BBH_MASS1_RANGE  = (5.0,   100.0)   # solar masses
_BBH_MASS2_RANGE  = (5.0,   100.0)   # solar masses  (enforced mass1 >= mass2)
_BBH_SPIN_RANGE   = (-0.99, 0.99)
_BBH_DIST_RANGE   = (50.0,  2000.0)  # Mpc
_BBH_APPROXIMANTS = ["IMRPhenomTPHM", "IMRPhenomXPHM", "SEOBNRv4PHM",
                     "NRSur7dq4", "IMRPhenomD"]
_BBH_HRSS_MIN     = 1e-23
_BBH_HRSS_STEPS   = 5


def _make_wnb(n: int, seed: int) -> list[dict]:
    """Return ``n`` WNB injection parameter dicts."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    hrss_choices = [_WNB_HRSS_MIN * (2 ** i) for i in range(_WNB_HRSS_STEPS)]

    records: list[dict] = []
    for i in range(n):
        freq = rng.uniform(*_WNB_FREQ_RANGE)
        bw   = rng.uniform(*_WNB_BW_RANGE)
        dur  = 10 ** rng.uniform(*_WNB_DUR_LOG_RANGE)
        rec: dict = {
            "approximant": "WNB",
            "name":        f"WNB_{i:06d}",
            "frequency":   freq,
            "bandwidth":   bw,
            "duration":    dur,
            "inj_length":  1.0,
            "pseed":       seed + 100000 + i,
            "xseed":       seed + 100001 + i,
            "t_start":     -1.0,
            "t_end":        1.0,
            "mode":         1,
            "hrss":         rng.choice(hrss_choices),
            "ellipticity":  float(np_rng.uniform(0.0, math.pi / 2)),
            "pol":          0.0,
            # sky position – filled by pipeline normally; synthesised here
            "ra":           float(np_rng.uniform(0.0, 2 * math.pi)),
            "dec":          float(np_rng.uniform(-math.pi / 2, math.pi / 2)),
            "gps_time":     float(rng.uniform(1126259462.4, 1369368018.0)),
        }
        records.append(rec)
    return records


def _make_sg(n: int, seed: int) -> list[dict]:
    """Return ``n`` Sine-Gaussian injection parameter dicts."""
    rng = random.Random(seed + 1)
    np_rng = np.random.default_rng(seed + 1)
    hrss_choices = [_SG_HRSS_MIN * (2 ** i) for i in range(_SG_HRSS_STEPS)]
    n_configs = len(_SG_CONFIGS)

    records: list[dict] = []
    for i in range(n):
        cfg_idx = i % n_configs
        freq, q, prefix = _SG_CONFIGS[cfg_idx]
        rec: dict = {
            "approximant": "SGE",
            "name":        f"{prefix}_{int(freq)}Hz_{i:06d}",
            "frequency":   float(freq),
            "Q":           float(q),
            "t_start":     -1.0,
            "t_end":        1.0,
            "ellipticity":  float(np_rng.uniform(0.0, math.pi / 2)),
            "hrss":         rng.choice(hrss_choices),
            "pol":          0.0,
            "ra":           float(np_rng.uniform(0.0, 2 * math.pi)),
            "dec":          float(np_rng.uniform(-math.pi / 2, math.pi / 2)),
            "gps_time":     float(rng.uniform(1126259462.4, 1369368018.0)),
        }
        records.append(rec)
    return records


def _make_bbh(n: int, seed: int) -> list[dict]:
    """Return ``n`` BBH (binary black hole) injection parameter dicts."""
    rng = random.Random(seed + 2)
    np_rng = np.random.default_rng(seed + 2)
    hrss_choices = [_BBH_HRSS_MIN * (2 ** i) for i in range(_BBH_HRSS_STEPS)]

    records: list[dict] = []
    for i in range(n):
        m1 = rng.uniform(*_BBH_MASS1_RANGE)
        m2 = rng.uniform(_BBH_MASS2_RANGE[0], m1)  # m1 >= m2
        approx = rng.choice(_BBH_APPROXIMANTS)
        rec: dict = {
            "approximant":   approx,
            "name":          f"BBH_{approx}_{i:06d}",
            "mass1":         m1,
            "mass2":         m2,
            "spin1x":        float(np_rng.uniform(*_BBH_SPIN_RANGE)),
            "spin1y":        float(np_rng.uniform(*_BBH_SPIN_RANGE)),
            "spin1z":        float(np_rng.uniform(*_BBH_SPIN_RANGE)),
            "spin2x":        float(np_rng.uniform(*_BBH_SPIN_RANGE)),
            "spin2y":        float(np_rng.uniform(*_BBH_SPIN_RANGE)),
            "spin2z":        float(np_rng.uniform(*_BBH_SPIN_RANGE)),
            "distance":      rng.uniform(*_BBH_DIST_RANGE),
            "inclination":   float(np_rng.uniform(0.0, math.pi)),
            "polarization":  float(np_rng.uniform(0.0, math.pi)),
            "coa_phase":     float(np_rng.uniform(0.0, 2 * math.pi)),
            "hrss":          rng.choice(hrss_choices),
            "ra":            float(np_rng.uniform(0.0, 2 * math.pi)),
            "dec":           float(np_rng.uniform(-math.pi / 2, math.pi / 2)),
            "gps_time":      float(rng.uniform(1126259462.4, 1369368018.0)),
            "pol":           float(np_rng.uniform(0.0, math.pi)),
        }
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------

def generate_mixed(n_total: int, seed: int = 42) -> list[dict]:
    """Generate ``n_total`` records split evenly (1/3 each) between WNB, SG, BBH.

    The three sub-lists are interleaved so that the resulting list is uniformly
    mixed (useful for row-group-level statistics benchmarks).
    """
    per_type = n_total // 3
    remainder = n_total - per_type * 3

    wnb = _make_wnb(per_type,              seed)
    sg  = _make_sg (per_type,              seed)
    bbh = _make_bbh(per_type + remainder,  seed)

    # Interleave: WNB[0], SG[0], BBH[0], WNB[1], SG[1], BBH[1], …
    mixed: list[dict] = []
    it_wnb = iter(wnb)
    it_sg  = iter(sg)
    it_bbh = iter(bbh)
    for w, s, b in zip(it_wnb, it_sg, it_bbh):
        mixed.extend([w, s, b])
    # any BBH remainder (when n_total % 3 != 0)
    for b in it_bbh:
        mixed.append(b)

    return mixed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_SIZES = [10_000, 50_000, 100_000, 1_000_000]


def _write(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(records, fh)
    mb = os.path.getsize(path) / 1e6
    print(f"  wrote {len(records):>7,} records  →  {path}  ({mb:.1f} MB)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic injection parameters.")
    parser.add_argument("--size",    type=int, nargs="*",
                        default=None,
                        help="One or more sizes to generate (default: 10000 50000 100000).")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "data"),
                        help="Directory to write JSON files.")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Global random seed.")
    args = parser.parse_args(argv)

    sizes = args.size if args.size else _SIZES

    print(f"Generating injection parameter sets → {args.out_dir}")
    for n in sizes:
        tag = f"{n // 1000}k"
        path = os.path.join(args.out_dir, f"params_{tag}.json")
        recs = generate_mixed(n, seed=args.seed)
        _write(recs, path)
    print("Done.")


if __name__ == "__main__":
    main()
