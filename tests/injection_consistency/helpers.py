"""helpers.py — Baseline loader, column definitions, and numerical tolerances
for the pycWB e2e consistency test.

The core reference is ``baseline.json`` (committed, human-readable).
The binary ``catalog.parquet`` is NOT used by this module.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
BASELINE_JSON = HERE / "baseline.json"


# ---------------------------------------------------------------------------
# Load baseline
# ---------------------------------------------------------------------------

def load_baseline(path: str | Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the baseline JSON and return (events, metadata).

    Parameters
    ----------
    path : str or Path, optional
        Path to baseline.json.  Defaults to ``tests/injection_consistency/baseline.json``.

    Returns
    -------
    events : list[dict]
        List of per-event dicts.  Each dict has ``gps_time`` and all physics columns.
    metadata : dict
        Metadata dict with ``ifo_list``, ``n_triggers``, ``pycwb_version``, etc.
    """
    p = Path(path) if path else BASELINE_JSON
    with open(p) as f:
        data = json.load(f)
    return data.get("events", []), data.get("metadata", {})


# ---------------------------------------------------------------------------
# Core columns for consistency comparison
# ---------------------------------------------------------------------------
# These MUST match the columns extracted by ``extract_baseline.py``.
# The lists are grouped for readability; the flattened CORE_COLUMNS
# tuple is what the test iterates over.
# ---------------------------------------------------------------------------

_BOOKKEEPING = [
    "job_id", "lag_idx", "trial_idx", "cluster_id",
    "event_index", "n_detectors", "hybrid",
]

_SNR_CORR = [
    "rho", "rho_alt", "net_cc", "sky_cc", "subnet_cc", "subnet_cc2",
]

_ENERGY = [
    "likelihood", "coherent_energy", "coherent_energy_norm",
    "net_energy_disb", "net_null", "net_energy", "like_sky", "energy_sky",
]

_QUALITY = [
    "network_sensitivity", "network_alignment_factor", "network_index",
    "packet_norm", "penalty", "cluster_union_size", "strain",
]

_PIXEL = [
    "n_pixels_total", "n_pixels_positive", "n_pixels_core", "sky_size",
]

_SKY = [
    "phi", "theta", "ra", "dec", "phi_det", "theta_det",
    "psi", "iota",
]

_CHIRP = [
    "mchirp", "mchirp_err", "chirp_ellip", "chirp_pfrac", "chirp_efrac",
]

_POSTPROC = ["ifar", "q_veto", "q_factor"]

# Per-IFO fields  (suffix _{ifo} appended at runtime)
_PER_IFO_FIELDS = [
    "time", "segment_start", "event_start", "event_stop",
    "left_edge", "right_edge", "duration", "time_lag", "segment_lag",
    "central_freq", "freq_low", "freq_high", "bandwidth", "sample_rate",
    "hrss", "noise_rms", "data_energy", "signal_energy",
    "cross_energy", "null_energy", "residual_energy", "fp", "fx",
]

# Flattened list of all scalar columns (no per-IFO, no injection)
CORE_COLUMNS = tuple(
    _BOOKKEEPING + _SNR_CORR + _ENERGY + _QUALITY +
    _PIXEL + _SKY + _CHIRP + _POSTPROC
)


def per_ifo_columns(ifo_list: list[str]) -> list[str]:
    """Return per-IFO column names for the given IFO list.

    Example
    -------
    >>> per_ifo_columns(["L1", "H1"])
    ['time_L1', 'time_H1', 'segment_start_L1', ..., 'fx_H1']
    """
    cols = []
    for ifo in ifo_list:
        for field in _PER_IFO_FIELDS:
            cols.append(f"{field}_{ifo}")
    return cols


# ---------------------------------------------------------------------------
# Numerical tolerances
# ---------------------------------------------------------------------------
# Each column has an ``abs`` (absolute) and ``rel`` (relative) tolerance.
# The test checks that the RMS difference between baseline and new-run values
# is within:  rms_diff <= max(abs_tol, rel_tol * abs(baseline_mean)).
#
# Initial values below are intentionally **wide** — they will be tightened
# after the first pipeline run establishes the actual RMS noise floor.
# ---------------------------------------------------------------------------

def _default_tol(abs_tol: float = 0.0, rel_tol: float = 0.10) -> dict[str, float]:
    return {"abs": abs_tol, "rel": rel_tol}


#: Per-column numeric tolerances: ``{column_name: {"abs": ..., "rel": ...}}``.
#: Columns NOT listed here are compared with ``abs=0, rel=0.10`` (10%).
NUMERIC_TOLERANCE: dict[str, dict[str, float]] = {
    # Bookkeeping — exact match expected
    "job_id":         _default_tol(abs_tol=0, rel_tol=0),
    "lag_idx":        _default_tol(abs_tol=0, rel_tol=0),
    "trial_idx":      _default_tol(abs_tol=0, rel_tol=0),
    "n_detectors":    _default_tol(abs_tol=0, rel_tol=0),
    "hybrid":         _default_tol(abs_tol=0, rel_tol=0),

    # GPS time — exact match within floating precision
    "gps_time":       _default_tol(abs_tol=1e-6, rel_tol=0),

    # SNR / correlation — ~5-10% Gaussian noise floor
    "rho":            _default_tol(abs_tol=0.5, rel_tol=0.10),
    "rho_alt":        _default_tol(abs_tol=0.5, rel_tol=0.10),
    "net_cc":         _default_tol(abs_tol=0.02, rel_tol=0.05),
    "sky_cc":         _default_tol(abs_tol=0.02, rel_tol=0.05),
    "subnet_cc":      _default_tol(abs_tol=0.02, rel_tol=0.05),

    # Energy / likelihood
    "likelihood":     _default_tol(abs_tol=100, rel_tol=0.15),
    "coherent_energy": _default_tol(abs_tol=100, rel_tol=0.15),

    # Sky angles — 1 degree absolute
    "phi":            _default_tol(abs_tol=1.0, rel_tol=0.02),
    "theta":          _default_tol(abs_tol=1.0, rel_tol=0.02),
    "ra":             _default_tol(abs_tol=1.0, rel_tol=0.02),
    "dec":            _default_tol(abs_tol=1.0, rel_tol=0.02),
}


def get_tolerance(col: str) -> dict[str, float]:
    """Return the tolerance dict for a column, falling back to default."""
    if col in NUMERIC_TOLERANCE:
        return dict(NUMERIC_TOLERANCE[col])
    # Per-IFO fallback
    for ifo_field in _PER_IFO_FIELDS:
        if col.endswith(f"_{ifo_field}"):
            return _default_tol(abs_tol=0.0, rel_tol=0.10)
    return _default_tol(abs_tol=0.0, rel_tol=0.10)
