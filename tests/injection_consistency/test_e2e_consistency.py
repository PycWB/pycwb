"""test_e2e_consistency.py — End-to-end consistency test for the pycWB pipeline.

Runs the full native pipeline with a fixed config and injection seed, then
compares the output against committed baseline values in ``baseline.json``.

Marked ``@pytest.mark.slow`` — skipped in default CI runs.  Run with::

    pytest tests/injection_consistency/test_e2e_consistency.py -v -s
"""
from __future__ import annotations

import numpy as np
import pytest

from .runner import run_pipeline
from .helpers import load_baseline, CORE_COLUMNS, per_ifo_columns
from .comparison import (
    match_triggers_by_gps,
    compare_columns,
    assert_consistency,
    load_new_catalog,
)


# ═══════════════════════════════════════════════════════════════════════════
# GPS matching tolerance
# ═══════════════════════════════════════════════════════════════════════════
GPS_TOL = 0.05  # 50 ms — same as compare_pyc_runs.py


# ═══════════════════════════════════════════════════════════════════════════
# Helper: build the full column list (scalar + per-IFO)
# ═══════════════════════════════════════════════════════════════════════════

def _all_columns(ifo_list: list[str]) -> list[str]:
    """Return all columns to compare: scalar + per-IFO + gps_time."""
    cols = list(CORE_COLUMNS) + per_ifo_columns(ifo_list)
    # Ensure gps_time is included (it's used for matching)
    if "gps_time" not in cols:
        cols.append("gps_time")
    return cols


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_e2e_consistency(tmp_path):
    """Run the full pipeline and compare output against baseline.json."""
    # 1. Run pipeline
    output_catalog = run_pipeline(tmp_path)

    # 2. Load baseline
    baseline_events, metadata = load_baseline()
    ifo_list = metadata.get("ifo_list", [])
    assert len(ifo_list) > 0, "ifo_list missing from baseline metadata"

    # 3. Load new output
    new_df = load_new_catalog(output_catalog, ifo_list)

    # 4. Match triggers by GPS time
    matched, unmatched_base, unmatched_new = match_triggers_by_gps(
        baseline_events, new_df, tol=GPS_TOL,
    )

    # 5. Basic trigger-count sanity
    n_base = len(baseline_events)
    n_new = len(new_df)
    assert n_new > 0, "Pipeline produced zero triggers"
    assert len(unmatched_base) <= max(1, n_base // 2), (
        f"Too many unmatched baseline triggers: {len(unmatched_base)}/{n_base}"
    )
    assert len(unmatched_new) <= max(1, n_new // 2), (
        f"Too many unmatched new triggers: {len(unmatched_new)}/{n_new}"
    )
    assert len(matched) > 0, "No triggers matched between baseline and new run"

    # 6. Column-level consistency
    columns = _all_columns(ifo_list)
    assert_consistency(matched, columns)

    # 7. Print summary for debugging
    stats = compare_columns(matched, columns)
    passed = sum(1 for s in stats.values() if s["passed"])
    print(f"\nConsistency summary: {passed}/{len(stats)} columns passed")
    for col, s in stats.items():
        if not s["passed"]:
            print(f"  FAIL {col}: RMS={s['rms_diff']:.4g} N={s['n']}")


@pytest.mark.slow
def test_trigger_count_consistency(tmp_path):
    """Verify the new run produces roughly the same number of triggers."""
    output_catalog = run_pipeline(tmp_path)
    new_df = load_new_catalog(output_catalog)

    _, metadata = load_baseline()
    n_baseline = metadata.get("n_triggers", 0)
    n_new = len(new_df)

    # Allow ±50% variation (generous — stochastic noise can add/remove triggers)
    if n_baseline > 0:
        ratio = n_new / n_baseline
        assert 0.5 <= ratio <= 2.0, (
            f"Trigger count mismatch: baseline={n_baseline}, new={n_new}, ratio={ratio:.2f}"
        )


@pytest.mark.slow
def test_seed_determinism(tmp_path):
    """Verify the pipeline is deterministic — running twice gives identical output."""
    catalog1 = run_pipeline(tmp_path / "run1")

    # Second run — different tmp subdir to avoid overwrite
    catalog2 = run_pipeline(tmp_path / "run2")

    df1 = load_new_catalog(catalog1)
    df2 = load_new_catalog(catalog2)

    # Same number of triggers
    assert len(df1) == len(df2), (
        f"Non-deterministic trigger count: run1={len(df1)}, run2={len(df2)}"
    )

    # GPS times should match exactly
    if len(df1) > 0:
        gps1 = sorted(df1["gps_time"].tolist())
        gps2 = sorted(df2["gps_time"].tolist())
        for i, (t1, t2) in enumerate(zip(gps1, gps2)):
            assert abs(t1 - t2) < 1e-6, (
                f"GPS time mismatch at index {i}: {t1} vs {t2}"
            )

        # All XGBoost-relevant physics columns must be deterministic.
        # Columns are sourced from the field mapping in
        # examples/pycwb_cwb_consistency/compare_pycwb_vs_cwb.py (SCALAR_MAP
        # + per-IFO sSNR/noise/freq/bandwidth/duration).
        xgb_scalar_cols = [
            "rho", "rho_alt",
            "net_cc", "sky_cc", "subnet_cc",
            "likelihood", "coherent_energy",
            "packet_norm", "penalty",
            "q_veto", "q_factor",
            "phi", "theta", "ra", "dec",
            "network_sensitivity", "network_alignment_factor",
        ]
        # Per-IFO XGBoost features (dynamically discovered from column names)
        per_ifo_suffixes = [
            "signal_energy", "noise_rms",
            "central_freq", "freq_low", "freq_high",
            "bandwidth", "duration",
        ]
        xgb_per_ifo_cols = [
            c for c in df1.columns
            if any(c.endswith(f"_{suf}") for suf in per_ifo_suffixes)
        ]
        xgb_cols = [c for c in xgb_scalar_cols + xgb_per_ifo_cols
                    if c in df1.columns and c in df2.columns]

        for col in xgb_cols:
            vals1 = sorted(df1[col].dropna().tolist())
            vals2 = sorted(df2[col].dropna().tolist())
            if len(vals1) > 0 and len(vals2) > 0:
                rms = float(np.sqrt(np.mean(
                    (np.array(vals1) - np.array(vals2)) ** 2
                )))
                assert rms < 1e-6, (
                    f"Non-deterministic {col}: RMS diff = {rms:.2e}"
                )
