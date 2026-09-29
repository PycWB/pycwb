"""Compare shared-processor products against actual cWB report tables."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pycwb.modules.postprocess.background import process_background
from pycwb.post_production.action_spec import action_spec


@action_spec(
    inputs=["catalog_file", "progress_file", "reference_dir", "efficiency_file"],
    outputs=["output_file"],
    description="Check rates and efficiencies against a cWB report",
)
def compare_cwb_report(
    work_dir,
    output_file,
    catalog_file=None,
    progress_file=None,
    reference_dir=None,
    efficiency_file=None,
    simulation_reference_dir=None,
    ranking_par="rho_alt",
    **kwargs,
):
    """Compare counts; allow only the reference text's printed precision.

    Reference directories must be the data subdirectories of cwb_report.
    FAR values are printed to six significant digits and amplitudes to seven.
    The caller supplies matching selections and the complete injection truth.
    A failed check is recorded in JSON, then raises to stop the workflow.
    """
    def resolve(path: str) -> Path:
        return Path(work_dir) / path
    checks = []
    resolve(output_file).parent.mkdir(parents=True, exist_ok=True)
    if catalog_file:
        ref = np.loadtxt(resolve(reference_dir) / "far_rho.txt", ndmin=2)
        result = process_background(
            resolve(catalog_file),
            resolve(progress_file),
            ranking_par=ranking_par,
            thresholds=ref[:, 0],
            comparison=">",
        )
        actual = result["curve"]
        expected = np.rint(ref[:, 1] * result["livetime"]).astype(int)
        difference = np.abs(actual["count"].to_numpy() - expected)
        ok = bool(np.all(difference == 0))
        checks.append(
            {
                "check": "Background cumulative counts",
                "status": "PASS" if ok else "FAIL",
                "details": f"{len(ref)} thresholds; {int(np.sum(difference != 0))} disagree; livetime {result['livetime']:.9g} s",
                "thresholds": len(ref),
                "mismatched": int(np.sum(difference != 0)),
                "livetime": result["livetime"],
            }
        )
        actual.to_csv(resolve(output_file).with_suffix(".background.csv"), index=False)
    if efficiency_file:
        native = pd.read_csv(resolve(efficiency_file))
        npoints = 0
        mismatches = []
        paths = list(resolve(simulation_reference_dir).glob("eff_*.txt"))
        refs = {p.stem[4:]: p for p in paths}
        if set(refs) != set(native.waveform):
            mismatches.append("Missing reference waveform tables")
        for name, group in native.groupby("waveform"):
            if name not in refs:
                continue
            reference = np.loadtxt(refs[name], ndmin=2)
            group = group.sort_values("hrss")
            reference = reference[np.argsort(reference[:, 0])]
            npoints += len(reference)
            ok = (
                len(reference) == len(group)
                and np.allclose(reference[:, 0], group.hrss, rtol=6e-7, atol=0)
                and np.array_equal(reference[:, 1], group.n_detected)
                and np.array_equal(reference[:, 2], group.n_injected)
                and np.allclose(
                    reference[:, 3], group.efficiency, atol=0.000501, rtol=0
                )
            )
            if not ok:
                mismatches.append(name)
        checks.append(
            {
                "check": "Injection efficiency counts",
                "status": "PASS" if not mismatches else "FAIL",
                "details": f"{len(refs)} waveforms, {npoints} amplitude bins; mismatches: {mismatches}",
                "waveforms": len(refs),
                "amplitude_bins": npoints,
                "mismatches": mismatches,
            }
        )
    if not checks:
        raise ValueError("Supply background or simulation products to compare")
    result = {
        "description": "Comparison with tables written by the standard cWB report command.",
        "checks": checks,
    }
    path = resolve(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    if any(c["status"] == "FAIL" for c in checks):
        raise ValueError(f"cWB report comparison failed; see {path}")
    return result


@action_spec(
    inputs=["catalog_file", "far_file"],
    outputs=["output_file"],
    description="Attach IFAR with the cWB report step-curve convention",
)
def attach_cwb_ifar(
    work_dir, catalog_file, far_file, output_file, ranking_par="rhor", **kwargs
):
    """Use cWB setIFAR's step graph, including tiny linear boundary ramps.

    This explicit compatibility action reads a two-column FAR table (Hz),
    drops zero-rate bins, clamps outside the range, and stores IFAR seconds.
    Ranking and IFAR are rounded to ROOT float precision. It does not replace
    the native empirical-tail calibration or assert infinite tail exposure.
    """
    frame = pd.read_parquet(Path(work_dir) / catalog_file)
    curve = np.loadtxt(Path(work_dir) / far_file, ndmin=2)
    if (
        curve.shape[1] < 2
        or not np.isfinite(curve[:, :2]).all()
        or np.any(curve[:, 1] < 0)
    ):
        raise ValueError("Expected finite thresholds and nonnegative FAR values")
    curve = curve[curve[:, 1] > 0]
    if (
        not len(curve)
        or np.any(np.diff(curve[:, 0]) <= 0)
        or np.any(np.diff(curve[:, 1]) > 0)
    ):
        raise ValueError(
            "FAR must be nonincreasing on increasing thresholds, with positive exposure"
        )
    values = frame[ranking_par].to_numpy(dtype="float32").astype(float)
    if not np.isfinite(values).all():
        raise ValueError("Ranking values must be finite")
    if len(curve) == 1:
        far = np.full(len(frame), curve[0, 1])
    else:
        dx = np.diff(curve[:, 0]) / 100000.0
        x = np.column_stack([curve[1:, 0] - dx, curve[1:, 0] + dx]).ravel()
        y = np.column_stack([curve[:-1, 1], curve[1:, 1]]).ravel()
        far = np.interp(values, x, y, left=curve[0, 1], right=curve[-1, 1])
    frame["far_hz"] = far
    frame["ifar"] = (1 / far).astype("float32")
    output = Path(work_dir) / output_file
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return {
        "output_file": str(output),
        "n_events": len(frame),
        "calibration": "cWB report step graph",
        "ifar_units": "seconds",
    }


@action_spec(
    inputs=["scored_file", "reference_file", "independent_scored_file"],
    outputs=["output_file"],
    description="Compare event membership, scores and IFAR with scored cWB ROOT events",
)
def compare_cwb_scores(
    work_dir: str,
    scored_file: str,
    reference_file: str,
    output_file: str,
    independent_scored_file: str | None = None,
    time_window: float | None = None,
    **kwargs: object,
) -> dict:
    """Write membership, score and IFAR comparisons; time_window is in GPS seconds."""
    import uproot

    native = pd.read_parquet(Path(work_dir) / scored_file)
    with uproot.open(Path(work_dir) / reference_file) as root:
        branches = ["time", "rho", "lag", "run", "eventID"]
        if "ifar" in root["waveburst"]:
            branches.append("ifar")
        arrays = root["waveburst"].arrays(branches, library="np")
    nifo = len(arrays["lag"][0]) - 1 if len(arrays["lag"]) else 2
    ref = pd.DataFrame(
        {
            "gps_time": arrays["time"][:, 0],
            "root_run": arrays["run"],
            "lag_idx": arrays["lag"][:, nifo],
            "cluster_id": arrays["eventID"][:, 0],
            "rhor_root": arrays["rho"][:, 1],
            "prob_root": arrays["rho"][:, 2],
        }
    )
    if "ifar" in arrays:
        ref["ifar_root"] = arrays["ifar"]
    if time_window is not None:
        ref = ref[np.abs(arrays["time"][:, 0] - arrays["time"][:, nifo]) <= time_window]
    # The first detector's time has a portable canonical alias, gps_time.
    if "root_run" not in native:
        native["root_run"] = native.job_id
    both = native.merge(
        ref,
        on=["gps_time", "root_run", "lag_idx", "cluster_id"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    checks = []

    def check(name, ok, detail):
        checks.append(
            {"check": name, "status": "PASS" if ok else "FAIL", "details": detail}
        )

    matched = both[both._merge == "both"]
    check(
        "Scored event membership",
        bool((both._merge == "both").all()),
        f"{len(native)} native; {len(ref)} cWB; {len(matched)} common events",
    )
    for label, native_column, ref_column in [
        ("Classification probability", "xgb_prob", "prob_root"),
        ("Ranking at ROOT precision", "rhor", "rhor_root"),
        ("IFAR seconds", "ifar", "ifar_root"),
    ]:
        if ref_column not in matched or native_column not in matched:
            continue
        a = matched[native_column].to_numpy(dtype="float32")
        b = matched[ref_column].to_numpy(dtype="float32")
        check(
            label,
            bool(np.array_equal(a, b)),
            f"{len(a)} events; max absolute difference {float(np.max(abs(a - b))) if len(a) else 0:g}",
        )
    if independent_scored_file:
        second = pd.read_parquet(Path(work_dir) / independent_scored_file)
        comparison = native[["id", "xgb_prob", "rhor"]].merge(
            second[["id", "xgb_prob", "rhor"]],
            on="id",
            how="outer",
            suffixes=("_reference", "_native"),
            validate="one_to_one",
        )
        for col in ["xgb_prob", "rhor"]:
            a = comparison[col + "_reference"].to_numpy()
            b = comparison[col + "_native"].to_numpy()
            check(
                "Independent training: " + col,
                bool(np.array_equal(a, b)),
                f"{len(comparison)} events; max absolute difference {float(np.nanmax(abs(a - b))) if len(a) else 0:g}",
            )
    output = Path(work_dir) / output_file
    output.parent.mkdir(parents=True, exist_ok=True)
    both.to_parquet(output.with_suffix(".parquet"), index=False)
    result = {
        "description": "Event-level comparison with standard cWB XGBoost / setIFAR output.",
        "checks": checks,
    }
    output.write_text(json.dumps(result, indent=2))
    if any(x["status"] == "FAIL" for x in checks):
        raise ValueError(f"cWB score comparison failed; see {output}")
    return result


@action_spec(
    inputs=["comparison_files"],
    outputs=["output_file"],
    description="Collect independently recorded consistency checks",
)
def collect_comparisons(work_dir: str, comparison_files: list[str], output_file: str, notes: list[str] | None = None, **kwargs: object) -> dict:
    """Combine recorded JSON checks without rerunning analyses; write and return the report."""
    checks = []
    for name in comparison_files:
        result = json.loads((Path(work_dir) / name).read_text())
        for check in result["checks"]:
            checks.append({**check, "check": f"{Path(name).stem}: {check['check']}"})
    result = {
        "description": "Same cWB triggers and injection truth processed through standard cWB commands and the pycWB YAML workflow.",
        "checks": checks,
        "notes": notes or [],
    }
    path = Path(work_dir) / output_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return result
