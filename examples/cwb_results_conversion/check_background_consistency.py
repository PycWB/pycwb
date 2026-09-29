"""Cross-check identical ROOT triggers through the shared background processor.

Example: python check_background_consistency.py --wave wave.root --ifo L1 H1
    --root-python /path/to/root/env/bin/python --output check
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from pycwb.modules.postprocess.background import process_background
from pycwb.modules.postprocess.root_adapter import read_cwb_root
from pycwb.modules.postprocess.selection import trigger_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", nargs="+", required=True)
    parser.add_argument("--live", nargs="+")
    parser.add_argument("--ifo", nargs="+", required=True)
    parser.add_argument("--root-python", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rho-index", type=int, choices=[0, 1], default=0)
    parser.add_argument("--root-cut", default="1")
    parser.add_argument("--query", default=None)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    waves = [str(Path(p).resolve()) for p in args.wave]
    lives = [str(Path(p).resolve()) for p in (args.live or args.wave)]
    adapted = read_cwb_root(waves, args.ifo, live_files=lives)
    paths = adapted.write(out / "adapted")
    ranking = "rho" if args.rho_index == 0 else "rho_alt"
    # Include exact event values to exercise the strict/inclusive boundary.
    values = adapted.triggers[ranking].to_numpy()
    thresholds = np.unique(
        np.r_[np.arange(0, 31, 0.1), values[:: max(1, len(values) // 20)]]
    ).tolist()
    request = {
        "wave_files": waves,
        "live_files": lives,
        "nifo": len(args.ifo),
        "root_cut": args.root_cut,
        "rho_index": args.rho_index,
        "thresholds": thresholds,
    }
    (out / "request.json").write_text(json.dumps(request, indent=2))
    with (out / "root.log").open("w") as log:
        subprocess.run(
            [
                args.root_python,
                str(Path(__file__).with_name("root_background_reference.py")),
                "--request",
                str(out / "request.json"),
                "--output",
                str(out / "reference.json"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    reference = json.loads((out / "reference.json").read_text())
    kwargs = {
        "ranking_par": ranking,
        "thresholds": thresholds,
        "comparison": ">",
        "trigger_query": args.query,
    }
    memory = process_background(adapted.triggers, adapted.progress, **kwargs)
    disk = process_background(paths["catalog_file"], paths["progress_file"], **kwargs)
    pd.testing.assert_frame_equal(memory["curve"], disk["curve"])
    actual = memory["triggers"]
    selected = sorted(zip(actual.root_file, actual.root_entry))
    assert selected == sorted(tuple(r) for r in reference["selected"]), (
        "Selected event membership differs"
    )
    np.testing.assert_allclose(memory["livetime"], reference["livetime"], rtol=1e-12)
    np.testing.assert_array_equal(memory["curve"]["count"], reference["counts"])
    np.testing.assert_allclose(memory["curve"].far_hz, reference["far_hz"], rtol=1e-12)
    # Exercise the existing file-based selection action too.
    selection = trigger_selection(
        ".",
        paths["catalog_file"],
        paths["progress_file"],
        trigger_filter={"query": args.query} if args.query else None,
    )
    assert len(selection["triggers"]) == len(actual)
    np.testing.assert_allclose(
        selection["livetime"]["seconds"], memory["livetime"], rtol=1e-12
    )
    memory["curve"].to_csv(out / "curve.csv", index=False)
    actual[["root_file", "root_entry", "job_id", "lag_idx", ranking]].to_csv(
        out / "selected.csv", index=False
    )
    hashes = {}
    for path in dict.fromkeys(waves + lives):
        with open(path, "rb") as source:
            hashes[path] = hashlib.file_digest(source, "sha256").hexdigest()
    summary = {
        "status": "passed",
        "scope": "background membership, livetime, cumulative counts and FAR; explicit cuts only",
        "root_version": reference["root_version"],
        "ifo": args.ifo,
        "ranking": ranking,
        "root_cut": args.root_cut,
        "query": args.query,
        "events": adapted.triggers.num_rows,
        "selected": len(actual),
        "exposure_rows": adapted.progress.num_rows,
        "background_seconds": memory["livetime"],
        "thresholds": len(thresholds),
        "comparison": ">",
        "input_sha256": hashes,
        "limitations": [
            "Not a full cWB report macro run",
            "No simulation efficiency, external veto or model-training validation",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
