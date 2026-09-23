"""Independent PyROOT reference for cWB report selection/counting semantics.

Run in a ROOT environment. Does not import pycWB, its adapter or its metrics.
Implements cwb_report_prod_2.C's all-background lag exclusion and strict rho
threshold counting. Additional report cuts must be supplied explicitly.
"""

import argparse
import json
from pathlib import Path

import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    nifo = request["nifo"]
    background = f"!(lag[{nifo}]==0 && slag[{nifo}]==0)"
    cut = f"({background}) && ({request['root_cut']})"
    counts = [0] * len(request["thresholds"])
    selected = []
    for path in request["wave_files"]:
        source = ROOT.TFile.Open(path)
        tree = source.Get("waveburst")
        tree.SetEstimate(tree.GetEntries() + 1)
        n = tree.Draw("Entry$", cut, "goff")
        if n < 0:
            raise RuntimeError(f"ROOT rejected cut: {cut}")
        selected.extend(
            [str(Path(path).resolve()), int(tree.GetV1()[i])] for i in range(n)
        )
        for i, threshold in enumerate(request["thresholds"]):
            selection = f"({cut}) && rho[{request['rho_index']}]>{threshold:.17g}"
            count = int(tree.GetEntries(selection))
            if count < 0:
                raise RuntimeError(f"ROOT rejected cut: {selection}")
            counts[i] += count
        source.Close()
    livetime = 0.0
    for path in request["live_files"]:
        source = ROOT.TFile.Open(path)
        tree = source.Get("liveTime")
        tree.SetEstimate(tree.GetEntries() + 1)
        n = tree.Draw("live", background, "goff")
        if n < 0:
            raise RuntimeError("ROOT rejected liveTime selection")
        livetime += sum(float(tree.GetV1()[i]) for i in range(n))
        source.Close()
    Path(args.output).write_text(
        json.dumps(
            {
                "root_version": ROOT.gROOT.GetVersion(),
                "selected": selected,
                "counts": counts,
                "livetime": livetime,
                "far_hz": [c / livetime for c in counts],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
