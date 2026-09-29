"""Prepare independent tutorial configurations from the maintained CLI demo.

Run from the repository root: python examples/tutorials/prepare.py
This writes inputs only; it neither downloads detector data nor runs searches.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import healpy as hp
import numpy as np
import yaml


def prepare(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1] / "demo/user_parameters.yaml"
    base = yaml.safe_load(source.read_text())
    base.update(save_sky_map=True, plot_sky_map=False, plot_waveform=False)
    # Stable source identities let postproduction match truth across runs.
    base["injection"]["parameters"].update(sim_idx=0, trial_idx=0)
    configurations = {"all_sky": deepcopy(base)}
    center = {"ra": "1 rad", "dec": "0.3 rad"}
    masks = {
        "fixed": {"type": "Fixed", "coordsys": "icrs", "coordinates": center},
        "patch": {"type": "Patch", "coordsys": "icrs",
                  "patch": {"center": center, "radius": "15 deg"}},
        "offset": {"type": "Patch", "coordsys": "icrs",
                   "patch": {"center": {"ra": "3 rad", "dec": "-0.5 rad"},
                             "radius": "15 deg"}},
    }
    # A binary ICRS region, not a posterior probability distribution.
    region = np.zeros(hp.nside2npix(16))
    pixels = hp.query_disc(16, hp.ang2vec(np.pi / 2 - 0.3, 1.0), np.deg2rad(15))
    region[pixels] = 1
    map_path = destination / "region.fits"
    hp.write_map(str(map_path), region, nest=False, dtype=np.float64)
    masks["custom"] = {"type": "Custom", "coordsys": "icrs", "custom": {
        "healpix_map": str(map_path), "ordering": "ring", "threshold": 0.5}}
    for name, mask in masks.items():
        configurations[name] = deepcopy(base)
        configurations[name]["sky_mask"] = mask

    hlv = deepcopy(base)
    hlv["ifo"] = ["L1", "H1", "V1"]
    hlv["injection"]["segment"]["noise"]["seeds"] = [150914, 150915, 150916]
    configurations["hlv"] = hlv
    geometry = {"schema_version": 1, "geometries": {"X1:tutorial": {
        "detector": "X1", "parameters": {
            "name": "Illustrative tutorial detector", "lat": np.pi / 4,
            "lon": np.pi / 18, "elevation": 100.0,
            "x": {"az": np.pi / 2, "alt": 0.0, "midpoint": 2000.0},
            "y": {"az": 0.0, "alt": 0.0, "midpoint": 2000.0}}}}}
    geometry_path = destination / "detectors.json"
    geometry_path.write_text(json.dumps(geometry, indent=2) + "\n")
    custom = deepcopy(hlv)
    custom.update(ifo=["L1", "H1", "X1"],
                  detector_definitions_file=str(geometry_path),
                  detector_geometry={"X1": "X1:tutorial"})
    configurations["custom_network"] = custom

    population = deepcopy(base)
    parameters = []
    for sim_idx, (offset, amplitude) in enumerate(
        zip([-30, -10, 10, 30], [1e-23, 3e-22, 6e-22, 1e-21])
    ):
        injection = deepcopy(base["injection"]["parameters"])
        injection.update(gps_time=injection["gps_time"] + offset,
                         hrss=amplitude, sim_idx=sim_idx)
        parameters.append(injection)
    population["injection"]["parameters"] = parameters
    configurations["population"] = population

    gated = deepcopy(base)
    gated["selection"] = {"time_vetoes": [{
        "module": "pycwb.modules.conditioning_plugins.cwb_gating",
        "options": {"energy_threshold": 1e6, "integration_seconds": 0.5,
                    "padding_seconds": 1.5}}]}
    configurations["gated"] = gated
    bounded = deepcopy(base)
    bounded["execution"] = {"profile": "scalable", "memory_limit": "8GiB",
        "worker_memory": "2GiB", "cache_limit": "512MiB", "headroom": "512MiB",
        "batch_size": 1, "preload": "off"}
    configurations["bounded"] = bounded

    waveform = deepcopy(base)
    waveform["injection"]["generator"] = str(Path(__file__).with_name("waveform.py")) + ".get_td_waveform"
    configurations["custom_waveform"] = waveform

    # Public-data example: 20 minutes around GW150914, sampled at 4096 Hz.
    # No synthetic injection block or simulation mode is retained.
    real = deepcopy(base)
    real.pop("injection")
    real.pop("simulation")
    real.update(inRate=4096, levelR=2, gps_start=1126258862,
                gps_end=1126260062, segLen=600, segMLS=64, slagSize=0,
                save_injection=False, plot_injection=False,
                channelNamesRaw=["L1:GWOSC-4KHZ_R1_STRAIN", "H1:GWOSC-4KHZ_R1_STRAIN"],
                frFiles=[str(destination / "input/L1_frames.in"),
                         str(destination / "input/H1_frames.in")])
    real["DQF"] = [[ifo, str(destination / f"input/{ifo}_cat{cat}.txt"),
                    f"CWB_CAT{cat}", 0.0, False, False]
                   for ifo in real["ifo"] for cat in range(3)]
    configurations["open_data"] = real
    background = deepcopy(real)
    background.update(lagSize=9, lagStep=1.0, lagOff=1, lagMax=0,
                      save_waveform=False, save_sky_map=False)
    configurations["background"] = background

    for name, config in configurations.items():
        (destination / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Wrote {len(configurations)} configurations to {destination}")
    return configurations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tutorial-work"))
    prepare(parser.parse_args().output)
