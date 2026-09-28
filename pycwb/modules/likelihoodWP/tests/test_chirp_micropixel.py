"""Release-oracle regression tests; no ROOT dependency at test time."""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from pycwb.modules.likelihoodWP.chirp_micropixel import estimate_chirp, build_micropixels, root_uniforms

REF = Path(__file__).parent / "reference"
FIELDS = ["time", "frequency", "layers", "rate", "core", "likelihood"]
MAPPING = {
    "mass": "mchirp",
    "mass_error": "mchirperr",
    "merger_time": "tmrgr",
    "merger_time_error": "tmrgrerr",
    "ellipticity": "chirpEllip",
    "energy_fraction": "chirpEfrac",
    "symmetry": "chirpPfrac",
}


@pytest.mark.parametrize("seed", [1, 2, 42])
def test_release_random_sequence(seed):
    with np.load(REF / "chirp_oracle.npz") as saved:
        np.testing.assert_array_equal(root_uniforms(seed, 20000), saved[f"uniforms_{seed}"])


@pytest.mark.parametrize("case", json.loads((REF / "chirp_oracle.json").read_text()))
def test_synthetic_release(case):
    with np.load(REF / "chirp_oracle.npz") as saved:
        pixels = SimpleNamespace(**{k: saved[case["case"] + "__" + k] for k in FIELDS})
        cells, dt = build_micropixels(pixels, 256.0)
        np.testing.assert_array_equal(cells, saved[case["case"] + "__cells"])
    assert dt == case["min_dt"]
    result = estimate_chirp(pixels, 256.0, case["seed"])
    for field, reference in MAPPING.items():
        assert getattr(result, field) == case[reference]


@pytest.mark.parametrize("case", json.loads((REF / "chirp_real_oracle.json").read_text()))
def test_real_release(case):
    with np.load(REF / "chirp_real_inputs.npz") as inputs:
        pixels = SimpleNamespace(**{k: inputs[case["name"] + "__" + k] for k in FIELDS})
    cells, dt = build_micropixels(pixels, case["analysis_rate"])
    with np.load(REF / "chirp_real_oracle_cells.npz") as saved:
        np.testing.assert_array_equal(cells, saved[case["name"]])
    result = estimate_chirp(pixels, case["analysis_rate"], case["seed"])
    for field, reference in MAPPING.items():
        assert getattr(result, field) == case[reference]


def test_empty_pixels_and_explicit_seed_contract():
    pixels = SimpleNamespace(core=np.array([], dtype=bool))
    assert all(x == 0.0 for x in dataclasses.asdict(estimate_chirp(pixels, 256.0, 1)).values())
    with pytest.raises(ValueError, match="nonzero"):
        root_uniforms(0, 10)
