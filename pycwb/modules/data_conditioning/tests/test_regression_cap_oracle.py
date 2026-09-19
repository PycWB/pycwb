"""Replay installed cWB 6.4.6.9 regression fixtures without ROOT at test time."""

import json
from pathlib import Path

import numpy as np
import pytest

from pycwb.modules.data_conditioning.regression import (
    _cap_witness_numba,
    _jax_layer_apply_filters,
)

REFERENCE = Path(__file__).with_name("reference")
CASES = json.loads((REFERENCE / "regression_cap_oracle.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["key"])
def test_release_filter_prediction(case):
    with np.load(REFERENCE / "regression_cap_oracle.npz") as data:
        key = case["key"]
        a = data[key + "__real"] / case["norm"]
        b = data[key + "__imag"] / case["norm"]
        _cap_witness_numba(a, b, case["fraction"])
        f, g = data[key + "__filter00"], data[key + "__filter90"]
        x, y = np.zeros_like(a), np.zeros_like(b)
        half = case["K"]
        for i in range(half, len(a) - half):
            for k in range(-half, half + 1):
                x[i] += f[k + half] * a[i + k] - g[k + half] * b[i + k]
                y[i] += g[k + half] * a[i + k] + f[k + half] * b[i + k]
        jx, jy = _jax_layer_apply_filters(
            data[key + "__real"],
            data[key + "__imag"],
            case["norm"],
            f,
            g,
            half,
            case["fraction"],
        )
        for actual, expected in [
            (x, data[key + "__noise00"]),
            (y, data[key + "__noise90"]),
            (jx, data[key + "__noise00"]),
            (jy, data[key + "__noise90"]),
        ]:
            np.testing.assert_allclose(
                np.asarray(actual) * case["target_norm"],
                expected,
                rtol=3e-13,
                atol=3e-13,
            )
