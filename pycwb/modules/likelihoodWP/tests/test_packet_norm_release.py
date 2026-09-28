"""Regression against independently evaluated cWB 6.4.6.9 SSE arithmetic."""

from pathlib import Path

import numpy as np
import pytest

from pycwb.modules.likelihoodWP.packet_ops import compute_packet_norms

GOLDEN = Path(__file__).with_name("packet_norm_release_golden.npz")
with np.load(GOLDEN) as archive:
    CASES = sorted(key.removesuffix("__expected") for key in archive.files if key.endswith("__expected"))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_release_packet_norm(case, dtype):
    # Float64 callers must enter the same float32 decision path as cWB.
    with np.load(GOLDEN) as archive:
        args = [
            archive[case + "__" + key].astype(np.int64 if key == "lookup" else dtype)
            for key in ["p", "q", "xtalk", "lookup", "mask", "energy"]
        ]
        expected = archive[case + "__expected"]
    saved = [a.copy() for a in args]
    outputs = compute_packet_norms(*args)
    assert all(a.dtype == np.float32 for a in outputs)
    actual = np.concatenate([a.ravel() for a in outputs])
    assert actual.tobytes() == expected.tobytes()
    for a, original in zip(args, saved):
        np.testing.assert_array_equal(a, original)


def test_unit_ratio_excludes_pixel():
    # Double precision retains this pixel because of the 1e-12 denominator
    # offset; cWB float32 rounds the ratio to one and removes it.
    p = np.ones((1, 1), dtype=np.float32)
    q = np.zeros_like(p)
    xtalk = np.zeros((1, 8), dtype=np.float32)
    xtalk[0, 4] = xtalk[0, 7] = 1
    _, _, _, qnorm = compute_packet_norms(p, q, xtalk, np.array([[0, 1]], dtype=np.int64), p[0], p[:, 0])
    assert qnorm[0, 0] == 0
