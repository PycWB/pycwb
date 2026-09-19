import numpy as np
import pytest

from pycwb.modules.likelihoodWP.dpf import dpf_np_loops_vec, calculate_dpf
from pycwb.modules.likelihoodWP.dpf_regulator import dpf_index_only, calculate_dpf_scalar


@pytest.mark.parametrize("n_ifo", [1, 2, 3, 4, 8])
def test_scalar_index_exact(n_ifo):
    rng = np.random.default_rng(52)
    for n_pix in [0, 1, 3, 7, 31, 100, 1000]:
        for scale in [0.0, 0.001, 1.0, 100.0]:
            fp = rng.normal(size=n_ifo).astype("f4")
            fx = rng.normal(size=n_ifo).astype("f4")
            rms = (rng.uniform(0.01, 2, (n_pix, n_ifo)) * scale).astype("f4")
            expected = dpf_np_loops_vec(fp, fx, rms)[0]
            actual = dpf_index_only(fp, fx, rms)
            assert float(actual).hex() == float(expected).hex()


def test_regulator_masks_and_thresholds_exact():
    rng = np.random.default_rng(53)
    fp = rng.normal(size=(31, 2)).astype("f4")
    fx = rng.normal(size=(31, 2)).astype("f4")
    rms = rng.uniform(0.1, 2, (7, 2)).astype("f4")
    boundary = dpf_np_loops_vec(fp[3], fx[3], rms)[0]
    for indices in [np.arange(31), np.array([3, 1, 25]), np.array([], dtype="i8")]:
        for gamma in [-1.0, 0.0, 1.0, boundary, np.nextafter(boundary, -np.inf), np.nextafter(boundary, np.inf)]:
            args = (fp, fx, rms, 31, 2, gamma, 4.0, indices)
            assert calculate_dpf_scalar(*args) == calculate_dpf(*args)
