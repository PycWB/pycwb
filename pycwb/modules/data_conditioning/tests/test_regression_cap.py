import numpy as np
import pytest
from pycwb.modules.data_conditioning.regression import (
    _cap_witness_numba,
    _cap_witness_jax,
    _numba_process_layers,
    _jax_process_layers,
)


@pytest.mark.parametrize("pattern", ["zero", "flat", "outlier", "edge_outliers"])
@pytest.mark.parametrize("fraction", [1.0, 0.95, 0.5])
def test_cap_preserves_phase_and_uses_whole_segment(pattern, fraction):
    rng = np.random.default_rng(87)
    a = rng.normal(size=100)
    b = rng.normal(size=100)
    if pattern == "zero":
        a[:] = b[:] = 0
    if pattern == "flat":
        a[:] = 3
        b[:] = 4
    if pattern == "outlier":
        a[50] = 1e4
        b[50] = -2e4
    if pattern == "edge_outliers":
        a[:4] = 1e4
        b[-4:] = -2e4
    original_a, original_b = a.copy(), b.copy()
    energy = a * a + b * b
    expected = np.ones(100)
    if fraction < 1:
        threshold = 5 * np.sort(energy)[int(fraction * 100 - 1)]
        mask = energy > threshold
        expected[mask] = np.sqrt(threshold / energy[mask])
    ja, jb = _cap_witness_jax(a, b, fraction)
    ja, jb = np.asarray(ja).copy(), np.asarray(jb).copy()
    _cap_witness_numba(a, b, fraction)
    np.testing.assert_array_equal(a, original_a * expected)
    np.testing.assert_array_equal(b, original_b * expected)
    np.testing.assert_allclose(ja, a, rtol=2e-15, atol=1e-15)
    np.testing.assert_allclose(jb, b, rtol=2e-15, atol=1e-15)


def test_batch_backends_propagate_fraction_and_keep_legacy_default():
    rng = np.random.default_rng(78)
    a = rng.normal(size=(2, 256))
    b = rng.normal(size=(2, 256))
    a[:, 120:124] *= 100
    b[:, 120:124] *= 100
    # K=2, matrix 10x10, top ten eigenvectors; avoid a degenerate eigenspace cut.
    args = (a, b, 2, 4, 10, 5, 0.95, 20, 0.0, 0.0, 10, 0, 0.0, 8.0, 2.0)
    old, old_mask = _numba_process_layers(*args, 1)
    explicit, explicit_mask = _numba_process_layers(*args, 1, 1.0)
    np.testing.assert_array_equal(old, explicit)
    np.testing.assert_array_equal(old_mask, explicit_mask)
    cap, mask = _numba_process_layers(*args, 1, 0.95)
    assert np.max(np.abs(old - cap)) > 1e-3
    jcap, jmask = _jax_process_layers(*args, 0.95)
    np.testing.assert_array_equal(mask, jmask)
    np.testing.assert_allclose(cap, jcap, rtol=1e-10, atol=1e-10)


def test_search_old_option_selects_uncapped_behavior(monkeypatch):
    from types import SimpleNamespace
    from pycwb.modules.data_conditioning.regression import _regression_apply_fraction

    from pycwb.constants.execution_profile import ExecutionProfile

    enabled = ExecutionProfile(regression_cap=True)
    assert _regression_apply_fraction(SimpleNamespace(execution_profile=enabled, Search="CBC"), 0.95) == 0.95
    assert (
        _regression_apply_fraction(SimpleNamespace(execution_profile=enabled, Search="CBC --regression OLD"), 0.95)
        == 1.0
    )
    assert (
        _regression_apply_fraction(
            SimpleNamespace(execution_profile=enabled, Search="CBC --regression OLD --foo value"), 0.95
        )
        == 1.0
    )
    enabled = ExecutionProfile(regression_cap=False)
    assert _regression_apply_fraction(SimpleNamespace(execution_profile=enabled, Search="CBC"), 0.95) == 1.0
