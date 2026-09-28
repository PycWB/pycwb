"""Conditioning dispatch, public API, and shared frequency-boundary contracts."""

import importlib
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

conditioning = importlib.import_module("pycwb.modules.data_conditioning.data_conditioning")


@pytest.mark.parametrize("method", ["wavelet", "python", "mesa"])
def test_multi_and_single_detector_dispatch_preserves_order(monkeypatch, method):
    calls = []

    def regress(config, strain):
        calls.append(("regress", strain))
        return strain + 10

    def whiten(config, strain):
        calls.append(("whiten", strain))
        return strain * 2, strain * 3

    monkeypatch.setattr(conditioning, "apply_regression", regress)
    if method == "mesa":
        # Dispatch can be tested even if optional MESA dependencies are absent.
        import sys
        monkeypatch.setitem(sys.modules, "pycwb.modules.data_conditioning.whitening_mesa",
                            SimpleNamespace(whiten_mesa=whiten))
    else:
        monkeypatch.setattr(conditioning, "whiten_wavelet", whiten)
    config = SimpleNamespace(whiteMethod=method)
    assert conditioning.condition_strains(config, [1, 2]) == ((22, 24), (33, 36))
    assert calls == [("regress", 1), ("regress", 2), ("whiten", 11), ("whiten", 12)]
    calls.clear()
    assert conditioning.condition_strain(config, 1) == (22, 33)
    assert calls == [("regress", 1), ("whiten", 11)]


def test_invalid_method_fails_before_regression(monkeypatch):
    def forbidden(*args):
        pytest.fail("Invalid whitening method must fail before regression")
    monkeypatch.setattr(conditioning, "apply_regression", forbidden)
    config = SimpleNamespace(whiteMethod="mixed")
    for function, data in [(conditioning.condition_strain, 1), (conditioning.condition_strains, [1])]:
        with pytest.raises(ValueError, match="not a valid"):
            function(config, data)


def test_public_api_is_explicit_and_has_no_legacy_function_aliases():
    package = importlib.import_module("pycwb.modules.data_conditioning")
    assert set(package.__all__) == {
        "condition_strains", "condition_strain", "apply_regression", "apply_regression_jax",
        "whiten_wavelet", "whiten_mesa", "whiten_injection_strain", "apply_psd_correction",
    }
    for name in package.__all__:
        if name != "whiten_mesa":
            assert callable(getattr(package, name))
    for name in ["whitening_python", "whitening_mesa_python", "regression_python", "whitening_mdc", "np", "jax"]:
        assert not hasattr(package, name)
    assert "nproc" not in inspect.signature(package.condition_strains).parameters


@pytest.mark.parametrize("f1,f2,expected", [
    (16.0, 0.0, [False, False, False, True, True, True, True, False, False]),
    (0.0, 32.0, [False, False, False, True, False, False, False, False, False]),
])
def test_bandpass_constant_preserves_input_and_bin_boundaries(f1, f2, expected):
    from pycwb.modules.data_conditioning.whitening_common import _apply_cwb_bandpass_constant
    original = np.arange(27, dtype=float).reshape(9, 3) + 2
    before = original.copy()
    result = _apply_cwb_bandpass_constant(original, f1, f2, 1.0, 8.0, 16.0, 56.0)
    keep = np.asarray(expected)
    np.testing.assert_array_equal(original, before)
    np.testing.assert_array_equal(result[keep], original[keep])
    np.testing.assert_array_equal(result[~keep], 1.0)
