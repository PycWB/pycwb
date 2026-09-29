"""Nonzero-mean self-witness regression against captured cWB statistics."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pycwb.config.processing import ExecutionProfile
from pycwb.modules.data_conditioning import regression_numba as numba_backend
from pycwb.modules.data_conditioning import regression_jax as jax_backend
from pycwb.types.time_series import TimeSeries

REFERENCE = Path(__file__).with_name("reference") / "regression_witness_oracle.npz"


@pytest.mark.parametrize("backend", ["numba", "jax"])
def test_dispatch_demeans_only_witness_and_preserves_input(monkeypatch, backend):
    regression = importlib.import_module("pycwb.modules.data_conditioning.regression")
    original = np.arange(128, dtype=np.float64) + 100.0
    strain = TimeSeries(original.copy(), dt=1 / 64, t0=1234.0)
    transforms = []
    received = []

    class RecordingWDM:
        def __init__(self, **kwargs):
            pass

        def t2w(self, data, **kwargs):
            transforms.append(data.copy())
            return SimpleNamespace(data=np.tile(data, (3, 1)).astype(complex), df=4.0, dt=0.125)

    def capture(target_real, target_imag, witness_real, witness_imag, *args, **kwargs):
        received.append(tuple(np.asarray(v) for v in (target_real, target_imag, witness_real, witness_imag)))
        return np.zeros_like(target_real, dtype=complex), np.zeros(len(target_real), dtype=bool)

    monkeypatch.setattr(regression, "WDM", RecordingWDM)
    module = numba_backend if backend == "numba" else jax_backend
    monkeypatch.setattr(module, f"_{backend}_process_layers", capture)
    config = SimpleNamespace(rateANA=64, fHigh=32, segEdge=0,
                             execution_profile=ExecutionProfile(regression_engine=backend))
    result = regression.apply_regression(config, strain)
    assert len(transforms) == 2
    np.testing.assert_array_equal(transforms[0], original)
    np.testing.assert_array_equal(transforms[1], original - original.mean())
    np.testing.assert_array_equal(received[0][0][0], original)
    np.testing.assert_array_equal(received[0][2][0], original - original.mean())
    np.testing.assert_array_equal(strain.data, original)
    np.testing.assert_array_equal(result.data, original)


def test_demeaned_witness_statistics_match_cwb_trim_boundary():
    with np.load(REFERENCE) as ref:
        tr, ti = ref['target'].T
        wr, wi = ref['witness'].T
        target_norm, valid, witness_norm, vector, acf, ccf = jax_backend._jax_layer_build_stats(
            tr, ti, wr, wi, 8, 16, 34, 17, 0.95, 80, 0.0,
        )
        assert valid
        np.testing.assert_allclose(target_norm, ref['target_norm'], rtol=2e-13, atol=0)
        np.testing.assert_allclose(witness_norm, ref['witness_norm'], rtol=2e-13, atol=0)
        np.testing.assert_allclose(vector, ref['cwb_cross'], rtol=2e-10, atol=2e-12)
        matrix = jax_backend._jax_build_matrix(acf, ccf, 8, 16, 0.0)
        np.testing.assert_allclose(matrix, ref['cwb_matrix'], rtol=2e-10, atol=2e-12)
        # The former target-as-witness path selects a different 95% trim sample.
        old = jax_backend._jax_layer_build_stats(tr, ti, tr, ti, 8, 16, 34, 17, 0.95, 80, 0.0)
        old_matrix = jax_backend._jax_build_matrix(old[-2], old[-1], 8, 16, 0.0)
        assert np.max(np.abs(old_matrix - matrix)) > 2e-4


@pytest.mark.parametrize("backend", ["numba", "numba_python", "jax", "numba_batch", "jax_batch"])
def test_predicted_noise_uses_witness_and_target_normalization(backend):
    with np.load(REFERENCE) as ref:
        tr, ti = ref['target'].T.copy()
        wr, wi = ref['witness'].T.copy()
        before = [v.copy() for v in (tr, ti, wr, wi)]
        # Independent prediction from cWB's captured taps and normalizations.
        witness = (wr + 1j * wi) / ref['witness_norm']
        taps = ref['cwb_filter'][:17] + 1j * ref['cwb_filter'][17:]
        expected = np.zeros_like(witness)
        expected[8:-8] = np.convolve(witness, taps[::-1], mode='valid') * ref['target_norm']
        args = (tr, ti, wr, wi, 8, 16, 34, 17, 0.95, 80, 0.0, 0.0, 10, 0, 0.0, 8.0, 10.0)
        if backend.endswith('_batch'):
            batched = tuple(v[None, :] for v in (tr, ti, wr, wi)) + args[4:]
            if backend == 'numba_batch':
                actual, included = numba_backend._numba_process_layers(*batched, 1)
            else:
                actual, included = jax_backend._jax_process_layers(*batched)
            actual, included = np.asarray(actual)[0], np.asarray(included)[0]
        elif backend == 'jax':
            actual, included = jax_backend._jax_process_one_layer(*args)
        else:
            kernel = numba_backend._numba_process_one_layer
            if backend == 'numba_python':
                kernel = kernel.py_func
            actual, included = kernel(*args, 1)
        assert included
        assert np.linalg.norm(np.asarray(actual) - expected) / np.linalg.norm(expected) < 2e-10
        for actual_input, original in zip((tr, ti, wr, wi), before):
            np.testing.assert_array_equal(actual_input, original)


@pytest.mark.parametrize("backend", ["numba", "jax"])
@pytest.mark.parametrize("empty", ["target", "witness", "both"])
def test_empty_target_or_witness_returns_zero_prediction(backend, empty):
    x = np.random.default_rng(5).normal(size=(4, 1, 128))
    if empty in ('target', 'both'):
        x[:2] = 0
    if empty in ('witness', 'both'):
        x[2:] = 0
    args = (*x, 2, 4, 10, 5, 0.95, 20, 0.0, 0.0, 10, 0, 0.0, 8.0, 2.0)
    if backend == 'numba':
        noise, included = numba_backend._numba_process_layers(*args, 1)
    else:
        noise, included = jax_backend._jax_process_layers(*args)
    assert not np.any(included)
    np.testing.assert_array_equal(noise, 0)
