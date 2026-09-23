"""Bit-exact parity of ``DPFRegulator`` with ``likelihoodWP.dpf_regulator.calculate_dpf_scalar``."""

from __future__ import annotations

import numpy as np
import pytest

from pycwb.modules.likelihoodWP.dpf_regulator import calculate_dpf_scalar, dpf_index_only

from ._helpers import assert_same_bits

pytestmark = pytest.mark.gpu


def _regulator():
    from pycwb.modules.background_cuda.dpf_regulator import DPFRegulator

    return DPFRegulator()


def _inputs(rng: np.random.Generator, n_sky: int, n_ifo: int, n_pix: int, n_valid: int) -> tuple:
    FP = rng.normal(size=(n_sky, n_ifo)).astype(np.float32)
    FX = rng.normal(size=(n_sky, n_ifo)).astype(np.float32)
    FP[0] = 0.0
    FX[0] = 0.0
    rms = rng.uniform(0.0, 2.0, size=(n_pix, n_ifo)).astype(np.float32)
    skies = np.sort(rng.permutation(n_sky)[:n_valid]).astype(np.int64)
    return FP, FX, rms, skies


@pytest.mark.parametrize("n_ifo", [2, 3])
@pytest.mark.parametrize(("n_pix", "n_valid"), [(1, 512), (11, 512), (64, 97), (257, 31)])
def test_regulator_matches_cpu_scalar(
    rng: np.random.Generator, reuse_workspace: str | None, n_ifo: int, n_pix: int, n_valid: int
) -> None:
    FP, FX, rms, skies = _inputs(rng, 512, n_ifo, n_pix, n_valid)
    gpu = _regulator()
    assert (gpu.workspace is not None) == (reuse_workspace == "1")
    for gamma, threshold in ((0.3, 0.5), (1.2, 4.0)):
        args = (FP, FX, rms, 512, n_ifo, gamma, threshold, skies)
        expected = calculate_dpf_scalar(*args)
        actual = gpu(*args)
        assert_same_bits(np.float64(actual), np.float64(expected))


@pytest.mark.parametrize("n_ifo", [2, 3])
def test_regulator_accepts_float64_inputs_like_cpu(rng: np.random.Generator, n_ifo: int) -> None:
    FP, FX, rms, skies = _inputs(rng, 128, n_ifo, 23, 128)
    args = (FP.astype(np.float64), FX.astype(np.float64), rms.astype(np.float64), 128, n_ifo, 0.3, 0.5, skies)
    assert_same_bits(np.float64(_regulator()(*args)), np.float64(calculate_dpf_scalar(*args)))


@pytest.mark.parametrize("n_ifo", [2, 3])
def test_per_direction_index_matches_dpf_index_only(rng: np.random.Generator, n_ifo: int) -> None:
    """The raw kernel output per direction is what the CPU scalar index computes."""
    import ctypes as ct

    from numba import cuda

    FP, FX, rms, skies = _inputs(rng, 256, n_ifo, 37, 256)
    gpu = _regulator()
    arrays = [cuda.to_device(x) for x in (FP, FX, rms, skies)]
    out = cuda.device_array(256, np.float64)
    gpu.module.launch("dpf_index", 256, [*arrays, ct.c_int(256), ct.c_int(37), ct.c_int(n_ifo), out])
    actual = out.copy_to_host()
    expected = np.array([dpf_index_only(FP[s], FX[s], rms) for s in skies])
    assert_same_bits(actual, expected)


def test_geometry_is_uploaded_once_per_trial(rng: np.random.Generator) -> None:
    FP, FX, rms, skies = _inputs(rng, 64, 2, 5, 64)
    gpu = _regulator()
    for _ in range(5):
        # Native code hands out fresh views of unchanged geometry every lag.
        gpu(FP.copy(), FX.copy(), rms, 64, 2, 0.3, 0.5, skies)
    assert len(gpu.geometry) == 1
    gpu(FP * 2, FX, rms, 64, 2, 0.3, 0.5, skies)
    assert len(gpu.geometry) == 2


def test_changing_shapes_reuse_workspace(rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYCWB_GPU_REUSE_WORKSPACE", "1")
    gpu = _regulator()
    for n_ifo in (2, 3):
        FP = rng.normal(size=(4096, n_ifo)).astype(np.float32)
        FX = rng.normal(size=(4096, n_ifo)).astype(np.float32)
        for n_pix, n_valid in ((1, 4096), (17, 31), (80, 512), (3, 16), (256, 2048), (11, 4096)):
            rms = rng.uniform(0.1, 1.0, size=(n_pix, n_ifo)).astype(np.float32)
            skies = rng.permutation(4096)[:n_valid].astype(np.int64)
            args = (FP, FX, rms, 4096, n_ifo, 0.3, 0.5, skies)
            assert_same_bits(np.float64(gpu(*args)), np.float64(calculate_dpf_scalar(*args)))
    assert set(gpu.workspace.slots) == {"weights", "skies", "output"}
    assert sum(slot.nbytes for slot in gpu.workspace.slots.values()) <= gpu.workspace.budget


def test_empty_sky_mask_returns_negative_threshold(rng: np.random.Generator) -> None:
    FP, FX, rms, _ = _inputs(rng, 16, 2, 4, 16)
    assert _regulator()(FP, FX, rms, 16, 2, 0.3, 0.75, np.empty(0, np.int64)) == -0.75


def test_input_validation(rng: np.random.Generator) -> None:
    FP, FX, rms, skies = _inputs(rng, 16, 2, 4, 16)
    gpu = _regulator()
    with pytest.raises(ValueError, match="2/3 detectors"):
        gpu(
            np.zeros((16, 4), np.float32),
            np.zeros((16, 4), np.float32),
            np.zeros((4, 4), np.float32),
            16,
            4,
            0.3,
            0.5,
            skies,
        )
    with pytest.raises(ValueError, match="matching sky/pixel shapes"):
        gpu(FP, FX, rms, 15, 2, 0.3, 0.5, skies)
    with pytest.raises(ValueError, match="matching sky/pixel shapes"):
        gpu(FP, FX[:, :1], rms, 16, 2, 0.3, 0.5, skies)
    with pytest.raises(ValueError, match="matching sky/pixel shapes"):
        gpu(FP, FX, rms[:, :1], 16, 2, 0.3, 0.5, skies)
    with pytest.raises(ValueError, match="Invalid sky indices"):
        gpu(FP, FX, rms, 16, 2, 0.3, 0.5, np.array([16], np.int64))
    with pytest.raises(ValueError, match="Invalid sky indices"):
        gpu(FP, FX, rms, 16, 2, 0.3, 0.5, np.array([-1], np.int64))
    with pytest.raises(ValueError, match="Invalid sky indices"):
        gpu(FP, FX, rms, 16, 2, 0.3, 0.5, skies.reshape(4, 4))
