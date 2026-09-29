"""Bit-exact parity of ``LikelihoodScan`` with ``likelihoodWP.sky_scan.scan_sky_for_best_fit``."""

from __future__ import annotations

import numpy as np
import pytest

from pycwb.modules.likelihoodWP.sky_scan import scan_sky


def reference_scan(n_ifo, n_pix, n_sky, FP, FX, rms, td00, td90, ml, *settings):
    return scan_sky((FP, FX, ml), (rms, td00, td90), settings, reuse_delays=False)


from pycwb.utils.tests.gpu_helpers import assert_same_tuple_bits, pixel_packet, sky_geometry

pytestmark = pytest.mark.gpu

REG = np.array([0.1, 2.0, 0.0], np.float32)


def _scan(options=None):
    from pycwb.modules.likelihood_gpu.likelihood_scan import LikelihoodScan

    return LikelihoodScan(options)


def _case(
    rng: np.random.Generator,
    n_ifo: int,
    n_pix: int,
    n_sky: int = 64,
    n_delay: int = 9,
    n_valid: int = 53,
):
    FP, FX, ml = sky_geometry(rng, n_sky, n_ifo, n_delay)
    rms, td00, td90 = pixel_packet(rng, n_pix, n_ifo, n_delay)
    skies = rng.permutation(n_sky).astype(np.int64)[:n_valid]
    return FP, FX, ml, rms, td00, td90, skies


@pytest.mark.parametrize("n_ifo", [2, 3])
@pytest.mark.parametrize("n_pix", [1, 11, 64])
@pytest.mark.parametrize("netCC", [-1.0, 0.5])
def test_scan_matches_cpu(
    rng: np.random.Generator,
    reuse_workspace: str | None,
    n_ifo: int,
    n_pix: int,
    netCC: float,
) -> None:
    FP, FX, ml, rms, td00, td90, skies = _case(rng, n_ifo, n_pix)
    gpu = _scan(reuse_workspace)
    assert (gpu.workspace is not None) == reuse_workspace["reuse_workspace"]
    args = (n_ifo, n_pix, 64, FP, FX, rms, td00, td90, ml, REG, netCC, 0.1, 0.5, skies)
    expected = reference_scan(*args)
    actual = gpu.scan_sky(
        (FP, FX, ml), (rms, td00, td90), (REG, netCC, 0.1, 0.5, skies)
    )
    assert len(expected) == 13
    assert int(actual[0]) == int(expected[0])
    assert_same_tuple_bits(actual[1:], expected[1:])


@pytest.mark.parametrize("n_ifo", [2, 3])
def test_scan_with_positive_delta_regulator_and_full_sky(
    rng: np.random.Generator, n_ifo: int
) -> None:
    FP, FX, ml, rms, td00, td90, _ = _case(rng, n_ifo, 23, n_sky=128, n_delay=7)
    skies = np.arange(128, dtype=np.int64)
    args = (n_ifo, 23, 128, FP, FX, rms, td00, td90, ml, REG, -1.0, 0.5, 2.0, skies)
    expected = reference_scan(*args)
    actual = _scan()(*args)
    assert int(actual[0]) == int(expected[0])
    assert_same_tuple_bits(actual[1:], expected[1:])


def test_scan_implements_release_grouped_interface(rng: np.random.Generator) -> None:
    """CPU delay reuse scheduling does not affect the CUDA arithmetic."""
    FP, FX, ml, rms, td00, td90, skies = _case(rng, 2, 7)
    args = (2, 7, 64, FP, FX, rms, td00, td90, ml, REG, 0.5, 0.1, 0.5, skies)
    expected = reference_scan(*args)
    actual = _scan().scan_sky(
        (FP, FX, ml), (rms, td00, td90), args[9:], reuse_delays=True
    )
    assert int(actual[0]) == int(expected[0])
    assert_same_tuple_bits(actual[1:], expected[1:])


def test_geometry_is_uploaded_once_per_trial(rng: np.random.Generator) -> None:
    FP, FX, ml, rms, td00, td90, skies = _case(rng, 2, 5)
    gpu = _scan()
    for _ in range(4):
        gpu(
            2,
            5,
            64,
            FP.copy(),
            FX.copy(),
            rms,
            td00,
            td90,
            ml.copy(),
            REG,
            0.5,
            0.1,
            0.5,
            skies,
        )
    assert len(gpu.geometry) == 1


def test_input_validation(rng: np.random.Generator) -> None:
    FP, FX, ml, rms, td00, td90, skies = _case(rng, 2, 5)
    gpu = _scan()
    good = [2, 5, 64, FP, FX, rms, td00, td90, ml, REG, 0.5, 0.1, 0.5, skies]
    with pytest.raises(ValueError, match="2/3 detectors"):
        gpu(*[4, *good[1:]])
    with pytest.raises(ValueError, match="nonempty pixels"):
        gpu(*[2, 0, *good[2:]])
    with pytest.raises(ValueError, match="Invalid sky mask"):
        gpu(*good[:-1], np.empty(0, np.int64))
    with pytest.raises(ValueError, match="Invalid sky mask"):
        gpu(*good[:-1], np.array([64], np.int64))
    bad = list(good)
    bad[5] = rms[:, :1]
    with pytest.raises(ValueError, match="Invalid scan shapes"):
        gpu(*bad)
    bad = list(good)
    bad[8] = ml.T
    with pytest.raises(ValueError, match="Invalid scan shapes"):
        gpu(*bad)
    bad = list(good)
    bad[8] = np.full_like(ml, 5)
    with pytest.raises(ValueError, match="Invalid delays"):
        gpu(*bad)
