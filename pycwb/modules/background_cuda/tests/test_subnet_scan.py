"""Bit-exact parity of ``SubnetScan`` with ``sub_net_cut.optimze_sky_loc_from_td``."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ._helpers import assert_same_bits, pixel_packet, sky_geometry

pytestmark = pytest.mark.gpu

cpu_scan = importlib.import_module("pycwb.modules.super_cluster_native.sub_net_cut").optimze_sky_loc_from_td


def _scan():
    from pycwb.modules.background_cuda.subnet_scan import SubnetScan

    return SubnetScan()


def _as_bits(result: tuple) -> np.ndarray:
    return np.array(result, np.float64)


@pytest.mark.parametrize("n_ifo", [2, 3])
@pytest.mark.parametrize("n_pix", [1, 11, 64])
@pytest.mark.parametrize("subcut", [-1.0, 0.1, 0.7])
def test_single_scan_matches_cpu(rng: np.random.Generator, n_ifo: int, n_pix: int, subcut: float) -> None:
    n_sky, n_delay = 256, 9
    FP, FX, ml = sky_geometry(rng, n_sky, n_ifo, n_delay)
    rms, td00, td90 = pixel_packet(rng, n_pix, n_ifo, n_delay)
    args = (n_ifo, n_pix, n_sky, FP, FX, rms, td00, td90, ml, 4.0, 1.0, subcut)
    expected = cpu_scan(*args)
    actual = _scan()(*args)
    assert len(actual) == len(expected) == 8
    assert_same_bits(_as_bits(actual), _as_bits(expected))


@pytest.mark.parametrize("n_ifo", [2, 3])
def test_all_zero_packet_returns_empty_result(rng: np.random.Generator, n_ifo: int) -> None:
    FP, FX, ml = sky_geometry(rng, 32, n_ifo, 5)
    rms, td00, td90 = pixel_packet(rng, 4, n_ifo, 5)
    td00[:] = 0.0
    td90[:] = 0.0
    args = (n_ifo, 4, 32, FP, FX, rms, td00, td90, ml, 4.0, 1.0, 0.1)
    expected = cpu_scan(*args)
    actual = _scan()(*args)
    assert actual == (0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0)
    assert_same_bits(_as_bits(actual), _as_bits(expected))


@pytest.mark.parametrize("n_ifo", [2, 3])
@pytest.mark.parametrize("batch_size", [1, 7, 24])
def test_scan_many_matches_cpu_and_single_scan(rng: np.random.Generator, n_ifo: int, batch_size: int) -> None:
    n_sky, n_delay = 512, 9
    FP, FX, ml = sky_geometry(rng, n_sky, n_ifo, n_delay)
    inputs = []
    for i in range(24):
        n_pix = [1, 3, 11, 32, 64][i % 5]
        rms, td00, td90 = pixel_packet(rng, n_pix, n_ifo, n_delay)
        if i == 0:
            td00.fill(0.0)
            td90.fill(0.0)
        inputs.append((rms, td00, td90))
    expected = [cpu_scan(n_ifo, len(r), n_sky, FP, FX, r, a, b, ml, 4.0, 1.0, 0.1) for r, a, b in inputs]
    gpu = _scan()
    single = [gpu(n_ifo, len(r), n_sky, FP, FX, r, a, b, ml, 4.0, 1.0, 0.1) for r, a, b in inputs]
    actual = []
    for begin in range(0, len(inputs), batch_size):
        actual.extend(gpu.scan_many(inputs[begin : begin + batch_size], n_ifo, n_sky, FP, FX, ml, 4.0, 1.0, 0.1))
    assert len(actual) == len(expected)
    for index, (a, e, s) in enumerate(zip(actual, expected, single, strict=True)):
        try:
            assert_same_bits(_as_bits(a), _as_bits(e))
            assert_same_bits(_as_bits(s), _as_bits(e))
        except AssertionError as error:
            raise AssertionError(f"packet {index} (pixels={len(inputs[index][0])}): {error}") from error
    assert len(gpu.geometry) == 1


def test_scan_many_with_empty_input_list(rng: np.random.Generator) -> None:
    FP, FX, ml = sky_geometry(rng, 16, 2, 5)
    assert _scan().scan_many([], 2, 16, FP, FX, ml, 4.0, 1.0, 0.1) == []


def test_geometry_is_uploaded_once_per_trial(rng: np.random.Generator) -> None:
    FP, FX, ml = sky_geometry(rng, 32, 2, 5)
    rms, td00, td90 = pixel_packet(rng, 3, 2, 5)
    gpu = _scan()
    for _ in range(3):
        gpu(2, 3, 32, FP.copy(), FX.copy(), rms, td00, td90, ml.copy(), 4.0, 1.0, 0.1)
        gpu.scan_many([(rms, td00, td90)], 2, 32, FP.copy(), FX.copy(), ml.copy(), 4.0, 1.0, 0.1)
    assert len(gpu.geometry) == 1


def test_input_validation(rng: np.random.Generator) -> None:
    FP, FX, ml = sky_geometry(rng, 32, 2, 5)
    rms, td00, td90 = pixel_packet(rng, 3, 2, 5)
    gpu = _scan()
    with pytest.raises(ValueError, match="2/3 detectors"):
        gpu(4, 3, 32, FP, FX, rms, td00, td90, ml, 4.0, 1.0, 0.1)
    with pytest.raises(ValueError, match="Invalid subnet array shapes"):
        gpu(2, 3, 32, FP, FX, rms[:2], td00, td90, ml, 4.0, 1.0, 0.1)
    with pytest.raises(ValueError, match="Invalid subnet array shapes"):
        gpu(2, 3, 32, FP, FX, rms, td00, td90[:, :, :2], ml, 4.0, 1.0, 0.1)
    with pytest.raises(ValueError, match="Delay out of bounds"):
        gpu(2, 3, 32, FP, FX, rms, td00, td90, np.full_like(ml, 3), 4.0, 1.0, 0.1)
    with pytest.raises(ValueError, match="Invalid geometry shapes"):
        gpu.scan_many([(rms, td00, td90)], 2, 32, FP, FX, ml.T, 4.0, 1.0, 0.1)
    with pytest.raises(ValueError, match="Invalid packet shapes"):
        gpu.scan_many([(rms[:0], td00[:, :, :0], td90[:, :, :0])], 2, 32, FP, FX, ml, 4.0, 1.0, 0.1)
    with pytest.raises(ValueError, match="Invalid packet delays"):
        gpu.scan_many([(rms, td00[:3], td90[:3])], 2, 32, FP, FX, ml, 4.0, 1.0, 0.1)
