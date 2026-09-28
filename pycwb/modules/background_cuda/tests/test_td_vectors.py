"""Bit-exact parity of the CUDA TD-vector session with ``TDBatchInputs.extract_td_vecs``."""

from __future__ import annotations

import numpy as np
import pytest

from pycwb.types.td_batch_inputs import TDBatchInputs

pytestmark = pytest.mark.gpu

TAPS = 6
N_TIME = 512


def _inputs(
    rng: np.random.Generator, M: int, offset: int, stop: int
) -> tuple[TDBatchInputs, np.ndarray]:
    J = 4 * M
    planes = [
        rng.normal(size=(N_TIME + 2 * TAPS, M + 1)).astype(np.float32) for _ in range(2)
    ]
    tables = [rng.normal(size=(2 * J + 1, 2 * TAPS + 1)) for _ in range(2)]
    inputs = TDBatchInputs(
        *(p[:, offset:stop].copy() for p in planes), *tables, M, TAPS, J, offset
    )
    bands = np.arange(max(0, offset + (offset > 0)), min(M + 1, stop - (stop < M + 1)))
    # Both time parities, all cached bands, repeated unsorted indices.
    indices = (rng.integers(180, 300, 256) * (M + 1) + rng.choice(bands, 256)).astype(
        np.int32
    )
    return inputs, indices


def _band_limits(M: int) -> list[tuple[int, int]]:
    return [(0, M + 1), (max(0, M // 4 - 1), min(M + 1, 3 * M // 4 + 2))]


@pytest.mark.parametrize("M", [8, 32])
@pytest.mark.parametrize("band", [0, 1], ids=["full_band", "band_limited"])
@pytest.mark.parametrize(("K", "stride"), [(0, 1), (16, 1), (16, 4), (64, 1)])
def test_extract_matches_cpu(
    rng: np.random.Generator,
    monkeypatch: pytest.MonkeyPatch,
    M: int,
    band: int,
    K: int,
    stride: int,
) -> None:
    from pycwb.modules.background_cuda.td_vectors import TDSession

    offset, stop = _band_limits(M)[band]
    inputs, indices = _inputs(rng, M, offset, stop)
    expected = inputs.extract_td_vecs(indices, K, delay_stride=stride)
    session = TDSession(inputs)
    assert session.workspace is None
    actual = session.extract_td_vecs(indices, K, delay_stride=stride)
    assert actual.shape == expected.shape == (len(indices), 4 * K + 2)
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("M", [8, 128])
def test_extract_with_reused_workspace(
    rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch, M: int
) -> None:
    from pycwb.modules.background_cuda.td_vectors import TDSession
    from pycwb.modules.background_cuda.workspace import Workspace

    offset, stop = _band_limits(M)[1]
    inputs, indices = _inputs(rng, M, offset, stop)
    session = TDSession(inputs, options={"reuse_td_workspace": True})
    assert isinstance(session.workspace, Workspace)
    assert session.workspace.budget == 384 * 1024**2
    for K, stride in ((16, 1), (2, 3), (64, 1), (0, 1)):
        expected = inputs.extract_td_vecs(indices, K, delay_stride=stride)
        actual = session.extract_td_vecs(indices, K, delay_stride=stride)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert set(session.workspace.slots) == {"output", "indices"}


def test_shared_module_and_explicit_workspace(
    rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pycwb.modules.background_cuda.td_vectors import TDSession
    from pycwb.modules.background_cuda.workspace import Workspace

    workspace = Workspace(budget=64 * 1024**2)
    first_inputs, first_indices = _inputs(rng, 8, 0, 9)
    second_inputs, second_indices = _inputs(rng, 32, 7, 26)
    first = TDSession(first_inputs, None, workspace)
    second = TDSession(second_inputs, first.module, workspace)
    assert second.module is first.module
    assert first.workspace is second.workspace is workspace
    for session, inputs, indices in (
        (first, first_inputs, first_indices),
        (second, second_inputs, second_indices),
    ):
        expected = inputs.extract_td_vecs(indices, 16)
        np.testing.assert_array_equal(
            session.extract_td_vecs(indices, 16).view(np.uint32),
            expected.view(np.uint32),
        )


def test_validate_td_flag_repeats_cpu_check(
    rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pycwb.modules.background_cuda.td_vectors import TDSession

    inputs, indices = _inputs(rng, 8, 0, 9)
    session = TDSession(inputs, options={"validate_td": True})
    result = session.extract_td_vecs(indices, 8)
    assert result.shape == (256, 34)


def test_empty_index_vector(rng: np.random.Generator) -> None:
    from pycwb.modules.background_cuda.td_vectors import TDSession

    inputs, _ = _inputs(rng, 8, 0, 9)
    result = TDSession(inputs).extract_td_vecs(np.empty(0, np.int32), 4)
    assert result.shape == (0, 18) and result.dtype == np.float32


def test_input_validation(rng: np.random.Generator) -> None:
    from pycwb.modules.background_cuda.td_vectors import TDSession

    inputs, indices = _inputs(rng, 8, 0, 9)
    session = TDSession(inputs)
    with pytest.raises(ValueError, match="Invalid TD delay range"):
        session.extract_td_vecs(indices, -1)
    with pytest.raises(ValueError, match="Invalid TD delay range"):
        session.extract_td_vecs(indices, 4, delay_stride=0)
    with pytest.raises(ValueError, match="nonnegative vector"):
        session.extract_td_vecs(np.array([-1], np.int32), 4)
    with pytest.raises(ValueError, match="padded time range"):
        # Time index N_TIME needs taps beyond the padded plane.
        session.extract_td_vecs(np.array([N_TIME * (inputs.M + 1)], np.int32), 4)
    limited, _ = _inputs(rng, 32, 7, 26)
    with pytest.raises(ValueError, match="outside cached frequency bands"):
        TDSession(limited).extract_td_vecs(np.array([200 * 33 + 2], np.int32), 4)
    bad = TDBatchInputs(
        inputs.padded00.astype(np.float64),
        inputs.padded90,
        inputs.T0,
        inputs.Tx,
        inputs.M,
        TAPS,
        inputs.J,
        0,
    )
    with pytest.raises(ValueError, match="FP32 planes"):
        TDSession(bad)
    bad = TDBatchInputs(
        inputs.padded00,
        inputs.padded90,
        inputs.T0[:-1],
        inputs.Tx,
        inputs.M,
        TAPS,
        inputs.J,
        0,
    )
    with pytest.raises(ValueError, match="filter dimensions"):
        TDSession(bad)


def test_gpu_time_delays_binds_native_populate(monkeypatch: pytest.MonkeyPatch) -> None:
    from pycwb.modules.background_cuda.td_vectors import GPUTimeDelays

    delays = GPUTimeDelays()
    assert (
        delays.sessions == {}
        and delays.resident_bytes == 0
        and delays.workspace is None
    )

    assert GPUTimeDelays({"reuse_td_workspace": True}).workspace is not None
