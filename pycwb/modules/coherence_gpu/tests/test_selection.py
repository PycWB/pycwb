"""GPU pixel selection (CUDA and JAX backends) against the native selector.

Both backends are compared with ``coherence_native.selection.select_network_pixels``
on a synthetic selection cache, and the support maps are compared with the native
``_align_threshold_map_numba`` kernel.
"""

from __future__ import annotations

import numpy as np
import pytest

from pycwb.modules.coherence_native import kernels, selection
from pycwb.modules.coherence_native.kernels import _align_threshold_map_numba

from pycwb.utils.tests.gpu_helpers import assert_same_bits, selection_cache


@pytest.fixture(autouse=True, scope="module")
def _fresh_oracle_compilation() -> None:
    """Compile the native oracle in-process instead of loading its on-disk Numba cache.

    A stale ``__pycache__`` entry for ``_align_threshold_map_numba`` written by a
    dynamically loaded copy of ``kernels.py`` (module name ``<dynamic>``) makes
    the cached overload unloadable; bypassing the cache here keeps the oracle
    independent of that environment state without writing new cache files.
    """
    from numba.core.caching import NullCache

    dispatchers = [kernels._align_threshold_map_numba, kernels._select_candidates_numba]
    previous = [dispatcher._cache for dispatcher in dispatchers]
    for dispatcher in dispatchers:
        dispatcher._cache = NullCache()
    yield
    for dispatcher, cache in zip(dispatchers, previous, strict=True):
        dispatcher._cache = cache


THRESHOLD = 3.0
LAG_SHIFTS = {
    2: [[0.0, 0.0], [0.0, 1.0625], [2.5, 0.0]],
    3: [[0.0, 0.0, 0.0], [0.0, 1.0625, 0.25], [2.5, 0.0, 6.125]],
}


def _veto(n_time: int) -> np.ndarray:
    veto = np.ones(n_time, np.int16)
    veto[[3, 8, 20, 21, 40]] = 0
    return veto


def _native(cache: dict, lag_shifts: list[float], veto: np.ndarray | None) -> dict:
    return selection.select_network_pixels(
        None,
        0,
        THRESHOLD,
        lag_shifts=lag_shifts,
        veto=veto,
        edge=0.0,
        selection_cache=cache,
    )


def _assert_payload_matches(actual: dict, expected: dict) -> None:
    for key in ("frequency", "time", "pix_det_index"):
        np.testing.assert_array_equal(
            np.asarray(actual[key]), expected[key], err_msg=key
        )
    for key in ("energy", "pix_det_energy"):
        assert_same_bits(actual[key], expected[key])
    np.testing.assert_array_equal(actual["mask"], expected["mask"])
    np.testing.assert_array_equal(actual["live_mask"], expected["live_mask"])
    assert actual["live_samples"] == expected["live_samples"]
    for key in ("rate", "layers", "start", "stop", "f_low", "f_high"):
        assert actual[key] == expected[key], key


@pytest.mark.gpu
class TestSelectionSession:
    @pytest.mark.parametrize("n_ifo", [2, 3])
    @pytest.mark.parametrize("veto_on", [False, True])
    def test_select_matches_native_selector(
        self, rng: np.random.Generator, n_ifo: int, veto_on: bool
    ) -> None:
        from pycwb.modules.coherence_gpu.selection_cuda import SelectionSession

        cache = selection_cache(rng, n_ifo)
        veto = _veto(cache["n_time"]) if veto_on else None
        session = SelectionSession(
            cache["arrays_stack"],
            cache["valid_start"],
            cache["nn_valid"],
            cache["ib"],
            veto,
        )
        for lag_shifts in LAG_SHIFTS[n_ifo]:
            expected = _native(cache, lag_shifts, veto)
            shifts = selection._shift_bins_from_lag_shifts(
                lag_shifts, n_ifo, cache["rate"]
            )
            payload, live = session.select(
                shifts, THRESHOLD, cache["ie"], cache["edge_bins"], capacity=4096
            )
            frequency, time, energy, det_energy, det_index = payload
            assert len(frequency) > 0, "synthetic case must select pixels"
            np.testing.assert_array_equal(frequency, expected["frequency"])
            np.testing.assert_array_equal(time, expected["time"])
            assert_same_bits(energy, expected["energy"])
            assert_same_bits(det_energy, expected["pix_det_energy"])
            np.testing.assert_array_equal(det_index, expected["pix_det_index"])
            np.testing.assert_array_equal(live, expected["live_mask"])
            # The support map is the native aligned/clipped map.
            combined, live_mask = _align_threshold_map_numba(
                cache["arrays_stack"],
                shifts,
                cache["valid_start"],
                cache["nn_valid"],
                veto if veto_on else np.zeros(0, np.int16),
                veto_on,
                cache["edge_bins"],
                cache["ib"],
                cache["ie"],
                THRESHOLD,
                2.0 * THRESHOLD,
            )
            assert_same_bits(session.support.copy_to_host(), combined)
            np.testing.assert_array_equal(live_mask, live)

    def test_overflow_reports_required_count(self, rng: np.random.Generator) -> None:
        from pycwb.modules.coherence_gpu.selection_cuda import SelectionSession

        cache = selection_cache(rng, 2)
        session = SelectionSession(
            cache["arrays_stack"], cache["valid_start"], cache["nn_valid"], cache["ib"]
        )
        shifts = np.zeros(2, np.int64)
        expected = _native(cache, [0.0, 0.0], None)
        count = len(expected["frequency"])
        assert count > 2
        with pytest.raises(
            OverflowError, match=f"{count} selected pixels exceed capacity 2"
        ):
            session.select(
                shifts, THRESHOLD, cache["ie"], cache["edge_bins"], capacity=2
            )
        payload, _ = session.select(
            shifts, THRESHOLD, cache["ie"], cache["edge_bins"], capacity=count
        )
        np.testing.assert_array_equal(payload[0], expected["frequency"])

    def test_constructor_and_select_validation(self, rng: np.random.Generator) -> None:
        from pycwb.modules.coherence_gpu.selection_cuda import SelectionSession

        cache = selection_cache(rng, 2)
        maps = cache["arrays_stack"]
        with pytest.raises(ValueError, match="FP64"):
            SelectionSession(maps.astype(np.float32), 2, 60, 1)
        with pytest.raises(ValueError, match="Invalid support bounds"):
            SelectionSession(maps, 2, 63, 1)
        with pytest.raises(ValueError, match="Invalid support bounds"):
            SelectionSession(maps, 0, 64, maps.shape[1] + 1)
        with pytest.raises(ValueError, match="Veto must match"):
            SelectionSession(maps, 2, 60, 1, np.ones(5, np.int16))
        session = SelectionSession(maps, 2, 60, 1)
        with pytest.raises(ValueError, match="integer detector shifts"):
            session.select(np.zeros(2, np.float64), THRESHOLD, 8, 2)
        with pytest.raises(ValueError, match="integer detector shifts"):
            session.select(np.zeros(3, np.int64), THRESHOLD, 8, 2)
        with pytest.raises(ValueError, match="finite and nonnegative"):
            session.select(np.zeros(2, np.int64), -1.0, 8, 2)
        with pytest.raises(ValueError, match="Invalid sparse output capacity"):
            session.select(np.zeros(2, np.int64), THRESHOLD, 8, 2, capacity=0)
        with pytest.raises(MemoryError, match="256 MiB"):
            session.select(np.zeros(2, np.int64), THRESHOLD, 8, 2, capacity=2**30)

    def test_shared_module_between_sessions(self, rng: np.random.Generator) -> None:
        from pycwb.modules.coherence_gpu.selection_cuda import SelectionSession

        cache = selection_cache(rng, 2)
        first = SelectionSession(cache["arrays_stack"], 2, 60, 1)
        second = SelectionSession(cache["arrays_stack"], 2, 60, 1, None, first.module)
        assert second.module is first.module


@pytest.mark.gpu
class TestGPUSelectorCUDA:
    @pytest.mark.parametrize("n_ifo", [2, 3])
    def test_selector_payload_matches_native(
        self, rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch, n_ifo: int
    ) -> None:

        from pycwb.workflow.subflow.process_job_segment_gpu import GPUSelector

        cache = selection_cache(rng, n_ifo)
        veto = _veto(cache["n_time"])
        selector = GPUSelector({"selection_cuda": True})
        assert selector.use_cuda
        for lag_shifts in LAG_SHIFTS[n_ifo]:
            expected = _native(cache, lag_shifts, veto)
            actual = selector(
                None, 0, THRESHOLD, lag_shifts, veto, selection_cache=cache
            )
            _assert_payload_matches(actual, expected)
        assert len(selector.sessions) == 1
        # A veto of the wrong length is ignored like the native selector does.
        expected = _native(cache, LAG_SHIFTS[n_ifo][1], np.zeros(3, np.int16))
        actual = selector(
            None,
            0,
            THRESHOLD,
            LAG_SHIFTS[n_ifo][1],
            np.zeros(3, np.int16),
            selection_cache=cache,
        )
        _assert_payload_matches(actual, expected)
        assert len(selector.sessions) == 1

    def test_selector_requires_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:

        from pycwb.workflow.subflow.process_job_segment_gpu import GPUSelector

        with pytest.raises(ValueError, match="requires a prepared selection cache"):
            GPUSelector({"selection_cuda": True})(
                None, 0, THRESHOLD, [0.0, 0.0], None, selection_cache=None
            )

    def test_selector_uses_preindexed_shift_table(
        self, rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        from pycwb.workflow.subflow.process_job_segment_gpu import GPUSelector

        cache = selection_cache(rng, 2)
        rows = LAG_SHIFTS[2]
        cache["shift_bins_by_lag"] = np.vstack(
            [selection._shift_bins_from_lag_shifts(r, 2, cache["rate"]) for r in rows]
        ).astype(np.int64)
        selector = GPUSelector({"selection_cuda": True})
        for lag_index in range(len(rows)):
            expected = selection.select_network_pixels(
                None, lag_index, THRESHOLD, selection_cache=cache, preindex_shifts=True
            )
            actual = selector(
                None,
                lag_index,
                THRESHOLD,
                None,
                None,
                selection_cache=cache,
                preindex_shifts=True,
            )
            _assert_payload_matches(actual, expected)
            assert len(actual["frequency"]) > 0


@pytest.mark.jax_gpu
class TestAlignmentSession:
    @pytest.mark.parametrize("n_ifo", [2, 3])
    @pytest.mark.parametrize("veto_on", [False, True])
    @pytest.mark.parametrize("length", [0, 9])
    def test_align_matches_native_support_map(
        self, rng: np.random.Generator, n_ifo: int, veto_on: bool, length: int
    ) -> None:
        import jax

        from pycwb.modules.coherence_gpu.alignment_jax import AlignmentSession

        maps = rng.uniform(-1.0, 8.0, (n_ifo, 6, 13))
        maps[:, 2, :6] = 0.0
        maps[0, 2, :6] = [
            3.0,
            np.nextafter(3.0, 0.0),
            np.nextafter(3.0, 4.0),
            6.0,
            np.nextafter(6.0, 0.0),
            6.5,
        ]
        veto = np.ones(13, np.int16)
        veto[[3, 8]] = 0
        shifts = np.array([[0] * n_ifo, [-17, 100, 5][:n_ifo], [1] * n_ifo], np.int64)
        session = AlignmentSession(maps, 2, length, 1, veto if veto_on else None)
        with jax.default_device(session.device):
            out, live = session.align(shifts, 3.0)
        out, live = np.asarray(out), np.asarray(live)
        assert out.shape == (3, 6, 13) and live.shape == (3, 13)
        for lag, shift in enumerate(shifts):
            expected, expected_live = _align_threshold_map_numba(
                maps, shift, 2, length, veto, veto_on, 2, 1, 4, 3.0, 6.0
            )
            assert_same_bits(out[lag], expected)
            np.testing.assert_array_equal(live[lag], expected_live)

    def test_budget_and_contract(self) -> None:
        from pycwb.modules.coherence_gpu.alignment_jax import AlignmentSession

        maps = np.zeros((2, 3, 8), np.float64)
        with pytest.raises(ValueError, match="scratch budget"):
            AlignmentSession(maps, 0, 8, 1, scratch_bytes=1)
        with pytest.raises(ValueError, match="float64"):
            AlignmentSession(maps.astype(np.float32), 0, 8, 1)
        with pytest.raises(ValueError, match="time interval"):
            AlignmentSession(maps, 1, 8, 1)
        with pytest.raises(ValueError, match="frequency bound"):
            AlignmentSession(maps, 0, 8, 4)
        with pytest.raises(ValueError, match="veto shape"):
            AlignmentSession(maps, 0, 8, 1, np.ones(3, np.int16))
        session = AlignmentSession(maps, 0, 8, 1, scratch_bytes=200)
        assert session.max_batch == 1
        with pytest.raises(ValueError, match="scratch budget"):
            session.align(np.zeros((2, 2), np.int64), 1.0)
        with pytest.raises(ValueError, match="integer"):
            session.align(np.zeros((1, 2), np.float64), 1.0)
        with pytest.raises(ValueError, match="finite and nonnegative"):
            session.align(np.zeros((1, 2), np.int64), float("nan"))

    @pytest.mark.parametrize("n_ifo", [2, 3])
    def test_jax_selector_payload_matches_native(
        self, rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch, n_ifo: int
    ) -> None:
        monkeypatch.delenv("PYCWB_GPU_SELECTION_CUDA", raising=False)
        from pycwb.workflow.subflow.process_job_segment_gpu import GPUSelector

        cache = selection_cache(rng, n_ifo)
        veto = _veto(cache["n_time"])
        selector = GPUSelector()
        assert not selector.use_cuda
        for lag_shifts in LAG_SHIFTS[n_ifo]:
            expected = _native(cache, lag_shifts, veto)
            actual = selector(
                None, 0, THRESHOLD, lag_shifts, veto, selection_cache=cache
            )
            _assert_payload_matches(actual, expected)
        expected = _native(cache, LAG_SHIFTS[n_ifo][0], None)
        actual = selector(
            None, 0, THRESHOLD, LAG_SHIFTS[n_ifo][0], None, selection_cache=cache
        )
        _assert_payload_matches(actual, expected)
        selector.sessions.clear()
