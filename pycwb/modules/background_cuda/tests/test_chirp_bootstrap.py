"""Bit-exact parity of ``ChirpBootstrap`` with ``chirp_micropixel._bootstrap``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pycwb.modules.background_cuda.chirp_bootstrap_plan import finish_bootstrap, prepare_bootstrap
from pycwb.modules.likelihoodWP import chirp_micropixel
from pycwb.modules.likelihoodWP.chirp_micropixel import _bootstrap, root_uniforms

from ._helpers import assert_same_bits

MINDT = 1.0 / 64.0
REFERENCE = Path(chirp_micropixel.__file__).parent / "tests" / "reference"


def _synthetic_cells(rng: np.random.Generator, count: int, slope_sign: float = -1.0) -> tuple:
    """Micropixels roughly along a chirp track with spread so trials differ."""
    t = np.sort(rng.uniform(0.0, 1.5, count))
    f = 128.0 * np.clip(slope_sign * (t - 1.6) * 40.0, 0.2, 40.0) ** (-3.0 / 8.0)
    f = np.clip(f + rng.normal(0.0, 4.0, count), 24.0, 900.0)
    energy = rng.uniform(1.0, 40.0, count)
    return t, f, energy


def _reference_cases() -> list[tuple[str, np.ndarray]]:
    cases = []
    for filename in ("chirp_real_oracle_cells.npz", "chirp_oracle.npz"):
        path = REFERENCE / filename
        if not path.exists():
            continue
        with np.load(path) as data:
            for key in data.files:
                cells = data[key]
                if cells.ndim == 2 and cells.shape[1] == 3 and len(cells) >= 13:
                    cases.append((f"{filename}:{key}", cells))
    return cases


def test_prepare_bootstrap_reproduces_native_sampling(rng: np.random.Generator) -> None:
    """The CPU plan and the native bootstrap consume the same uniform stream."""
    x, f, energy = _synthetic_cells(rng, 21)
    uniforms = root_uniforms(7, 65536)
    _, _, _, _, _, _, cursor, ready = prepare_bootstrap(x, f, energy, MINDT, uniforms)
    expected, expected_cursor = _bootstrap(x, f, energy, MINDT, uniforms)
    assert ready
    assert cursor == expected_cursor
    assert expected.shape == (5,)


def test_finish_bootstrap_small_selection_returns_zeros(rng: np.random.Generator) -> None:
    x, f, energy = _synthetic_cells(rng, 13)
    values, cursor = finish_bootstrap(x, f, energy, MINDT, -1.0, 0.5, 3, 0.2, 0.1, 99)
    np.testing.assert_array_equal(values, np.zeros(5))
    assert cursor == 99


@pytest.mark.gpu
class TestChirpBootstrap:
    @pytest.mark.parametrize("count", [13, 40, 257])
    @pytest.mark.parametrize("seed", [1, 2, 42])
    def test_synthetic_parity(self, rng: np.random.Generator, count: int, seed: int) -> None:
        from pycwb.modules.background_cuda.chirp_bootstrap import ChirpBootstrap

        x, f, energy = _synthetic_cells(rng, count, slope_sign=-1.0 if seed != 2 else 1.0)
        uniforms = root_uniforms(seed, 65536)
        expected, expected_cursor = _bootstrap(x, f, energy, MINDT, uniforms)
        actual, cursor = ChirpBootstrap()(x, f, energy, MINDT, uniforms)
        assert cursor == expected_cursor
        assert_same_bits(actual, expected)

    @pytest.mark.parametrize(("name", "cells"), _reference_cases(), ids=lambda c: c if isinstance(c, str) else "")
    @pytest.mark.parametrize("seed", [1, 42])
    def test_reference_oracle_cells_parity(self, name: str, cells: np.ndarray, seed: int) -> None:
        from pycwb.modules.background_cuda.chirp_bootstrap import ChirpBootstrap

        args = (*[np.ascontiguousarray(cells[:, i]) for i in range(3)], MINDT, root_uniforms(seed, 65536))
        expected, expected_cursor = _bootstrap(*args)
        actual, cursor = ChirpBootstrap()(*args)
        assert cursor == expected_cursor, name
        assert_same_bits(actual, expected)

    def test_exhausted_uniform_stream_returns_nan_like_cpu(self, rng: np.random.Generator) -> None:
        from pycwb.modules.background_cuda.chirp_bootstrap import ChirpBootstrap

        x, f, energy = _synthetic_cells(rng, 20)
        uniforms = root_uniforms(3, 50)
        expected, expected_cursor = _bootstrap(x, f, energy, MINDT, uniforms)
        actual, cursor = ChirpBootstrap()(x, f, energy, MINDT, uniforms)
        assert np.all(np.isnan(expected)) and np.all(np.isnan(actual))
        assert cursor == expected_cursor == len(uniforms)
        assert_same_bits(actual, expected)

    def test_rejects_fewer_than_13_micropixels(self, rng: np.random.Generator) -> None:
        from pycwb.modules.background_cuda.chirp_bootstrap import ChirpBootstrap

        x, f, energy = _synthetic_cells(rng, 12)
        with pytest.raises(ValueError, match="at least 13"):
            ChirpBootstrap()(x, f, energy, MINDT, root_uniforms(1, 1024))
        x, f, energy = _synthetic_cells(rng, 13)
        with pytest.raises(ValueError, match="at least 13"):
            ChirpBootstrap()(x, f[:12], energy, MINDT, root_uniforms(1, 1024))

    def test_make_chirp_update_binds_gpu_bootstrap(self) -> None:
        import importlib

        from pycwb.modules.background_cuda.chirp_bootstrap import ChirpBootstrap, make_chirp_update

        native = importlib.import_module("pycwb.modules.likelihoodWP.chirp_micropixel")
        update = make_chirp_update()
        assert update.__closure__ is not None
        specialized = next(
            cell.cell_contents
            for cell in update.__closure__
            if getattr(cell.cell_contents, "func", None) is native.estimate_chirp
        )
        assert isinstance(specialized.keywords["bootstrap"], ChirpBootstrap)
        assert native.estimate_chirp.__globals__["_bootstrap"] is _bootstrap
