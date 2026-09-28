"""Exact-value and storage-ownership contracts for compact coherence setup."""

from pycwb.constants.execution_profile import ExecutionProfile

from types import SimpleNamespace
import numpy as np
import pytest
from pycwb.modules.coherence_native import setup as coherence_setup_module
from pycwb.modules.coherence_native import tf_batch_generation as batch


def test_real_energy_views_share_storage_without_changing_values():
    arrays = [np.arange(20, dtype="f8").reshape(4, 5), np.asfortranarray(np.arange(20, dtype="f8").reshape(4, 5))]
    arrays[0][0, 0] = -0.0
    arrays[1][1, 1] = np.nan
    maps = [SimpleNamespace(data=a) for a in arrays]
    stack = np.stack(arrays)
    coherence_setup_module._share_prepared_energy_storage(maps, {"arrays_stack": stack})
    for i, m in enumerate(maps):
        assert m.data.dtype == arrays[i].dtype
        assert m.data.tobytes() == arrays[i].tobytes()
        assert np.shares_memory(m.data, stack)
        assert m.data.flags.c_contiguous


def test_complex_and_other_dtypes_retain_their_original_storage():
    arrays = [np.ones((3, 4), dtype="c16") * (1 + 2j), np.ones((3, 4), dtype="f4")]
    maps = [SimpleNamespace(data=a) for a in arrays]
    coherence_setup_module._share_prepared_energy_storage(
        maps, {"arrays_stack": np.stack([a.real for a in arrays]).astype("f8")}
    )
    assert all(m.data is a for m, a in zip(maps, arrays))


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("n_ifo", [1, 2, 3])
def test_batch_output_bit_exact_for_both_transform_paths(monkeypatch, fallback, n_ifo):
    rng = np.random.default_rng(124)
    raw = rng.normal(size=(n_ifo, 2, 128, 5)).astype("f8")
    raw[0, 0, 0, 0] = -0.0
    original = raw.copy()
    strains = [SimpleNamespace(data=rng.normal(size=512), sample_rate=4096.0) for _ in range(n_ifo)]
    wavelet = SimpleNamespace(M=4, m_H=2, filter=np.array([1.0, 0.5]))
    monkeypatch.setattr(batch, "_LOW_LEVEL_T2W_JAX_IMPL", None if fallback else object())
    monkeypatch.setattr(batch, "_batch_t2w_impl", lambda *a: raw)
    monkeypatch.setattr(batch, "_batch_t2w_fallback", lambda *a: raw)
    expected, meta = batch.batch_t2w_detectors(strains, wavelet)
    actual, other = batch.batch_t2w_detectors(strains, wavelet, profile=ExecutionProfile(compact_coherence=True))
    assert meta == other
    for a, b in zip(actual, expected):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes()
    assert raw.tobytes() == original.tobytes()
