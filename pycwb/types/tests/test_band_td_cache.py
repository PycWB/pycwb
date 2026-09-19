"""Cropping must preserve TD vector bits, global phase and whole-bin shifts."""

from types import SimpleNamespace
import numpy as np
import pytest
from wdm_wavelet.wdm import WDM
from pycwb.types.time_frequency_map import TimeFrequencyMap


@pytest.mark.parametrize("M", [8, 16, 32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("stride", [1, 4])
def test_cropped_filter_support_exact(monkeypatch, M, stride):
    monkeypatch.setenv("PYCWB_COMPACT_TD_CACHE", "1")
    w = WDM(M=M, K=M, beta_order=6, precision=10)
    w.set_td_filter(12, 4)
    rng = np.random.default_rng(875 + M)
    a = rng.normal(size=(M + 1, 128)) + 1j * rng.normal(size=(M + 1, 128))
    tf = TimeFrequencyMap(a, True, 1.0, 1.0, 0.0, 128.0, 0.0, float(M), None, None)
    full = tf.prepare_td_inputs(w.td_filters)
    low = M // 4
    high = 3 * M // 4 + 1
    cropped = tf.prepare_td_inputs(w.td_filters, frequency_bounds=(low, high))
    assert not np.shares_memory(cropped.padded00, a)
    assert cropped.padded00.nbytes < full.padded00.nbytes
    bands = np.array([low + 1, high - 2, (low + high) // 2])
    indices = (np.array([40, 41, 70]) * (M + 1) + bands).astype(np.int32)
    K = 2 * M + 3 if M <= 16 else 19
    x = full.extract_td_vecs(indices, K, delay_stride=stride)
    y = cropped.extract_td_vecs(indices, K, delay_stride=stride)
    assert x.tobytes() == y.tobytes()
    with pytest.raises(ValueError, match="support"):
        cropped.extract_td_vecs(np.array([40 * (M + 1) + low]), K)


@pytest.mark.parametrize("bounds", [(0, 9), (0, 5), (4, 9)])
def test_global_dc_nyquist_edges(monkeypatch, bounds):
    monkeypatch.setenv("PYCWB_COMPACT_TD_CACHE", "1")
    M = 8
    w = WDM(M=M, K=M, beta_order=6, precision=10)
    w.set_td_filter(12, 4)
    rng = np.random.default_rng(784)
    a = rng.normal(size=(9, 128)) + 1j * rng.normal(size=(9, 128))
    tf = TimeFrequencyMap(a, True, 1.0, 1.0, 0.0, 128.0, 0.0, 8.0, None, None)
    full = tf.prepare_td_inputs(w.td_filters)
    crop = tf.prepare_td_inputs(w.td_filters, frequency_bounds=bounds)
    band = 0 if bounds[0] == 0 else 8
    indices = np.array([60 * 9 + band, 61 * 9 + band], dtype=np.int32)
    assert full.extract_td_vecs(indices, 19, 4).tobytes() == crop.extract_td_vecs(indices, 19, 4).tobytes()
