"""Real WDM equivalence across tile seams, segment padding, and parity."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from wdm_wavelet.wdm import WDM

from pycwb.modules.coherence_native import tf_batch_generation as batch


@pytest.mark.parametrize('M', [8, 16, 32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize('extra', [0, 32, 64])
def test_tiles_preserve_every_output_bit(monkeypatch, M, extra):
    wavelet = WDM(M=M, K=M, beta_order=6, precision=10)
    rng = np.random.default_rng(198)
    strains = [SimpleNamespace(data=rng.normal(size=(1024+extra)*M), sample_rate=8192.)
               for _ in range(2)]
    # Include impulses at seams and both physical boundaries.
    strains[0].data[[0, 256*M-1, 256*M, -1]] = [4., -7., 9., 3.]
    monkeypatch.setattr(batch, '_TILED_WDM', False)
    expected, _ = batch.batch_t2w_detectors(strains, wavelet)
    taps = jnp.asarray(np.asarray(wavelet.filter)[:wavelet.m_H], dtype=jnp.float64)
    actual = batch._tiled_t2w_detectors(strains, M, wavelet.m_H, taps, sample_budget=256*M)
    for a, b in zip(actual, expected):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize('n_time', [65, 1025, 1026, 1032])
def test_unaligned_shapes_retain_original_path(monkeypatch, n_time):
    wavelet = WDM(M=16, K=16, beta_order=6, precision=10)
    strain = SimpleNamespace(data=np.random.default_rng(18).normal(size=n_time*16-1),
                             sample_rate=8192.)
    monkeypatch.setattr(batch, '_TILED_WDM', False)
    expected, metadata = batch.batch_t2w_detectors([strain], wavelet)
    monkeypatch.setattr(batch, '_TILED_WDM', True)
    def forbidden(*args, **kwargs):
        raise AssertionError('unaligned input must use the original kernel shape')
    monkeypatch.setattr(batch, '_tiled_t2w_detectors', forbidden)
    actual, other_metadata = batch.batch_t2w_detectors([strain], wavelet)
    assert metadata == other_metadata
    assert actual[0].tobytes() == expected[0].tobytes()
