"""GPU sky-mask contract tests for likelihoodWP setup consumers."""

import importlib.util

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="JAX is not installed",
)


def test_gpu_dpf_regulator_uses_valid_sky_indices(monkeypatch):
    """The GPU DPF regulator should count only unmasked sky directions."""
    from pycwb.modules.likelihoodWPGPU import dpf as gpu_dpf

    def fake_dpf_quality_single(fp_sky, fx_sky, rms):
        return fp_sky[0]

    monkeypatch.setattr(gpu_dpf, "_dpf_quality_single", fake_dpf_quality_single)

    fp = np.arange(5, dtype=np.float32).reshape(5, 1)
    fx = np.zeros_like(fp)
    rms = np.ones((1, 1), dtype=np.float32)
    valid = np.array([1, 3], dtype=np.int64)

    regulator = gpu_dpf.calculate_dpf_regulator(
        fp, fx, rms,
        gamma_regulator=2.0,
        network_energy_threshold=1.0,
        sky_batch_size=1,
        sky_valid_indices=valid,
    )

    assert regulator == pytest.approx(3.0)


def test_gpu_sky_scan_uses_valid_sky_indices(monkeypatch):
    """Masked sky directions must not win l_max or populate skymap stats."""
    from pycwb.modules.likelihoodWPGPU import sky_scan

    def fake_sky_direction_statistics(
        fp_sky, fx_sky, rms, v00, v90, REG, netCC,
        delta_regulator, energy_threshold, n_ifo,
    ):
        score = fp_sky[0]
        return {
            "AA": score,
            "antenna_prior": score,
            "alignment": score,
            "likelihood": score,
            "null_energy": score,
            "coherent_energy": score,
            "correlation": score,
            "sky_stat": score,
            "disbalance": score,
            "net_index": score,
            "ellipticity": score,
            "polarisation": score,
        }

    monkeypatch.setattr(sky_scan, "_sky_direction_statistics", fake_sky_direction_statistics)

    n_ifo = 1
    n_pix = 1
    n_sky = 3
    fp = np.array([[1.0], [10.0], [3.0]], dtype=np.float32)
    fx = np.zeros_like(fp)
    rms = np.ones((n_pix, n_ifo), dtype=np.float32)
    td00 = np.zeros((1, n_ifo, n_pix), dtype=np.float32)
    td90 = np.zeros_like(td00)
    ml = np.zeros((n_ifo, n_sky), dtype=np.int32)
    reg = np.zeros(3, dtype=np.float32)
    valid = np.array([0, 2], dtype=np.int64)

    result = sky_scan.find_optimal_sky_localization(
        n_ifo, n_pix, n_sky, fp, fx, rms, td00, td90, ml, reg,
        netCC=0.0,
        delta_regulator=0.0,
        network_energy_threshold=0.0,
        sky_batch_size=1,
        sky_valid_indices=valid,
    )

    l_max = result[0]
    n_sky_stat = result[7]
    sky_stat_max = result[-1]

    assert l_max == 2
    assert sky_stat_max == pytest.approx(3.0)
    assert n_sky_stat.tolist() == pytest.approx([1.0, 0.0, 3.0])
