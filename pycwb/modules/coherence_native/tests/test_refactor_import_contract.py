"""Characterization tests for the coherence-native refactor contract."""

from __future__ import annotations

import os
import sys
import importlib
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from scipy.special import gammaincc, gammainccinv


try:
    import wdm_wavelet.wdm  # noqa: F401
except Exception:
    wdm_package = ModuleType("wdm_wavelet")
    wdm_package.__path__ = []
    wdm_module = ModuleType("wdm_wavelet.wdm")
    wdm_module.WDM = object
    wdm_module.t2w_jax = lambda *args, **kwargs: None
    wdm_module.w2t_jax = lambda *args, **kwargs: None
    wdm_module.t2w_numba = lambda *args, **kwargs: None
    wdm_module.w2t_numba = lambda *args, **kwargs: None
    sys.modules.setdefault("wdm_wavelet", wdm_package)
    sys.modules.setdefault("wdm_wavelet.wdm", wdm_module)


def _tf_map(
    data, *, dt=1.0, df=1.0, start=0.0, f_low=0.0, f_high=None, wavelet_rate=1.0
):
    data = np.asarray(data)
    return SimpleNamespace(
        data=data,
        dt=dt,
        df=df,
        start=start,
        stop=start + data.shape[-1] * dt,
        f_low=f_low,
        f_high=float(data.shape[0] - 1) * df
        if f_high is None and data.ndim == 2
        else f_high,
        wavelet_rate=wavelet_rate,
    )


def test_coherence_facade_public_and_private_aliases():
    import pycwb.modules.coherence_native as package
    from pycwb.modules.coherence_native import (
        cluster_pixels,
        coherence,
        coherence_single_lag,
        compute_threshold,
        max_energy,
        select_network_pixels,
        setup_coherence,
    )
    from pycwb.modules.coherence_native import (
        clustering,
        pipeline,
        projection,
        selection,
        setup,
        veto_threshold,
    )

    facade = importlib.import_module("pycwb.modules.coherence_native.coherence")

    assert package.__all__ == [
        "coherence",
        "setup_coherence",
        "coherence_single_lag",
        "max_energy",
        "compute_threshold",
        "apply_veto",
        "select_network_pixels",
        "cluster_pixels",
        "LagPlan",
        "build_lag_plan_from_config",
    ]
    assert coherence is pipeline.coherence
    assert coherence_single_lag is pipeline.coherence_single_lag
    assert setup_coherence is setup.setup_coherence
    assert max_energy is projection.max_energy
    assert compute_threshold is veto_threshold.compute_threshold
    assert select_network_pixels is selection.select_network_pixels
    assert cluster_pixels is clustering.cluster_pixels
    assert facade._setup_coherence_single_res is setup._setup_coherence_single_res
    assert facade._build_selection_cache is selection._build_selection_cache


def test_time_delay_facade_aliases_backend_modules():
    from pycwb.modules.coherence_native import (
        time_delay_jax,
        time_delay_numba,
        time_delay_packet,
    )
    from pycwb.modules.coherence_native import time_delay_max_energy as facade

    assert facade.__all__ == ["time_delay_max_energy", "time_delay_max_energy_numba"]
    assert facade.time_delay_max_energy is time_delay_jax.time_delay_max_energy
    assert (
        facade.time_delay_max_energy_numba
        is time_delay_numba.time_delay_max_energy_numba
    )
    assert (
        facade._compute_packet_energy_params
        is time_delay_packet._compute_packet_energy_params
    )


def test_projection_backend_dispatch(monkeypatch):
    from pycwb.modules.coherence_native import projection

    calls = []

    def fake_jax(tf_map, max_delay, *, downsample, pattern, hist):
        calls.append(("jax", max_delay, downsample, pattern, hist))
        return "jax-map", 1.0

    def fake_numba(tf_map, max_delay, *, downsample, pattern, hist):
        calls.append(("numba", max_delay, downsample, pattern, hist))
        return "numba-map", 2.0

    monkeypatch.setattr(projection, "time_delay_max_energy", fake_jax)
    monkeypatch.setattr(projection, "time_delay_max_energy_numba", fake_numba)

    tf_map = SimpleNamespace(
        wavelet=SimpleNamespace(M=64),
        bandpass=lambda **kwargs: calls.append(("bandpass", kwargs)),
    )

    assert projection.max_energy(
        tf_map, 0.1, 2, 3, f_low=10.0, f_high=100.0, hist=[], backend="xla"
    ) == ("jax-map", 1.0)
    assert calls[0] == ("bandpass", {"f_low": 10.0, "f_high": 100.0})
    assert calls[1] == ("jax", 0.1, 2, 3, [])

    calls.clear()
    assert projection.max_energy(tf_map, 0.2, 4, 5, backend="auto") == (
        "numba-map",
        2.0,
    )
    assert calls[-1] == ("numba", 0.2, 4, 5, None)


def _expected_shape_threshold(tf_maps, bpp, alp, edge):
    n_ifo = len(tf_maps)
    pw0 = tf_maps[0]
    arr0 = np.asarray(pw0.data, dtype=np.float64)
    if np.iscomplexobj(arr0):
        arr0 = arr0.real
    m_layers = int(arr0.shape[0]) if arr0.ndim == 2 else 1
    work = arr0.T.ravel().copy() if arr0.ndim == 2 else arr0.ravel().copy()
    for tfm in tf_maps[1:]:
        arr = np.asarray(tfm.data, dtype=np.float64)
        if np.iscomplexobj(arr):
            arr = arr.real
        work += arr.T.ravel() if arr.ndim == 2 else arr.ravel()

    n_left = int(float(edge or 0.0) * pw0.wavelet_rate * m_layers)
    n_right = int(work.size) - n_left - 1
    positive = np.clip(work[n_left:n_right], 0.0, n_ifo * 100.0)
    positive = positive[positive > 1.0e-3]
    avg = float(np.mean(positive))
    log_delta = np.log(avg) - float(np.mean(np.log(positive)))
    alp_fit = (
        3 - log_delta + np.sqrt((log_delta - 3) * (log_delta - 3) + 24 * log_delta)
    ) / (12 * log_delta)
    bpp_corr = float(bpp) * alp_fit / float(alp)
    return avg * float(gammainccinv(alp_fit, bpp_corr)) / alp_fit / 2.0


def _expected_no_shape_threshold(tf_maps, bpp, edge):
    n_ifo = len(tf_maps)
    arrays = []
    for tfm in tf_maps:
        arr = np.asarray(tfm.data, dtype=np.float64)
        crop = int(max(0, round(float(edge) / float(tfm.dt))))
        if crop > 0 and arr.shape[1] > 2 * crop:
            arr = arr[:, crop:-crop]
        arrays.append(arr)
    work = np.clip(np.sum(arrays, axis=0).ravel(), 0.0, n_ifo * 100.0)
    fill_fraction = float(np.sum(work > 1.0e-4) / work.size)
    sorted_work = np.sort(work)
    n_total = work.size
    k_val = int(float(bpp) * fill_fraction * n_total)
    k_med = int(0.2 * fill_fraction * n_total)
    val = (
        float(sorted_work[max(0, n_total - k_val - 1)])
        if k_val > 0
        else float(sorted_work[-1])
    )
    med = (
        float(sorted_work[max(0, n_total - k_med - 1)])
        if k_med > 0
        else float(sorted_work[-1])
    )
    shape = 1.0
    probability = 0.0
    while probability < 0.2:
        probability = float(gammaincc(n_ifo * shape, med))
        shape += 0.01
    if shape > 1.01:
        shape -= 0.01
    return 0.3 * (
        float(gammainccinv(n_ifo * shape, float(bpp))) + val
    ) + n_ifo * np.log(shape)


def test_compute_threshold_characterization_shape_and_no_shape_paths():
    from pycwb.modules.coherence_native.veto_threshold import compute_threshold

    first = _tf_map(np.arange(1, 19, dtype=float).reshape(3, 6), wavelet_rate=1.0)
    second = _tf_map(np.arange(2, 20, dtype=float).reshape(3, 6), wavelet_rate=1.0)

    assert compute_threshold(
        [first, second], bpp=0.15, alp=2.0, edge=1.0
    ) == pytest.approx(
        _expected_shape_threshold([first, second], bpp=0.15, alp=2.0, edge=1.0)
    )
    assert compute_threshold(
        [first, second], bpp=0.15, alp=None, edge=1.0
    ) == pytest.approx(
        _expected_no_shape_threshold([first, second], bpp=0.15, edge=1.0)
    )


def test_build_veto_mask_and_apply_veto_characterization():
    from pycwb.modules.coherence_native.veto_threshold import (
        apply_veto,
        build_veto_mask,
    )

    tf_map = _tf_map(np.zeros((2, 10)), dt=1.0, start=0.0)
    mask = build_veto_mask(tf_map, [(2.2, 5.9), (8.0, 12.0)])

    np.testing.assert_array_equal(
        mask, np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1], dtype=np.int16)
    )

    live_time, live_mask = apply_veto(
        tf_map, tw=1.0, segment_list=[(2.0, 8.0)], edge=1.0, return_mask=True
    )
    assert live_time == pytest.approx(6.0)
    np.testing.assert_array_equal(
        live_mask, np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0], dtype=np.int16)
    )


def _pixel_payload(points, *, shape=(6, 10), n_ifo=1):
    freq = np.array([p[0] for p in points], dtype=np.int64)
    time = np.array([p[1] for p in points], dtype=np.int64)
    mask = np.zeros(shape, dtype=bool)
    mask[freq, time] = True
    n_pix = len(points)
    return {
        "mask": mask,
        "time": time,
        "frequency": freq,
        "energy": np.full(n_pix, 4.0, dtype=np.float64),
        "pix_det_energy": np.full((n_pix, n_ifo), 4.0, dtype=np.float64),
        "pix_det_index": np.arange(n_pix * n_ifo, dtype=np.int64).reshape(n_pix, n_ifo),
        "rate": 2.0,
        "layers": shape[0],
        "start": 0.0,
        "stop": 5.0,
        "f_low": 0.0,
        "f_high": 10.0,
    }


def test_cluster_pixels_empty_single_and_multiple_components():
    from pycwb.modules.coherence_native.clustering import cluster_pixels

    empty = cluster_pixels(_pixel_payload([]))
    assert empty.event_count() == 0
    assert empty.pixel_count() == 0

    one = cluster_pixels(_pixel_payload([(2, 3), (2, 4)]), kt=1, kf=1)
    assert one.event_count() == 1
    assert one.pixel_count() == 2

    multiple = cluster_pixels(_pixel_payload([(1, 2), (4, 7)]), kt=1, kf=1)
    assert multiple.event_count() == 2
    assert multiple.pixel_count() == 2


def test_time_delay_packet_pattern_parameters():
    from pycwb.modules.coherence_native.time_delay_packet import (
        _compute_packet_energy_params,
    )

    jb, je, m_low, m_high, mean, offsets = _compute_packet_energy_params(
        M=8, T=20, pattern=9, edge=1.0, wavelet_rate=16, f_low=0.0, f_high=7.0, df=1.0
    )

    assert (jb, je, m_low, m_high, mean) == (32, 128, 1, 6, 9.0)
    np.testing.assert_array_equal(offsets, np.array([0, 1, -1, 8, -8, 9, 7, -7, -9]))


@pytest.mark.slow
def test_jax_numba_max_energy_parity_tiny_wdm_map():
    if os.getenv("PYCWB_RUN_MAX_ENERGY_PARITY") != "1":
        pytest.skip(
            "Set PYCWB_RUN_MAX_ENERGY_PARITY=1 to run backend parity smoke test"
        )

    pytest.importorskip("jax")
    pytest.importorskip("numba")
    wdm = pytest.importorskip("wdm_wavelet.wdm")
    if getattr(wdm, "WDM", object) is object:
        pytest.skip("real wdm-wavelet package is unavailable")

    from pycwb.modules.coherence_native.time_delay_jax import time_delay_max_energy
    from pycwb.modules.coherence_native.time_delay_numba import (
        time_delay_max_energy_numba,
    )
    from pycwb.types.time_frequency_map import TimeFrequencyMap
    from pycwb.types.time_series import TimeSeries

    wavelet = wdm.WDM(M=4, K=4, beta_order=6, precision=10)
    ts = TimeSeries(np.sin(np.linspace(0.0, 4.0 * np.pi, 64)), t0=0.0, dt=1.0 / 64.0)
    base = TimeFrequencyMap.from_timeseries(
        ts, wavelet, is_whitened=True, f_low=0.0, f_high=16.0, edge=0.0
    )

    jax_map, jax_alp = time_delay_max_energy(base, 0.01, downsample=1, pattern=1)
    numba_map, numba_alp = time_delay_max_energy_numba(
        base, 0.01, downsample=1, pattern=1
    )

    np.testing.assert_allclose(numba_map.data, jax_map.data, rtol=1.0e-5, atol=1.0e-7)
    assert numba_alp == pytest.approx(jax_alp, rel=1.0e-5, abs=1.0e-7)
