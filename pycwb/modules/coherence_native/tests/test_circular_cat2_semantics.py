from types import SimpleNamespace
import sys
from types import ModuleType

import numpy as np

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

from pycwb.modules.coherence_native.coherence import _build_selection_cache, select_network_pixels
from pycwb.modules.job_segment.dq_segment import build_cat2_veto_windows
from pycwb.modules.job_segment.job_segment import job_segment_from_dq
from pycwb.types.data_quality_file import DQFile
from pycwb.types.job import WaveSegment


def _segment(*, lag_array=None, veto_windows=None, cwb_veto_windows=None):
    return WaveSegment(
        index=1,
        ifos=["H1", "L1"],
        analyze_start=0.0,
        analyze_end=10.0,
        sample_rate=1024,
        seg_edge=0.0,
        lag_array=lag_array or [[0.0, 0.0]],
        veto_windows=veto_windows,
        cwb_veto_windows=cwb_veto_windows,
    )


def _tf_map(data, *, dt=1.0, df=1.0, f_low=0.0, f_high=None):
    data = np.asarray(data, dtype=float)
    return SimpleNamespace(
        data=data,
        dt=dt,
        df=df,
        start=0.0,
        stop=float(data.shape[1]) * dt,
        f_low=f_low,
        f_high=float(data.shape[0] - 1) * df if f_high is None else f_high,
    )


def _reference_select_network_pixels(tf_maps, energy_threshold, lag_shifts=None, veto=None, edge=0.0):
    arrays = [np.asarray(tfm.data, dtype=np.float64) for tfm in tf_maps]
    n_ifo = len(arrays)
    n_freq, n_time = arrays[0].shape
    dt = float(tf_maps[0].dt)
    rate = 1.0 / dt

    if lag_shifts is None:
        shifts_sec = np.zeros(n_ifo, dtype=float)
    else:
        shifts_sec = np.asarray(lag_shifts, dtype=float)
    ref = float(np.min(shifts_sec)) if shifts_sec.size else 0.0
    shift_bins = np.asarray([int((float(s) - ref) * rate + 0.001) for s in shifts_sec], dtype=np.int64)

    edge_bins = int(max(0, float(edge) * rate + 0.001))
    valid_start = edge_bins
    valid_stop = n_time - edge_bins
    nn_valid = valid_stop - valid_start

    f_low = float(getattr(tf_maps[0], "f_low", 0.0) or 0.0)
    f_high = float(getattr(tf_maps[0], "f_high", (n_freq - 1) * tf_maps[0].df))
    df = float(getattr(tf_maps[0], "df", 0.0) or 0.0)
    ib = 1
    ie = n_freq
    if df > 0:
        for idx_f, freq in enumerate(np.arange(n_freq, dtype=np.float64) * df):
            if freq <= f_high:
                ie = idx_f
            if freq <= f_low:
                ib = idx_f + 1
    ie = min(ie, n_freq - 1)
    ib = max(ib, 1)

    eo = float(energy_threshold)
    em = 2.0 * eo
    eh = em * em
    has_veto = veto is not None and len(veto) == n_time
    veto_arr = np.asarray(veto, dtype=np.int16) if has_veto else np.zeros(0, dtype=np.int16)

    live_mask = np.zeros(n_time, dtype=bool)
    combined_raw = np.zeros((n_freq, n_time), dtype=np.float64)
    for t in range(valid_start, valid_start + nn_valid):
        u = t - valid_start
        live = True
        if has_veto:
            for d in range(n_ifo):
                src_t = valid_start + (u + shift_bins[d]) % nn_valid
                if veto_arr[src_t] == 0:
                    live = False
                    break
        live_mask[t] = live

    for fi in range(n_freq):
        for t in range(valid_start, valid_start + nn_valid):
            u = t - valid_start
            combined_raw[fi, t] = sum(
                arrays[d][fi, valid_start + (u + shift_bins[d]) % nn_valid]
                for d in range(n_ifo)
            )

    combined = combined_raw.copy()
    if has_veto:
        combined[:, ~live_mask] = 0.0
    if edge_bins > 0:
        combined[:, :edge_bins] = 0.0
        combined[:, n_time - edge_bins:] = 0.0
    combined[:ib, :] = 0.0
    combined = np.where(combined < eo, 0.0, np.where(combined > em, em + 0.1, combined))

    ii = n_freq - 2
    margin = max(edge_bins, 2)
    selected = np.zeros((n_freq, n_time), dtype=bool)
    for fi in range(ib, min(max(ie, ib), n_freq - 1)):
        for t in range(margin, n_time - margin):
            e_val = combined[fi, t]
            if e_val < eo:
                continue
            ct = combined[fi + 1, t] + combined[fi, t + 1] + combined[fi + 1, t + 1]
            cb = combined[fi - 1, t] + combined[fi, t - 1] + combined[fi - 1, t - 1]
            ht = combined[fi + 1, t + 2]
            if fi < ii:
                ht += combined[fi + 2, t + 2] + combined[fi + 2, t + 1]
            hb = combined[fi - 1, t - 2]
            if fi >= 2:
                hb += combined[fi - 2, t - 2] + combined[fi - 2, t - 1]
            if not ((ct + cb) * e_val < eh
                    and (ct + ht) * e_val < eh
                    and (cb + hb) * e_val < eh
                    and e_val < em):
                selected[fi, t] = True

    freq_idx, time_idx = np.where(selected)
    pix_det_energy = np.empty((len(freq_idx), n_ifo), dtype=np.float64)
    pix_det_index = np.empty((len(freq_idx), n_ifo), dtype=np.int64)
    for idx, (fi, t) in enumerate(zip(freq_idx, time_idx)):
        for d in range(n_ifo):
            if nn_valid > 0 and valid_start <= t < valid_stop:
                u = t - valid_start
                det_t = valid_start + (u + shift_bins[d]) % nn_valid
            else:
                det_t = t
            e = arrays[d][fi, det_t]
            pix_det_energy[idx, d] = e if e > 0.0 else 0.0
            pix_det_index[idx, d] = det_t * n_freq + fi

    return {
        "mask": selected,
        "time": time_idx,
        "frequency": freq_idx,
        "energy": combined_raw[freq_idx, time_idx],
        "pix_det_energy": pix_det_energy,
        "pix_det_index": pix_det_index,
        "live_mask": live_mask,
        "live_samples": int(np.sum(live_mask)),
    }


def _assert_selection_payload_equal(actual, expected):
    np.testing.assert_array_equal(actual["mask"], expected["mask"])
    np.testing.assert_array_equal(actual["time"], expected["time"])
    np.testing.assert_array_equal(actual["frequency"], expected["frequency"])
    np.testing.assert_allclose(actual["energy"], expected["energy"])
    np.testing.assert_allclose(actual["pix_det_energy"], expected["pix_det_energy"])
    np.testing.assert_array_equal(actual["pix_det_index"], expected["pix_det_index"])
    np.testing.assert_array_equal(actual["live_mask"], expected["live_mask"])
    assert actual["live_samples"] == expected["live_samples"]


def test_circular_livetime_zero_lag_matches_linear_overlap():
    seg = _segment(veto_windows=[(2.0, 6.0)])

    assert seg.livetime(0) == 4.0
    assert seg.circular_livetime(0) == 4.0


def test_circular_livetime_wraps_shifted_keep_windows():
    seg = _segment(
        lag_array=[[0.0, 8.0]],
        veto_windows=[(0.0, 5.0)],
        cwb_veto_windows=[(0.0, 5.0)],
    )

    assert seg.livetime(0) == 0.0
    assert seg.circular_livetime(0) == 3.0


def test_circular_livetime_merges_wrapped_multiple_keep_windows():
    seg = _segment(
        lag_array=[[0.0, 2.0]],
        cwb_veto_windows=[(0.0, 2.0), (8.0, 10.0)],
    )

    assert seg.circular_livetime(0) == 2.0


def test_circular_livetime_without_veto_windows_returns_full_duration():
    assert _segment(lag_array=[[0.0, 8.0]]).circular_livetime(0) == 10.0


def test_select_network_pixels_vetoes_shifted_detector_source_bins():
    data = np.ones((5, 8), dtype=float) * 10.0
    tf_maps = [_tf_map(data), _tf_map(data)]

    no_veto = select_network_pixels(
        tf_maps=tf_maps,
        lag_index=0,
        energy_threshold=1.0,
        lag_shifts=[0.0, 2.0],
        veto=None,
        edge=0.0,
    )
    assert 4 in set(no_veto["time"])

    veto = np.zeros(8, dtype=np.int16)
    veto[4] = 1
    rejected = select_network_pixels(
        tf_maps=tf_maps,
        lag_index=0,
        energy_threshold=1.0,
        lag_shifts=[0.0, 2.0],
        veto=veto,
        edge=0.0,
    )
    assert 4 not in set(rejected["time"])
    assert not rejected["live_mask"][4]

    veto[6] = 1
    accepted = select_network_pixels(
        tf_maps=tf_maps,
        lag_index=0,
        energy_threshold=1.0,
        lag_shifts=[0.0, 2.0],
        veto=veto,
        edge=0.0,
    )
    assert 4 in set(accepted["time"])
    assert accepted["live_mask"][4]
    idx = np.where(accepted["time"] == 4)[0][0]
    assert list(accepted["pix_det_index"][idx] // data.shape[0]) == [4, 6]


def test_select_network_pixels_zero_lag_uses_reference_veto_mask():
    data = np.ones((5, 8), dtype=float) * 10.0
    veto = np.zeros(8, dtype=np.int16)
    veto[4] = 1

    candidates = select_network_pixels(
        tf_maps=[_tf_map(data), _tf_map(data)],
        lag_index=0,
        energy_threshold=1.0,
        lag_shifts=[0.0, 0.0],
        veto=veto,
        edge=0.0,
    )

    assert 4 in set(candidates["time"])


def test_select_network_pixels_matches_dense_reference_across_edge_cases():
    base0 = np.zeros((7, 12), dtype=float)
    base1 = np.zeros_like(base0)
    base0[1:6, 2:10] = 4.0
    base1[1:6, 2:10] = 3.0
    base0[3, 4] = 30.0  # hard-cap path: clipped support map, raw output energy
    base1[3, 7] = 25.0  # shifted/circular source bin can contribute at t=4

    cases = [
        {
            "name": "zero_lag",
            "tf_maps": [_tf_map(base0), _tf_map(base1)],
            "lag_shifts": [0.0, 0.0],
            "edge": 0.0,
            "veto": None,
        },
        {
            "name": "shifted_wrap",
            "tf_maps": [_tf_map(base0), _tf_map(base1)],
            "lag_shifts": [0.0, 3.0],
            "edge": 0.0,
            "veto": None,
        },
        {
            "name": "shifted_veto",
            "tf_maps": [_tf_map(base0), _tf_map(base1)],
            "lag_shifts": [0.0, 3.0],
            "edge": 0.0,
            "veto": np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.int16),
        },
        {
            "name": "edge_crop",
            "tf_maps": [_tf_map(base0), _tf_map(base1)],
            "lag_shifts": [0.0, 0.0],
            "edge": 2.0,
            "veto": None,
        },
        {
            "name": "frequency_band",
            "tf_maps": [
                _tf_map(base0, f_low=2.0, f_high=4.0),
                _tf_map(base1, f_low=2.0, f_high=4.0),
            ],
            "lag_shifts": [0.0, 0.0],
            "edge": 0.0,
            "veto": None,
        },
        {
            "name": "empty",
            "tf_maps": [_tf_map(np.zeros_like(base0)), _tf_map(np.zeros_like(base1))],
            "lag_shifts": [0.0, 0.0],
            "edge": 0.0,
            "veto": None,
        },
    ]

    for case in cases:
        actual = select_network_pixels(
            tf_maps=case["tf_maps"],
            lag_index=0,
            energy_threshold=5.0,
            lag_shifts=case["lag_shifts"],
            veto=case["veto"],
            edge=case["edge"],
        )
        expected = _reference_select_network_pixels(
            tf_maps=case["tf_maps"],
            energy_threshold=5.0,
            lag_shifts=case["lag_shifts"],
            veto=case["veto"],
            edge=case["edge"],
        )
        try:
            _assert_selection_payload_equal(actual, expected)
        except AssertionError as exc:
            raise AssertionError(case["name"]) from exc


def test_select_network_pixels_uses_cached_shift_bins_by_lag():
    data = np.ones((5, 8), dtype=float) * 10.0
    tf_maps = [_tf_map(data), _tf_map(data)]
    cache = _build_selection_cache(
        tf_maps,
        edge=0.0,
        lag_shifts_by_lag=[[0.0, 0.0], [0.0, 2.0]],
    )

    actual = select_network_pixels(
        tf_maps=tf_maps,
        lag_index=1,
        energy_threshold=1.0,
        lag_shifts=None,
        edge=0.0,
        selection_cache=cache,
    )
    expected = select_network_pixels(
        tf_maps=tf_maps,
        lag_index=0,
        energy_threshold=1.0,
        lag_shifts=[0.0, 2.0],
        edge=0.0,
    )

    _assert_selection_payload_equal(actual, expected)


def test_build_cat2_veto_windows_applies_per_ifo_superlag_shift(tmp_path):
    h1 = tmp_path / "H1_cat2.txt"
    l1 = tmp_path / "L1_cat2.txt"
    h1.write_text("0 10\n")
    l1.write_text("2 12\n")
    dq_files = [
        DQFile("H1", str(h1), "CWB_CAT2", 0.0, False, False),
        DQFile("L1", str(l1), "CWB_CAT2", 0.0, False, False),
    ]

    assert build_cat2_veto_windows(dq_files, ["H1", "L1"]) == [(2.0, 10.0)]
    assert build_cat2_veto_windows(
        dq_files, ["H1", "L1"], shifts_by_ifo={"L1": -2.0},
    ) == [(0.0, 10.0)]


def test_job_segment_cwb_cat2_uses_pycwb_superlag_sign(tmp_path):
    h1_cat1 = tmp_path / "H1_cat1.txt"
    l1_cat1 = tmp_path / "L1_cat1.txt"
    l1_cat2 = tmp_path / "L1_cat2.txt"
    h1_cat1.write_text("1000 1030\n")
    l1_cat1.write_text("1000 1030\n")
    l1_cat2.write_text("1015 1016\n")
    dq_files = [
        DQFile("H1", str(h1_cat1), "CWB_CAT1", 0.0, False, False),
        DQFile("L1", str(l1_cat1), "CWB_CAT1", 0.0, False, False),
        DQFile("L1", str(l1_cat2), "CWB_CAT2", 0.0, True, False),
    ]

    jobs = job_segment_from_dq(
        dq_files,
        ["H1", "L1"],
        seg_len=10,
        seg_mls=1,
        seg_edge=0,
        seg_overlap=0,
        rateANA=1024,
        l_high=10,
        sample_rate=1024,
        periods=([1000], [1030]),
        slag_size=2,
        slag_min=1,
        slag_max=1,
    )

    shifted_job = next(
        job for job in jobs
        if list(job.shift) == [0, -10] and job.analyze_start == 1000
    )

    assert shifted_job.physical_analyze_starts["L1"] == 1010
    assert shifted_job.cwb_veto_windows == [(1000.0, 1005.0), (1006.0, 1010.0)]
