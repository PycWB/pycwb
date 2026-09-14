import warnings

import numpy as np
import pytest

from pycwb.modules.reconstruction.waveform_report_metrics import (
    compute_cumulative_hrss,
    compute_fitting_factor,
    compute_hrss,
    compute_leakage,
    compute_overlap,
)
from pycwb.types.time_series import TimeSeries
from pycwb.types.waveform import Waveform


def _waveform(data):
    return Waveform(TimeSeries(np.asarray(data, dtype=float), t0=0.0, dt=0.1))


def test_compute_overlap_identical_waveform_is_one():
    reference = _waveform([0.0, 1.0, 0.0, -1.0, 0.0])

    overlap = compute_overlap(np.asarray([reference.data]), reference)

    assert np.isclose(overlap, 1.0)


def test_compute_fitting_factor_is_network_normalized_correlation():
    injected = {
        "H1": TimeSeries(np.array([1.0, 2.0, 3.0, 4.0]), t0=0.0, dt=0.1),
        "L1": TimeSeries(np.array([2.0, 1.0, 0.0, -1.0]), t0=0.0, dt=0.1),
    }
    reconstructed = {
        "H1": TimeSeries(2.0 * injected["H1"].data, t0=0.0, dt=0.1),
        "L1": TimeSeries(0.5 * injected["L1"].data, t0=0.0, dt=0.1),
    }

    # The amplitudes differ by IFO, but the network waveforms are parallel
    # only when all IFOs share the same scale.  This expected value is the
    # direct network dot-product result over samples at t > 0 and t <= 0.3.
    inj = np.concatenate([injected["H1"].data[1:], injected["L1"].data[1:]])
    rec = np.concatenate([reconstructed["H1"].data[1:], reconstructed["L1"].data[1:]])
    expected = np.dot(inj, rec) / np.sqrt(np.dot(inj, inj) * np.dot(rec, rec))

    assert np.isclose(
        compute_fitting_factor(["H1", "L1"], injected, reconstructed, 0.0, 0.3),
        expected,
    )


def test_compute_fitting_factor_handles_sample_aligned_time_offset():
    injected = {"H1": TimeSeries(np.array([0.0, 1.0, 2.0, 3.0]), t0=0.0, dt=0.1)}
    reconstructed = {"H1": TimeSeries(np.array([1.0, 2.0, 3.0]), t0=0.1, dt=0.1)}

    assert np.isclose(
        compute_fitting_factor(["H1"], injected, reconstructed, 0.0, 0.3),
        1.0,
    )


class _PyCBCCompatibleTimeSeries:
    def __init__(self, data, start_time, delta_t):
        self.data = np.asarray(data, dtype=float)
        self.start_time = start_time
        self.delta_t = delta_t


def test_compute_fitting_factor_converts_recognized_time_series_inputs():
    injected = {"H1": _PyCBCCompatibleTimeSeries([1.0, 2.0, 3.0], 0.0, 0.1)}
    reconstructed = {"H1": _PyCBCCompatibleTimeSeries([1.0, 2.0, 3.0], 0.0, 0.1)}

    assert np.isclose(
        compute_fitting_factor(["H1"], injected, reconstructed, 0.0, 0.2),
        1.0,
    )


def test_compute_fitting_factor_rejects_unrecognized_time_series_inputs():
    injected = {"H1": object()}
    reconstructed = {"H1": TimeSeries(np.array([1.0, 2.0]), t0=0.0, dt=0.1)}

    with pytest.raises(ValueError, match="injected\\['H1'\\] must be"):
        compute_fitting_factor(["H1"], injected, reconstructed, 0.0, 0.1)


def test_compute_cumulative_hrss_is_monotonic():
    cumulative = compute_cumulative_hrss(
        np.asarray([[0.0, 1.0, 2.0, 2.0]]),
        delta_t=0.25,
        axis=1,
    )

    assert np.all(np.diff(cumulative[0]) >= 0)
    assert np.isclose(cumulative[0, -1], compute_hrss([0.0, 1.0, 2.0, 2.0], 0.25))


def test_compute_leakage_runs_on_synthetic_waveform():
    reference = _waveform([0.0, 1.0, 0.0, -1.0, 0.0])
    reconstructed = [_waveform([0.0, 1.0, 0.0, -1.0, 0.0])]
    time = np.linspace(0.0, 0.2, 4)

    mean, std = compute_leakage(reconstructed, reference, time)

    assert mean.shape == time.shape
    assert std.shape == time.shape


def _single_ifo_ff(
    injected,
    reconstructed,
    *,
    t0=0.0,
    dt=0.1,
    offset=0,
    start=None,
    end=None,
    missing="raise",
):
    return compute_fitting_factor(
        ["H1"],
        {"H1": TimeSeries(np.asarray(injected), t0=t0, dt=dt)},
        {"H1": TimeSeries(np.asarray(reconstructed), t0=t0 + offset * dt, dt=dt)},
        start,
        end,
        missing=missing,
    )


def test_fitting_factor_includes_first_sample_when_start_precedes_data():
    # All three selected samples matter: discarding the first changes the sign.
    actual = _single_ifo_ff([10, 1, 1, 1], [-10, 1, 1, -1], dt=1.0, start=-1.0, end=2.0)
    assert actual == pytest.approx(-98.0 / 102.0, abs=1e-14)


@pytest.mark.parametrize("epoch", [0.0, 1_400_000_000.0])
def test_fitting_factor_includes_upper_boundary_at_gps_epoch(epoch):
    actual = _single_ifo_ff(
        [10, 1, 1, 1], [-10, 1, 1, -1], t0=epoch, start=epoch, end=epoch + 0.3
    )
    assert actual == pytest.approx(1.0 / 3.0, abs=1e-14)


@pytest.mark.parametrize("epoch", [0.0, 1_400_000_000.0])
@pytest.mark.parametrize("dt", [0.1, 1.0 / 4096])
@pytest.mark.parametrize("offset", [-2, -1, 1, 2])
def test_fitting_factor_aligns_integer_offsets_and_retains_unmatched_energy(
    epoch, dt, offset
):
    inj = np.array([3.0, -2.0, 7.0, 1.0, 4.0, -6.0])
    rec = np.array([1.0, 4.0, -3.0, 5.0, -2.0, 9.0])
    first, stop = max(0, offset), min(len(inj), offset + len(rec))
    a, b = inj[first:stop], rec[first - offset : stop - offset]
    expected = np.dot(a, b) / np.sqrt(np.dot(inj, inj) * np.dot(rec, rec))
    actual = _single_ifo_ff(
        inj,
        rec,
        t0=epoch,
        dt=dt,
        offset=offset,
        start=epoch - 10 * dt,
        end=epoch + 10 * dt,
        missing="zero",
    )
    assert actual == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("epoch", [0.0, 1_400_000_000.0])
@pytest.mark.parametrize("offset", [-0.25, 0.25, 1.25])
def test_fitting_factor_rejects_fractional_sample_offsets(epoch, offset):
    with pytest.raises(ValueError, match="sample-aligned"):
        _single_ifo_ff([1, 2, 3], [3, 1, 2], t0=epoch, offset=offset)


@pytest.mark.parametrize("epoch", [0.0, 1_400_000_000.0])
def test_fitting_factor_keeps_bounds_between_samples(epoch):
    # Only samples 1 and 2 fall in (0.05, 0.25].
    actual = _single_ifo_ff(
        [10, 1, 2, 7], [10, 2, -1, 9], t0=epoch, start=epoch + 0.05, end=epoch + 0.25
    )
    assert actual == pytest.approx(0.0, abs=1e-14)


def test_fitting_factor_default_window_includes_first_sample():
    actual = _single_ifo_ff([10, 1, 1], [-10, 1, -1])
    assert actual == pytest.approx(-100.0 / 102.0, abs=1e-14)


def test_fitting_factor_can_select_single_sample_with_explicit_bounds():
    assert _single_ifo_ff([2], [-3], start=-1.0, end=0.0) == pytest.approx(-1.0)


@pytest.mark.parametrize("start,end", [(1.0, 2.0), (-2.0, -1.0), (0.01, 0.09)])
def test_fitting_factor_rejects_intervals_without_samples(start, end):
    with pytest.raises(ValueError, match="no samples selected"):
        _single_ifo_ff([1, 2, 3], [3, 2, 1], start=start, end=end)


@pytest.mark.parametrize("epoch", [0.0, 1_400_000_000.0, -1_400_000_000.0])
@pytest.mark.parametrize("dt", [0.1, 1.0 / 4096])
@pytest.mark.parametrize("bounds", [(-1, 5), (0, 3), (0.5, 4.5)])
def test_network_fitting_factor_matches_integer_grid_reference(epoch, dt, bounds):
    # The reference selects integer lattice points directly, independently of
    # timestamp subtraction, rounding tolerances, or slice-index calculations.
    rng = np.random.default_rng(872)
    injected, reconstructed = {}, {}
    selected_inj, selected_rec = [], []
    for ifo, offset in [("H1", -1), ("L1", 2)]:
        a, b = rng.normal(size=7), rng.normal(size=6)
        injected[ifo] = TimeSeries(a, t0=epoch, dt=dt)
        reconstructed[ifo] = TimeSeries(b, t0=epoch + offset * dt, dt=dt)
        for i in range(min(0, offset), max(len(a), offset + len(b))):
            if bounds[0] < i <= bounds[1]:
                selected_inj.append(a[i] if 0 <= i < len(a) else 0.0)
                selected_rec.append(b[i - offset] if 0 <= i - offset < len(b) else 0.0)
    a, b = np.array(selected_inj), np.array(selected_rec)
    expected = np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))
    start, end = (epoch + bound * dt for bound in bounds)
    actual = compute_fitting_factor(
        ["H1", "L1"], injected, reconstructed, start, end, missing="zero"
    )
    reverse = compute_fitting_factor(
        ["H1", "L1"], reconstructed, injected, start, end, missing="zero"
    )
    assert actual == pytest.approx(expected, rel=0, abs=2e-14)
    assert reverse == pytest.approx(expected, rel=0, abs=2e-14)


@pytest.mark.parametrize("scale", [-2.0, 0.5, 3.0])
def test_fitting_factor_preserves_sign_and_common_amplitude_scaling(scale):
    injected = np.array([3.0, -1.0, 4.0, 2.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = _single_ifo_ff(injected, scale * injected, start=-0.1, end=0.3)
    assert actual == pytest.approx(np.sign(scale), rel=0, abs=1e-14)


@pytest.mark.parametrize(
    "inj_scale,rec_scale", [(1e-21, 1.0), (1.0, 1e-21), (1e-21, 1e-21)]
)
def test_fitting_factor_warns_about_raw_strain_scale_without_changing_match(
    inj_scale, rec_scale
):
    # The warning must detect either side, even when FF hides mixed units.
    injected = inj_scale * np.array([1.0, 2.0])
    reconstructed = rec_scale * np.array([2.0, -1.0])
    original_inj, original_rec = injected.copy(), reconstructed.copy()
    with pytest.warns(UserWarning, match="verify that this is whitened data") as caught:
        actual = _single_ifo_ff(injected, reconstructed, start=-1.0)
    assert len(caught) == int(inj_scale < 1e-12) + int(rec_scale < 1e-12)
    assert actual == pytest.approx(0.0, rel=0, abs=1e-14)
    np.testing.assert_array_equal(injected, original_inj)
    np.testing.assert_array_equal(reconstructed, original_rec)


def test_fitting_factor_amplitude_check_uses_selected_samples():
    with pytest.warns(UserWarning, match="injected waveform"):
        actual = _single_ifo_ff([1.0, 1e-21, 1e-21], [1.0, 1.0, 1.0], start=0.0)
    assert actual == pytest.approx(1.0, rel=0, abs=1e-14)


@pytest.mark.parametrize("amplitude", [1e-12, 1e-6, 1.0, 1e6])
def test_fitting_factor_allows_nonunit_whitened_amplitudes(amplitude):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = _single_ifo_ff([amplitude, -amplitude], [1.0, -1.0], start=-1.0)
    assert actual == pytest.approx(1.0, rel=0, abs=1e-14)


def test_fitting_factor_zero_amplitude_raises_without_whitening_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="non-zero network energy"):
            _single_ifo_ff([0.0, 0.0], [1.0, 1.0], start=-1.0)


@pytest.mark.parametrize("start,end", [(np.nan, None), (None, np.inf), (-np.inf, 1.0)])
def test_fitting_factor_rejects_nonfinite_bounds(start, end):
    with pytest.raises(ValueError, match="must be finite"):
        _single_ifo_ff([1, 2], [2, 1], start=start, end=end)


def test_fitting_factor_rejects_timestamps_that_cannot_resolve_samples():
    with pytest.raises(ValueError, match="precision is insufficient"):
        _single_ifo_ff([1, 2, 3], [3, 2, 1], t0=1e16)


def test_fitting_factor_requires_identical_sample_spacing():
    with pytest.raises(ValueError, match="sampling intervals differ"):
        compute_fitting_factor(
            ["H1"],
            {"H1": TimeSeries([1, 2, 3], 0.0, 0.1)},
            {"H1": TimeSeries([1, 2, 3], 0.0, 0.1 + 1e-13)},
        )


@pytest.mark.parametrize("offset", [-2, 0, 2])
def test_fitting_factor_rejects_unknown_missing_coverage(offset):
    with pytest.raises(ValueError, match="incomplete waveform coverage"):
        _single_ifo_ff([1, 1], [1, 1, 1, 1], offset=offset)


def test_fitting_factor_zero_support_penalizes_unmatched_energy_symmetrically():
    forward = _single_ifo_ff([1, 1], [1, 1, 1, 1], missing="zero")
    reverse = _single_ifo_ff([1, 1, 1, 1], [1, 1], missing="zero")
    assert forward == pytest.approx(1 / np.sqrt(2), rel=0, abs=1e-14)
    assert reverse == pytest.approx(forward, rel=0, abs=1e-14)
    # An explicitly restricted window measures only fidelity within that window.
    assert _single_ifo_ff([1, 1], [1, 1, 1, 1], end=0.1) == pytest.approx(1.0)


@pytest.mark.parametrize("offset", [-1000000000, 1000000000])
def test_fitting_factor_disjoint_known_zero_support_has_zero_match(offset):
    # Also ensures we do not allocate an array spanning the gap.
    assert _single_ifo_ff([1, 2], [2, 1], offset=offset, missing="zero") == 0.0


def test_fitting_factor_window_selecting_only_one_waveform_has_zero_norm():
    with pytest.raises(ValueError, match="non-zero network energy"):
        _single_ifo_ff([1, 1], [1, 1], offset=5, end=0.1, missing="zero")


def test_fitting_factor_rejects_invalid_missing_policy():
    with pytest.raises(ValueError, match="missing must be"):
        _single_ifo_ff([1], [1], missing="overlap")


@pytest.mark.parametrize(
    "inj_scale,rec_scale",
    [
        (1e300, 1e300),
        (1e-300, 1e-300),
        (1e300, 1e-300),
        (1e-300, 1e300),
        (1e-320, 1e-320),
    ],
)
def test_fitting_factor_extreme_amplitudes_match_analytic_reference(
    inj_scale, rec_scale
):
    # Integer coefficients retain their ratios even for subnormal input floats.
    # Dot = -1, squared norms = 5 and 2, independently of overall scales.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*verify that this is whitened data",
            category=UserWarning,
        )
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            actual = _single_ifo_ff(
                inj_scale * np.array([1.0, 2.0]),
                rec_scale * np.array([1.0, -1.0]),
            )
    assert actual == pytest.approx(-1 / np.sqrt(10), rel=0, abs=1e-14)


def test_fitting_factor_network_scaling_preserves_detector_weights_at_extreme_amplitude():
    injected = {
        "H1": TimeSeries([1e300, 2e300], 0.0, 0.1),
        "L1": TimeSeries([3e300, 4e300], 0.0, 0.1),
    }
    reconstructed = {
        "H1": TimeSeries([4e-300, 3e-300], 0.0, 0.1),
        "L1": TimeSeries([2e-300, 1e-300], 0.0, 0.1),
    }
    with pytest.warns(UserWarning):
        actual = compute_fitting_factor(["H1", "L1"], injected, reconstructed)
    assert actual == pytest.approx(20 / 30, rel=0, abs=1e-14)


def test_fitting_factor_rejects_duplicate_detector_names():
    waveforms = {"H1": TimeSeries([1.0, 2.0], 0.0, 0.1)}
    with pytest.raises(ValueError, match="duplicate detector"):
        compute_fitting_factor(["H1", "H1"], waveforms, waveforms)


@pytest.mark.parametrize("n", [255, 256])
def test_whitened_fitting_factor_matches_frequency_domain_noise_weighting(n):
    # Parseval gives an independent frequency-domain reference. Interior rFFT
    # bins represent +/- frequencies; DC and (for even n) Nyquist count once.
    rng = np.random.default_rng(872)
    dt = 1 / 1024
    frequency = np.fft.rfftfreq(n, dt)
    weight = np.full(len(frequency), 2.0)
    weight[0] = 1.0
    if n % 2 == 0:
        weight[-1] = 1.0
    injected, reconstructed = {}, {}
    cross, inj_energy, rec_energy = 0.0, 0.0, 0.0
    raw_a, raw_b = [], []
    for ifo, noise_scale in [("H1", 1.0), ("L1", 3.0)]:
        a, b = rng.normal(size=n), rng.normal(size=n)
        b += a
        psd = noise_scale * (1 + (frequency / 40) ** 4)
        af, bf = np.fft.rfft(a), np.fft.rfft(b)
        cross += np.sum(weight * (af.conj() * bf).real / psd)
        inj_energy += np.sum(weight * np.abs(af) ** 2 / psd)
        rec_energy += np.sum(weight * np.abs(bf) ** 2 / psd)
        injected[ifo] = TimeSeries(np.fft.irfft(af / np.sqrt(psd), n=n), 1.4e9, dt)
        reconstructed[ifo] = TimeSeries(np.fft.irfft(bf / np.sqrt(psd), n=n), 1.4e9, dt)
        raw_a.extend(a)
        raw_b.extend(b)
    expected = cross / np.sqrt(inj_energy * rec_energy)
    actual = compute_fitting_factor(["H1", "L1"], injected, reconstructed)
    assert actual == pytest.approx(expected, rel=0, abs=2e-14)
    raw_match = np.dot(raw_a, raw_b) / (np.linalg.norm(raw_a) * np.linalg.norm(raw_b))
    assert abs(raw_match - expected) > 0.01


def test_fitting_factor_allows_zero_energy_in_one_detector():
    injected = {
        "H1": TimeSeries([0.0, 0.0], 0.0, 0.1),
        "L1": TimeSeries([1.0, 2.0], 0.0, 0.1),
    }
    reconstructed = {
        "H1": TimeSeries([1.0, 1.0], 0.0, 0.1),
        "L1": TimeSeries([1.0, 2.0], 0.0, 0.1),
    }
    actual = compute_fitting_factor(["H1", "L1"], injected, reconstructed)
    assert actual == pytest.approx(5 / np.sqrt(35), rel=0, abs=1e-14)
