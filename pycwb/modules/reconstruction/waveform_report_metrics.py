import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np  # type: ignore

from pycwb.types.time_series import TimeSeries
from pycwb.types.waveform import Waveform

# Heuristic warning only: twelve orders below unit noise amplitude in cWB's
# whitening convention. This is not a physical lower bound on signal strength.
_WHITENED_AMPLITUDE_WARNING_THRESHOLD = 1e-12


def compute_confidence_intervals(
    waveforms,
    confidence_level=0.95,
    method="percentiles",
    reference_waveform=None,
):
    """Compute confidence intervals for a collection of waveforms."""
    if method == "percentiles":
        lower_bound, upper_bound = np.nanpercentile(
            waveforms,
            [(1 - confidence_level) * 100 / 2, (1 + confidence_level) * 100 / 2],
            axis=0,
        )

    elif method == "upper":
        lower_bound, upper_bound = np.nanpercentile(
            waveforms, [0, confidence_level * 100], axis=0
        )

    elif method == "lower":
        lower_bound, upper_bound = np.nanpercentile(
            waveforms, [(1 - confidence_level) * 100, 100], axis=0
        )

    elif method == "BCa":
        if reference_waveform is None:
            raise ValueError("Reference waveform must be provided for BCa intervals.")
        lower_bound, upper_bound = BCa_confidence_intervals(
            waveforms, reference_waveform, confidence_level=confidence_level
        )

    elif method == "studentized_bootstrap":
        if reference_waveform is None:
            raise ValueError(
                "Reference waveform must be provided for studentized bootstrap intervals."
            )
        lower_bound, upper_bound = studentized_bootstrap_confidence_intervals(
            waveforms, reference_waveform, confidence_level
        )

    else:
        raise ValueError("Unsupported ordering method.")

    return lower_bound, upper_bound


def studentized_bootstrap_confidence_intervals(
    waveforms,
    reference_waveform,
    confidence_level,
):
    """Compute studentized bootstrap confidence intervals."""
    waveforms = np.asarray(waveforms)
    alpha = (1 - confidence_level) / 2

    se = np.nanstd(waveforms, axis=0, ddof=1)
    studentized_residuals = (waveforms - reference_waveform) / se
    t_up, t_low = np.nanpercentile(
        studentized_residuals,
        [100 * alpha, 100 * (1 - alpha)],
        axis=0,
    )

    lower_bound = reference_waveform - t_low * se
    upper_bound = reference_waveform - t_up * se

    return lower_bound, upper_bound


def BCa_confidence_intervals(waveforms, reference_waveform, confidence_level=0.95):
    """Compute bias-corrected and accelerated confidence intervals."""
    from scipy.stats import norm  # type: ignore

    alpha = (1 - confidence_level) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100

    waveforms = np.array(waveforms)
    _, n_points = waveforms.shape

    lower_bound = np.zeros(n_points)
    upper_bound = np.zeros(n_points)

    for i in range(n_points):
        point_values = waveforms[:, i]
        point_values = point_values[~np.isnan(point_values)]

        if len(point_values) == 0:
            lower_bound[i] = np.nan
            upper_bound[i] = np.nan
            continue

        point_mean = reference_waveform[i]
        z0 = norm.ppf(np.sum(point_values < point_mean) / len(point_values))

        z_lower = norm.ppf(lower_percentile / 100)
        z_upper = norm.ppf(upper_percentile / 100)

        adj_lower = norm.cdf(2 * z0 + z_lower)
        adj_upper = norm.cdf(2 * z0 + z_upper)

        lower_bound[i] = np.percentile(point_values, adj_lower * 100)
        upper_bound[i] = np.percentile(point_values, adj_upper * 100)
    return lower_bound, upper_bound


def _waveform_data(waveform):
    return waveform.data if isinstance(waveform, Waveform) else waveform


def compute_overlap(reconstructed, reference_waveform):
    """Compute waveform overlap against one reference or paired references."""
    if isinstance(reference_waveform, Waveform):
        reconstructed = np.atleast_2d(reconstructed)
        reference_data = np.asarray(reference_waveform.data)
        norm1 = np.linalg.norm(reconstructed, axis=1)
        norm2 = np.linalg.norm(reference_data)
        overlaps = np.dot(reconstructed, reference_data) / (norm1 * norm2)

    elif isinstance(reference_waveform, list) and len(reference_waveform) == len(reconstructed):
        overlaps = []
        for reconstructed_waveform, ref_waveform in zip(reconstructed, reference_waveform):
            overlap = compute_overlap(reconstructed_waveform, ref_waveform)
            overlaps.append(overlap)

    else:
        raise ValueError(
            "Reference waveform list length must be 1 or equal to the number of "
            "reconstructed waveforms."
        )

    overlaps = np.array(overlaps)

    if overlaps.size == 1:
        return overlaps.item()

    return overlaps


def _sample_coordinate(time: float, origin: float, dt: float) -> float:
    """Convert seconds to samples, snapping only within floating-point roundoff.

    Each input float and arithmetic operation contributes at most half an ULP
    (the spacing between adjacent representable floats). Propagate those errors
    through (time - origin) / dt so GPS epochs receive an appropriate tolerance.
    This is numerical boundary handling, not a physical time-alignment tolerance.
    """
    delta = time - origin
    coordinate = delta / dt
    if not math.isfinite(coordinate):
        raise ValueError("time offset cannot be represented in sample coordinates")
    roundoff_seconds = 0.5 * (
        math.ulp(time)
        + math.ulp(origin)
        + math.ulp(delta)
        + abs(coordinate) * math.ulp(dt)
    )
    tolerance = roundoff_seconds / dt + 0.5 * math.ulp(coordinate)
    if tolerance >= 0.5:
        raise ValueError(
            "timestamp precision is insufficient to resolve the sample grid"
        )
    nearest = round(coordinate)
    return float(nearest) if abs(coordinate - nearest) <= tolerance else coordinate


def _validated_fitting_factor_series(value: object, label: str, ifo: str) -> TimeSeries:
    """Convert a supported input and validate samples and timing before alignment."""
    try:
        series = TimeSeries.from_input(value)
    except ValueError as exc:
        raise ValueError(
            f"{label}[{ifo!r}] must be a pycWB, PyCBC-compatible, "
            "or GWPy-compatible time series"
        ) from exc

    if series.data.ndim != 1 or series.data.size == 0:
        raise ValueError(
            f"{label} time series for IFO {ifo!r} must be non-empty and one-dimensional"
        )
    if not np.all(np.isfinite(series.data)):
        raise ValueError(f"non-finite samples found in {label} for IFO {ifo!r}")
    if not (np.isfinite(series.t0) and np.isfinite(series.dt)) or series.dt <= 0.0:
        raise ValueError(f"invalid timing metadata in {label} for IFO {ifo!r}")
    return series


def _select_fitting_factor_samples(
    injected: TimeSeries,
    reconstructed: TimeSeries,
    start: float | None,
    end: float | None,
    ifo: str,
    missing: Literal["raise", "zero"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select full energy windows and their shared samples for the cross term.

    Indices are on the injected grid. Missing samples contribute zero only
    when explicitly requested; no padded arrays are allocated, even for large
    gaps. Inputs have already been checked for finite timing and equal spacing.
    Returns the injected and reconstructed energy windows, followed by their
    paired overlap samples for the numerator.
    """
    coordinate = _sample_coordinate(reconstructed.t0, injected.t0, injected.dt)
    if not coordinate.is_integer():
        raise ValueError(f"time grids are not sample-aligned for IFO {ifo!r}")
    offset = int(coordinate)
    inj_stop = len(injected.data)
    rec_stop = offset + len(reconstructed.data)

    first, stop = min(0, offset), max(inj_stop, rec_stop)
    if start is not None:
        lower = _sample_coordinate(start, injected.t0, injected.dt)
        first = max(first, math.floor(lower) + 1)
    if end is not None:
        upper = _sample_coordinate(end, injected.t0, injected.dt)
        stop = min(stop, math.floor(upper) + 1)
    if first >= stop:
        raise ValueError(f"no samples selected for IFO {ifo!r}")

    if missing == "raise" and (
        first < max(0, offset) or stop > min(inj_stop, rec_stop)
    ):
        raise ValueError(
            f"incomplete waveform coverage for IFO {ifo!r}; provide a fully "
            "covered evaluation window, or use missing='zero' only when "
            "the waveforms are known to be zero outside their stored support"
        )

    def slice_on_grid(
        data: np.ndarray, origin: int, lower: int, upper: int
    ) -> np.ndarray:
        # Clamp both endpoints to avoid negative Python indices for disjoint data.
        begin = max(0, min(len(data), lower - origin))
        finish = max(begin, min(len(data), upper - origin))
        return data[begin:finish]

    inj_window = slice_on_grid(injected.data, 0, first, stop)
    rec_window = slice_on_grid(reconstructed.data, offset, first, stop)
    overlap_first = max(first, 0, offset)
    overlap_stop = max(overlap_first, min(stop, inj_stop, rec_stop))
    return (
        inj_window,
        rec_window,
        slice_on_grid(injected.data, 0, overlap_first, overlap_stop),
        slice_on_grid(reconstructed.data, offset, overlap_first, overlap_stop),
    )


def compute_fitting_factor(
    ifos: Sequence[str],
    injected: Mapping[str, object],
    reconstructed: Mapping[str, object],
    start: float | None = None,
    end: float | None = None,
    *,
    missing: Literal["raise", "zero"] = "raise",
) -> float:
    r"""Compute the network fitting factor for already-whitened waveforms.

    Both inputs must be whitened using the same detector-specific noise PSD
    and whitening normalization for each IFO, with consistent normalization
    across the network. This function does not whiten raw strain. The plain
    dot products below represent a noise-weighted match only for such whitened
    inputs. In pycWB, use ``whitened_injected_waveform`` and reconstruct with
    ``whiten=True``.

    The normalization is the same as ``CWB::mdc::GetMatchFactor("ff", ...)``;
    the window and missing-coverage policies below are explicit extensions:

    .. math::

        \mathrm{FF} = \frac{\sum_{I,t} r_{I,t}s_{I,t}}
                          {\sqrt{\left(\sum_{I,t}r_{I,t}^2\right)
                                 \left(\sum_{I,t}s_{I,t}^2\right)}}.

    Each input series is converted with :meth:`TimeSeries.from_input` before
    the calculation.  Therefore, inputs must be a pycWB, PyCBC-compatible, or
    GWPy-compatible time series.  The series may have different start times or
    lengths, extending cWB's equal-start/equal-length interface. All series
    must have identical sampling intervals. Paired samples must lie on the same
    grid; no interpolation or time/phase maximization is performed.

    Bounds and epoch offsets within propagated floating-point roundoff of an
    integer sample are treated as that sample. Genuine fractional-sample offsets
    are rejected, as are timestamps too imprecise to resolve the sample grid.

    Parameters
    ----------
    ifos:
        IFO names and the order in which they are checked.  The result is a
        network FF, not an average of per-IFO FFs.
    injected:
        Mapping from IFO name to the whitened injected (reference) time series.
    reconstructed:
        Mapping from IFO name to the whitened reconstructed time series.
    start, end:
        Optional common time bounds in seconds, in the same frame as each
        series' ``t0``. Samples follow cWB's convention ``start < t <= end``.
        Omitted bounds include all stored samples in each IFO's union of
        injection and reconstruction support, including the first sample.
        Explicit bounds restrict this union; they do not discard unmatched
        samples within the selected window.
    missing:
        ``"raise"`` (default) requires both waveforms to cover the selected
        window for each IFO. ``"zero"`` treats samples outside stored support
        as zero, retaining unmatched energy in the normalization. Use it only
        for known-zero support, never to hide unknown or truncated data.

    Returns
    -------
    float
        Signed network fitting factor.

    Raises
    ------
    ValueError
        If inputs are incomplete, sampling grids are incompatible, the
        selected interval contains no samples, coverage is incomplete under
        ``missing="raise"``, or either network norm is zero.

    Warns
    -----
    UserWarning
        If a selected nonzero waveform has peak absolute amplitude below
        ``1e-12``. This heuristic flags possible raw-strain inputs, twelve
        orders below unit noise amplitude in cWB's whitening convention.
        Amplitude alone cannot verify whitening; exceptionally weak whitened
        signals can also trigger it. No amplitude correction is applied.

    Notes
    -----
    Whitened signals need not have unit RMS or equal amplitudes. FF is invariant
    under independent positive overall scaling of either network waveform and
    therefore cannot measure absolute amplitude recovery. Relative detector
    amplitudes still contribute to the network match. Internally, each network
    is divided by its own maximum absolute selected amplitude before taking
    dot products, avoiding overflow/underflow from the overall amplitude scale.
    No individual detector is normalized separately and inputs are not modified.
    """
    if missing not in ("raise", "zero"):
        raise ValueError("missing must be 'raise' or 'zero'")
    if not ifos:
        raise ValueError("ifos must contain at least one IFO")
    if any(bound is not None and not np.isfinite(bound) for bound in (start, end)):
        raise ValueError("start and end must be finite when provided")
    if start is not None and end is not None and start >= end:
        raise ValueError("start must be smaller than end")

    if len(set(ifos)) != len(ifos):
        raise ValueError("ifos must not contain duplicate detector names")
    selected_windows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    injected_peak = 0.0
    reconstructed_peak = 0.0

    reference_dt = None
    for ifo in ifos:
        if ifo not in injected:
            raise ValueError(f"missing injected time series for IFO {ifo!r}")
        if ifo not in reconstructed:
            raise ValueError(f"missing reconstructed time series for IFO {ifo!r}")
        inj = _validated_fitting_factor_series(injected[ifo], "injected", ifo)
        rec = _validated_fitting_factor_series(reconstructed[ifo], "reconstructed", ifo)

        if inj.dt != rec.dt:
            raise ValueError(f"sampling intervals differ for IFO {ifo!r}")
        if reference_dt is None:
            reference_dt = inj.dt
        elif inj.dt != reference_dt:
            raise ValueError("sampling intervals must be identical across IFOs")

        windows = _select_fitting_factor_samples(inj, rec, start, end, ifo, missing)
        inj_slice, rec_slice, _, _ = windows
        selected_windows.append(windows)
        inj_peak = float(np.max(np.abs(inj_slice), initial=0.0))
        rec_peak = float(np.max(np.abs(rec_slice), initial=0.0))
        injected_peak = max(injected_peak, inj_peak)
        reconstructed_peak = max(reconstructed_peak, rec_peak)

        for label, peak in (("injected", inj_peak), ("reconstructed", rec_peak)):
            if 0.0 < peak < _WHITENED_AMPLITUDE_WARNING_THRESHOLD:
                warnings.warn(
                    f"{label} waveform for IFO {ifo!r} has selected peak amplitude "
                    f"{peak:.3g}, below {_WHITENED_AMPLITUDE_WARNING_THRESHOLD:g}; "
                    "verify that this is whitened data, not raw strain. "
                    "Amplitude alone cannot establish whitening.",
                    UserWarning,
                    stacklevel=2,
                )

    if injected_peak == 0.0 or reconstructed_peak == 0.0:
        raise ValueError("selected waveforms must both have non-zero network energy")

    # A common scale for every IFO preserves network weighting. Multiplying
    # unscaled energies would overflow even when their individual norms fit.
    injected_energies, reconstructed_energies, cross_energies = [], [], []
    for inj_window, rec_window, inj_overlap, rec_overlap in selected_windows:
        scaled_inj = inj_window / injected_peak
        scaled_rec = rec_window / reconstructed_peak
        injected_energies.append(float(np.dot(scaled_inj, scaled_inj)))
        reconstructed_energies.append(float(np.dot(scaled_rec, scaled_rec)))
        cross_energies.append(
            float(np.dot(inj_overlap / injected_peak, rec_overlap / reconstructed_peak))
        )

    return (
        math.fsum(cross_energies)
        / math.sqrt(math.fsum(injected_energies))
        / math.sqrt(math.fsum(reconstructed_energies))
    )


def compute_cumulative_hrss(waveform, delta_t, axis=1):
    """Compute cumulative hrss of a waveform."""
    return np.sqrt(np.cumsum(np.abs(waveform) ** 2, axis=axis) * delta_t)


def compute_leakage(reconstructed, reference_waveform, time):
    """Compute time leakage after the injected signal's end time."""
    reference_data = np.asarray(_waveform_data(reference_waveform))
    injected_hrss = np.sqrt(np.nansum(np.square(reference_data)))
    leaked_hrss = np.zeros(shape=(len(reconstructed), len(time)))

    if injected_hrss == 0 or reference_data.size == 0:
        return np.nanmean(leaked_hrss, axis=0), np.nanstd(leaked_hrss, axis=0)

    threshold = np.nanmax(np.abs(reference_data)) * 1e-3
    active = np.where(np.abs(reference_data) > threshold)[0]
    if active.size == 0:
        return np.nanmean(leaked_hrss, axis=0), np.nanstd(leaked_hrss, axis=0)

    end_time_idx = int(np.max(active))
    sample_times = reference_waveform.sample_times.data
    end_time = sample_times[min(end_time_idx + 1, len(sample_times) - 1)]
    dt = time[1] - time[0] if len(time) > 1 else 1

    for i, waveform in enumerate(reconstructed):
        for j, _ in enumerate(time):
            try:
                leaked_hrss[i, j] = (
                    np.sqrt(
                        np.nansum(
                            waveform.time_slice(
                                end_time + j * dt,
                                end_time + (j + 1) * dt,
                            ).data
                            ** 2
                        )
                    )
                    / injected_hrss
                )
            except (IndexError, ValueError):
                pass

    mean_leakage = np.nanmean(leaked_hrss, axis=0)
    std_leakage = np.nanstd(leaked_hrss, axis=0) / np.sqrt(len(reconstructed))
    return mean_leakage, std_leakage


def compute_hrss(waveform, delta_t):
    """Compute hrss for a waveform."""
    return np.sqrt(np.sum(np.abs(waveform) ** 2) * delta_t)
