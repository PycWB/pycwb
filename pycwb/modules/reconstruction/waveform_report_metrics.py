import numpy as np  # type: ignore

from pycwb.types.waveform import Waveform


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
