import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]
import numpy as np  # pyright: ignore[reportMissingImports]

from pycwb.modules.reconstruction.waveform_report_metrics import (
    compute_confidence_intervals,
    compute_cumulative_hrss,
    compute_hrss,
    compute_leakage,
    compute_overlap,
)


DEFAULT_PLOT_KWARGS = {
    "ref_color": "darkgray",
    "ref_ls": "-",
    "inj_color": "k",
    "inj_ls": "-",
    "inj_alpha": 0.3,
    "plot_mean": True,
    "plot_median": True,
    "mean_linestyle": "-.",
    "CL_color": "lightgray",
    "CL_alpha": 0.5,
    "injected_alpha": 0.3,
    "median_color": "blue",
    "median_linestyle": "--",
    "mean_color": "red",
    "figsize": (10, 6),
    "fontsize": 12,
    "percentile_method": "percentiles",
}


def _style(kwargs):
    return {**DEFAULT_PLOT_KWARGS, **kwargs}


def _waveform_data(waveform):
    return waveform.data if hasattr(waveform, "data") else waveform


def _waveform_array(waveforms):
    if isinstance(waveforms, (list, tuple)):
        return np.asarray([_waveform_data(waveform) for waveform in waveforms])
    return np.asarray(_waveform_data(waveforms))


def plot_time_waveform_reconstruction(
    reconstructed,
    reference=None,
    injected=None,
    confidence_level=0.95,
    **kwargs,
):
    """Plot time-domain waveform reconstruction with confidence intervals."""
    style = _style(kwargs)
    reconstructed_data = _waveform_array(reconstructed)
    reference_data = _waveform_data(reference) if reference is not None else None
    injected_data = _waveform_data(injected) if injected is not None else None

    to_save = {"CL": confidence_level}
    if reference_data is not None:
        to_save["reference"] = reference_data
    if injected_data is not None:
        to_save["injected"] = injected_data

    lower_bound, upper_bound = compute_confidence_intervals(
        reconstructed_data,
        confidence_level,
        method=style["percentile_method"],
        reference_waveform=reference_data,
    )
    time = reconstructed[0].sample_times.data
    to_save.update(
        {
            "time": time,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "CL": confidence_level,
        }
    )

    fig, ax = plt.subplots(figsize=style["figsize"])
    ax.fill_between(
        time,
        lower_bound,
        upper_bound,
        color=style["CL_color"],
        alpha=style["CL_alpha"],
        label=f"{confidence_level * 100}% CI",
    )

    if reference_data is not None:
        ax.plot(
            time,
            reference_data,
            label="On Source",
            color=style["ref_color"],
            linestyle=style["ref_ls"],
        )

    if injected_data is not None:
        ax.plot(
            time,
            injected_data,
            label="Injected",
            color=style["inj_color"],
            linestyle=style["inj_ls"],
            alpha=style["inj_alpha"],
        )

    if style["plot_mean"]:
        mean = np.nanmean(reconstructed_data, axis=0)
        ax.plot(
            time,
            mean,
            label="Mean",
            color=style["mean_color"],
            linestyle=style["mean_linestyle"],
        )
        to_save.update({"mean": mean})

    if style["plot_median"]:
        median = np.nanmedian(reconstructed_data, axis=0)
        ax.plot(
            time,
            median,
            label="Median",
            color=style["median_color"],
            linestyle=style["median_linestyle"],
        )
        to_save.update({"median": median})

    ax.set_xlabel("GPS Time [s]", fontsize=style["fontsize"])
    ax.set_ylabel("Strain", fontsize=style["fontsize"])
    ax.legend(fontsize=style["fontsize"])
    ax.tick_params(labelsize=style["fontsize"])
    ax.grid(True)
    plt.close()
    return fig, to_save


def plot_frequency_waveform_reconstruction(
    reconstructed,
    reference=None,
    injected=None,
    confidence_level=0.95,
    **kwargs,
):
    """Plot frequency-domain waveform reconstruction with confidence intervals."""
    style = _style(kwargs)

    delta_t = getattr(reconstructed[0], "_delta_t", None)
    reconstructed_data = np.abs(_waveform_array(reconstructed))
    reference_data = (
        np.abs(_waveform_data(reference)) if reference is not None else None
    )
    injected_data = np.abs(_waveform_data(injected)) if injected is not None else None

    to_save = {"CL": confidence_level}
    if reference_data is not None:
        to_save["reference"] = reference_data
    if injected_data is not None:
        to_save["injected"] = injected_data

    lower_bound, upper_bound = compute_confidence_intervals(
        reconstructed_data,
        confidence_level,
        method=style["percentile_method"],
        reference_waveform=reference_data,
    )
    frequency = np.fft.fftfreq(len(reconstructed_data[0]), d=delta_t)

    fig, ax = plt.subplots(figsize=style["figsize"])
    ax.fill_between(
        frequency,
        lower_bound,
        upper_bound,
        color=style["CL_color"],
        alpha=style["CL_alpha"],
        label=f"{confidence_level * 100}% CI",
    )

    if reference_data is not None:
        ax.plot(
            frequency,
            reference_data,
            label="On Source",
            color=style["ref_color"],
            linestyle=style["ref_ls"],
        )

    if injected_data is not None:
        ax.plot(
            frequency,
            injected_data,
            label="Injected",
            color=style["inj_color"],
            linestyle=style["inj_ls"],
            alpha=style["inj_alpha"],
        )

    if style["plot_mean"]:
        mean = np.nanmean(reconstructed_data, axis=0)
        ax.plot(
            frequency,
            mean,
            label="Mean",
            color=style["mean_color"],
            linestyle=style["mean_linestyle"],
        )
        to_save.update({"mean": mean})

    if style["plot_median"]:
        median = np.nanmedian(reconstructed_data, axis=0)
        ax.plot(
            frequency,
            median,
            label="Median",
            color=style["median_color"],
            linestyle=style["median_linestyle"],
        )
        to_save.update({"median": median})

    ax.set_xlabel("Frequency [Hz]", fontsize=style["fontsize"])
    ax.set_ylabel("Strain (magnitude)", fontsize=style["fontsize"])
    ax.legend(fontsize=style["fontsize"])
    ax.tick_params(labelsize=style["fontsize"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)
    to_save.update(
        {
            "frequency": frequency,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "CL": confidence_level,
        }
    )
    plt.close()
    return fig, to_save


def plot_time_bias(reconstructed, reference, confidence_level=0.95, **kwargs):
    """Plot the time-domain bias of the waveform reconstruction."""
    style = _style(kwargs)
    reconstructed_data = _waveform_array(reconstructed)
    reference_data = _waveform_data(reference)
    time = reference.sample_times.data
    to_save = {}

    bias = reconstructed_data - np.asarray(reference_data)

    if style.get("normalize", False):
        bias = bias / np.abs(np.asarray(reference_data))

    lower_bound, upper_bound = compute_confidence_intervals(
        bias,
        confidence_level,
        method=style["percentile_method"],
        reference_waveform=reference_data,
    )
    to_save.update(
        {
            "time": time,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "CL": confidence_level,
        }
    )

    fig, ax = plt.subplots(figsize=style["figsize"])

    ax.fill_between(
        time,
        lower_bound,
        upper_bound,
        color=style["CL_color"],
        alpha=style["CL_alpha"],
        label=f"{confidence_level}% CI",
    )

    mean_bias = np.nanmean(bias, axis=0)
    ax.plot(
        time,
        mean_bias,
        label="Mean Bias",
        color=style["mean_color"],
        linestyle=style["mean_linestyle"],
    )
    to_save.update({"mean_bias": mean_bias})

    if style["plot_median"]:
        median_bias = np.nanmedian(bias, axis=0)
        ax.plot(
            time,
            median_bias,
            label="Median Bias",
            color=style["median_color"],
            linestyle=style["median_linestyle"],
        )
        to_save.update({"median_bias": median_bias})

    ax.legend(fontsize=style["fontsize"])
    ax.grid(True)
    ax.tick_params(labelsize=style["fontsize"])
    ax.set_ylabel("Bias", fontsize=style["fontsize"])
    ax.set_xlabel("Time [s]", fontsize=style["fontsize"])

    plt.close()
    return fig, to_save


def plot_frequency_bias(reconstructed, reference, confidence_level=0.95, **kwargs):
    """Plot the frequency-domain bias of the waveform reconstruction."""
    style = _style(kwargs)
    delta_t = getattr(reconstructed[0], "_delta_t", None)

    reconstructed_data = np.abs(_waveform_array(reconstructed))
    reference_data = np.abs(_waveform_data(reference))
    to_save = {}

    n_samples = len(reconstructed_data[0])
    frequencies = np.fft.fftfreq(n_samples, d=delta_t)
    mask = frequencies >= 0
    frequencies = frequencies[mask]

    bias = reconstructed_data - reference_data
    if style.get("normalize", False):
        bias = bias / np.abs(reference_data)

    lower_bound, upper_bound = compute_confidence_intervals(
        bias,
        confidence_level,
        method=style["percentile_method"],
        reference_waveform=reference_data,
    )
    mean_bias = np.nanmean(bias, axis=0)
    to_save.update(
        {
            "frequency": frequencies,
            "lower_bound": lower_bound[mask],
            "upper_bound": upper_bound[mask],
            "CL": confidence_level,
            "mean_bias": mean_bias[mask],
        }
    )

    fig, ax = plt.subplots(figsize=style["figsize"])
    ax.fill_between(
        frequencies,
        lower_bound[mask],
        upper_bound[mask],
        color=style["CL_color"],
        alpha=style["CL_alpha"],
        label=f"{confidence_level}% CI",
    )
    ax.plot(
        frequencies,
        mean_bias[mask],
        label="Mean Bias",
        color=style["mean_color"],
        linestyle=style["mean_linestyle"],
    )

    if style["plot_median"]:
        median_bias = np.nanmedian(bias, axis=0)[mask]
        ax.plot(
            frequencies,
            median_bias,
            label="Median Bias",
            color=style["median_color"],
            linestyle=style["median_linestyle"],
        )
        to_save.update({"median_bias": median_bias})

    ax.legend(fontsize=style["fontsize"])
    ax.grid(True)
    ax.tick_params(labelsize=style["fontsize"])

    ax.set_ylabel("Bias", fontsize=style["fontsize"])
    ax.set_xlabel("Frequency [Hz]", fontsize=style["fontsize"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.close()
    return fig, to_save


def plot_overlap(reconstructed, reference, **kwargs):
    """Plot overlap between reconstructed waveforms and one injected waveform."""
    style = _style(kwargs)
    reconstructed_data = _waveform_array(reconstructed)
    to_save = {}
    overlaps = compute_overlap(reconstructed_data, reference)

    to_save.update({"overlaps": overlaps})

    fig, ax = plt.subplots(figsize=style["figsize"])
    bins = int(np.sqrt(len(overlaps)))
    ax.hist(
        overlaps,
        bins=bins,
        density=True,
        histtype="step",
        label="Reconstructed Overlap",
        color=style["inj_color"],
    )

    if style["plot_mean"]:
        mean_reconstructed = np.nanmean(reconstructed_data, axis=0)
        mean_overlap = compute_overlap(mean_reconstructed, reference)
        ax.axvline(
            mean_overlap,
            color=style["mean_color"],
            linestyle=style["mean_linestyle"],
            label="Overlap of the mean",
        )
        to_save.update({"mean_overlap": mean_overlap})

    if style["plot_median"]:
        median_reconstructed = np.nanmedian(reconstructed_data, axis=0)
        median_overlap = compute_overlap(median_reconstructed, reference)
        ax.axvline(
            median_overlap,
            color=style["median_color"],
            linestyle=style["median_linestyle"],
            label="Overlap of the median",
        )
        to_save.update({"median_overlap": median_overlap})

    ax.set_xlabel("Overlap", fontsize=style["fontsize"])
    ax.set_ylabel("Density", fontsize=style["fontsize"])
    ax.legend(fontsize=style["fontsize"])
    ax.grid(True)
    ax.tick_params(labelsize=style["fontsize"])
    plt.close()
    return fig, to_save


def plot_auto_overlap(reconstructed, injected):
    """Plot paired reconstructed/injected overlap."""
    overlaps = compute_overlap(reconstructed, injected)
    mean_overlap = np.mean(overlaps)

    fig, ax = plt.subplots(figsize=DEFAULT_PLOT_KWARGS["figsize"])
    bins = int(np.sqrt(len(overlaps)))
    ax.hist(
        overlaps,
        bins=bins,
        density=True,
        histtype="step",
        color=DEFAULT_PLOT_KWARGS["inj_color"],
    )
    ax.axvline(
        mean_overlap,
        color=DEFAULT_PLOT_KWARGS["mean_color"],
        linestyle=DEFAULT_PLOT_KWARGS["mean_linestyle"],
        label="Mean Overlap",
    )
    ax.set_xlabel("Overlap", fontsize=DEFAULT_PLOT_KWARGS["fontsize"])
    ax.set_ylabel("Density", fontsize=DEFAULT_PLOT_KWARGS["fontsize"])
    ax.grid(True)
    ax.tick_params(labelsize=DEFAULT_PLOT_KWARGS["fontsize"])

    to_save = {"overlaps": overlaps}
    plt.close()
    return fig, to_save


def plot_time_cumulative_hrss(
    reconstructed,
    reference,
    injected=None,
    confidence_level=0.95,
    **kwargs,
):
    """Plot the cumulative HRSS distribution in the time domain."""
    style = _style(kwargs)
    reconstructed_data = _waveform_array(reconstructed)
    reference_data = _waveform_data(reference)
    reference_hrss = compute_cumulative_hrss(reference_data, reference._delta_t, axis=0)
    reconstructed_hrss = (
        compute_cumulative_hrss(reconstructed_data, reference._delta_t, axis=1)
        / reference_hrss[-1]
    )
    lower_bound, upper_bound = compute_confidence_intervals(
        reconstructed_hrss,
        confidence_level=confidence_level,
        method=style["percentile_method"],
        reference_waveform=reference_data,
    )
    mean_hrss = reconstructed_hrss.mean(axis=0)
    time = reference.sample_times.data

    to_save = {
        "CL": confidence_level,
        "reference_hrss": reference_hrss / reference_hrss[-1],
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "mean_hrss": mean_hrss,
        "time": time,
    }

    fig, ax = plt.subplots(figsize=style["figsize"])

    ax.plot(
        time,
        reference_hrss / reference_hrss[-1],
        label="On-Source HRSS",
        color=style["inj_color"],
        linestyle=style["inj_ls"],
    )
    ax.plot(
        time,
        mean_hrss,
        label="Mean Reconstructed HRSS",
        color=style["mean_color"],
        linestyle=style["mean_linestyle"],
    )
    ax.fill_between(
        time,
        lower_bound,
        upper_bound,
        color=style["CL_color"],
        alpha=style["CL_alpha"],
        label=f"{confidence_level * 100}% CI",
    )

    if injected is not None:
        injected_hrss = compute_cumulative_hrss(
            _waveform_data(injected), injected._delta_t, axis=0
        )
        ax.plot(
            time,
            injected_hrss / reference_hrss[-1],
            label="Injected HRSS",
            color=style["inj_color"],
            linestyle=style["inj_ls"],
        )
        to_save["injected_hrss"] = injected_hrss

    if style["plot_median"]:
        median_hrss = np.median(reconstructed_hrss, axis=0)
        ax.plot(
            time,
            median_hrss,
            label="Median Reconstructed HRSS",
            color=style["median_color"],
            linestyle=style["median_linestyle"],
        )
        to_save["median_hrss"] = median_hrss

    ax.grid(True)
    ax.legend(fontsize=style["fontsize"])
    ax.tick_params(labelsize=style["fontsize"])
    ax.set_ylabel("Cumulative HRSS (normalized)", fontsize=style["fontsize"])
    ax.set_xlabel("GPS Time [s]", fontsize=style["fontsize"])

    plt.close()
    return fig, to_save


def plot_frequency_cumulative_hrss(
    reconstructed,
    reference,
    injected=None,
    confidence_level=0.95,
    plot_median=True,
    **kwargs,
):
    """Plot the cumulative HRSS distribution in the frequency domain."""
    style = _style({"plot_median": plot_median, **kwargs})
    reconstructed_data = _waveform_array(reconstructed)
    reference_data = _waveform_data(reference)

    reference_hrss = compute_cumulative_hrss(reference_data, reference._delta_t, axis=0)
    reconstructed_hrss = (
        compute_cumulative_hrss(reconstructed_data, reference._delta_t, axis=1)
        / reference_hrss[-1]
    )
    lower_bound, upper_bound = compute_confidence_intervals(
        reconstructed_hrss,
        confidence_level=confidence_level,
        method=style["percentile_method"],
        reference_waveform=reference_data,
    )
    mean_hrss = reconstructed_hrss.mean(axis=0)

    delta_t = reconstructed[0]._delta_t
    n_samples = len(reference_data)
    frequency = np.fft.fftfreq(n_samples, d=delta_t)
    mask = frequency >= 0
    frequency = frequency[mask]

    to_save = {
        "CL": confidence_level,
        "reference_hrss": reference_hrss[mask],
        "lower_bound": lower_bound[mask],
        "upper_bound": upper_bound[mask],
        "mean_hrss": mean_hrss[mask],
        "frequency": frequency,
    }

    fig, ax = plt.subplots(figsize=style["figsize"])
    ax.plot(
        frequency,
        reference_hrss[mask] / reference_hrss[-1],
        label="Injected HRSS",
        color=style["ref_color"],
        linestyle=style["ref_ls"],
    )
    ax.plot(
        frequency,
        mean_hrss[mask],
        label="Mean Reconstructed HRSS",
        color=style["mean_color"],
        linestyle=style["mean_linestyle"],
    )
    ax.fill_between(
        frequency,
        lower_bound[mask],
        upper_bound[mask],
        color=style["CL_color"],
        alpha=style["CL_alpha"],
        label=f"{confidence_level}% CI",
    )

    if injected is not None:
        injected_hrss = compute_cumulative_hrss(
            _waveform_data(injected), injected._delta_t, axis=0
        )
        ax.plot(
            frequency,
            injected_hrss[mask] / reference_hrss[-1],
            label="Injected HRSS",
            color=style["inj_color"],
            linestyle=style["inj_ls"],
        )
        to_save["injected_hrss"] = injected_hrss[mask]

    if style["plot_median"]:
        median_hrss = np.median(reconstructed_hrss, axis=0)[mask]
        ax.plot(
            frequency,
            median_hrss,
            label="Median Reconstructed HRSS",
            color=style["median_color"],
            linestyle=style["median_linestyle"],
        )
        to_save["median_hrss"] = median_hrss

    ax.grid(True)
    ax.legend(fontsize=style["fontsize"])
    ax.tick_params(labelsize=style["fontsize"])
    ax.set_ylabel("Cumulative HRSS (normalized)", fontsize=style["fontsize"])
    ax.set_xlabel("Frequency [Hz]", fontsize=style["fontsize"])
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.close()
    return fig, to_save


def plot_leakage(reconstructed, injected):
    """Plot leakage of reconstructed waveforms after the injected waveform."""
    t = np.arange(0, 20, 1) / 20
    leakage_mean, leakage_std = compute_leakage(reconstructed, injected, t)

    fig, ax = plt.subplots(figsize=DEFAULT_PLOT_KWARGS["figsize"])
    ax.errorbar(
        t,
        leakage_mean,
        yerr=leakage_std,
        fmt="o",
        color=DEFAULT_PLOT_KWARGS["inj_color"],
        label="Leakage",
    )

    ax.set_xlabel(
        "Time [s] - Injection end time", fontsize=DEFAULT_PLOT_KWARGS["fontsize"]
    )
    ax.set_ylabel("Leakage", fontsize=DEFAULT_PLOT_KWARGS["fontsize"])
    ax.legend(fontsize=DEFAULT_PLOT_KWARGS["fontsize"])
    ax.grid(True)
    ax.tick_params(labelsize=DEFAULT_PLOT_KWARGS["fontsize"])

    to_save = {"leakage_mean": leakage_mean, "leakage_std": leakage_std, "time": t}
    plt.close()
    return fig, to_save


def plot_hrss(reconstructed, injected):
    """Plot HRSS of reconstructed waveforms compared to the injected waveform."""
    reconstructed_hrss = []
    for waveform in reconstructed:
        reconstructed_hrss.append(compute_hrss(waveform.data, injected._delta_t))
    injected_hrss = compute_hrss(injected.data, injected._delta_t)

    fig, ax = plt.subplots(figsize=DEFAULT_PLOT_KWARGS["figsize"])
    ax.hist(
        reconstructed_hrss,
        bins=30,
        density=True,
        histtype="step",
        label="Reconstructed HRSS",
        color=DEFAULT_PLOT_KWARGS["inj_color"],
    )
    ax.axvline(
        injected_hrss,
        color=DEFAULT_PLOT_KWARGS["mean_color"],
        linestyle=DEFAULT_PLOT_KWARGS["mean_linestyle"],
        label="Injected HRSS",
    )

    ax.set_xlabel("hrss", fontsize=DEFAULT_PLOT_KWARGS["fontsize"])
    ax.set_ylabel("Density", fontsize=DEFAULT_PLOT_KWARGS["fontsize"])
    ax.legend(fontsize=DEFAULT_PLOT_KWARGS["fontsize"])
    ax.grid(True)
    ax.tick_params(labelsize=DEFAULT_PLOT_KWARGS["fontsize"])

    to_save = {"reconstructed_hrss": reconstructed_hrss, "injected_hrss": injected_hrss}
    plt.close()
    return fig, to_save
