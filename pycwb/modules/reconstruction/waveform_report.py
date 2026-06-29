import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import mode
from tqdm import tqdm

from pycwb.modules.reconstruction.waveform_report_io import (
    load_wf_from_wave,
    pad_waveforms,
    slice_waveforms,
    sync_waveforms,
)
from pycwb.modules.reconstruction.waveform_report_plots import (
    plot_frequency_bias,
    plot_frequency_cumulative_hrss,
    plot_frequency_waveform_reconstruction,
    plot_hrss,
    plot_overlap,
    plot_time_bias,
    plot_time_cumulative_hrss,
    plot_time_waveform_reconstruction,
)
from pycwb.types.waveform import Waveform


logger = logging.getLogger(__name__)


def generate_reconstruction_report(
    wave_file: str,
    ifo: str,
    output_folder: str,
    whitened: bool = False,
    confidence_level: float = 0.95,
    **kwargs,
) -> None:
    """
    Generate aggregate waveform reconstruction plots and result arrays.

    Reads event waveforms from the consolidated wave.h5 file written by
    ``add_wf_to_wave``.
    """
    plots_folder = os.path.join(output_folder, "reports/plots")
    results_folder = os.path.join(output_folder, "reports/results")
    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    rec_key = "wf_REC_whiten" if whitened else "wf_REC"
    inj_key = "wf_INJ_whiten" if whitened else "wf_INJ"
    nul_key = "wf_NUL_whiten" if whitened else "wf_NUL"

    logger.info("Processing waveform reconstruction report for %s from %s", ifo, wave_file)

    rec_waveforms, reference = _load_and_sync_waveforms(
        wave_file, ifo, rec_key, inj_key, **kwargs
    )
    if not rec_waveforms:
        logger.warning("No valid waveforms found for %s in %s", ifo, wave_file)
        return

    _generate_time_domain_reports(
        plots_folder,
        results_folder,
        ifo,
        rec_waveforms,
        reference,
        whitened,
        confidence_level,
        **kwargs,
    )
    _generate_frequency_domain_reports(
        plots_folder,
        results_folder,
        ifo,
        rec_waveforms,
        reference,
        whitened,
        confidence_level,
        **kwargs,
    )

    null_waveforms, _ = _load_and_sync_waveforms(wave_file, ifo, nul_key, inj_key, **kwargs)
    if null_waveforms:
        _generate_null_stream_report(
            plots_folder,
            results_folder,
            ifo,
            null_waveforms,
            whitened,
            confidence_level,
            **kwargs,
        )


def _sync_report_pair(args):
    rec_ts, inj_ts, scale, early_start = args
    rec = Waveform(rec_ts)
    ref = Waveform(inj_ts)
    if scale != 0:
        rec.data *= np.sqrt(2) ** scale
    try:
        rec, ref = sync_waveforms(rec, ref, None, None)
        if len(rec) != len(ref):
            rec, ref = pad_waveforms(rec, ref)
        rec, ref = slice_waveforms(rec, ref, early_start)
        return rec, ref
    except Exception:
        logger.debug("Skipping waveform pair during synchronization", exc_info=True)
        return None


def _load_and_sync_waveforms(wave_file: str, ifo: str, rec_key: str, inj_key: str, **kwargs):
    """
    Load paired reconstructed/injection waveforms from wave.h5 and synchronize them.
    """
    early_start = kwargs.get("early_start", 0)
    scale = kwargs.get("scale", 0)
    max_workers = kwargs.get("max_workers") or os.cpu_count()

    raw = load_wf_from_wave(wave_file, ifo, [rec_key, inj_key])
    pairs = list(zip(raw[rec_key], raw[inj_key]))
    if not pairs:
        return [], None

    sync_args = [(rec_ts, inj_ts, scale, early_start) for rec_ts, inj_ts in pairs]
    if max_workers == 1:
        results = list(
            tqdm(
                map(_sync_report_pair, sync_args),
                total=len(sync_args),
                desc=f"Syncing {ifo} waveforms",
            )
        )
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            results = list(
                tqdm(
                    exe.map(_sync_report_pair, sync_args),
                    total=len(sync_args),
                    desc=f"Syncing {ifo} waveforms",
                )
            )

    valid = [result for result in results if result is not None]
    if not valid:
        return [], None

    rec_list, ref_list = zip(*valid)
    rec_list = list(rec_list)

    lengths = [len(waveform) for waveform in rec_list]
    mode_length = int(np.asarray(mode(lengths, keepdims=False).mode).item())
    filtered = [
        (rec, inj)
        for rec, inj in zip(rec_list, ref_list)
        if len(rec) == mode_length
    ]
    if len(filtered) < len(rec_list):
        logger.warning(
            "Removed %s waveforms with non-modal length (mode=%s)",
            len(rec_list) - len(filtered),
            mode_length,
        )

    if not filtered:
        return [], None

    rec_list, ref_list = zip(*filtered)
    reference = next((ref for ref in ref_list if len(ref) == mode_length), None)
    return list(rec_list), reference


def _generate_time_domain_reports(
    plots_folder,
    results_folder,
    ifo,
    waveforms,
    reference,
    whitened,
    confidence_level,
    **kwargs,
):
    ordering = kwargs.get("ordering", "percentiles")
    suffix = "_wth" if whitened else ""

    fig, data = plot_time_waveform_reconstruction(
        waveforms,
        reference,
        confidence_level=confidence_level,
        percentile_method=ordering,
        **kwargs,
    )
    _save_report(fig, data, plots_folder, results_folder, f"time_waveform_reconstruction_{ifo}{suffix}")

    fig, data = plot_time_bias(
        waveforms,
        reference,
        confidence_level=confidence_level,
        percentile_method=ordering,
        normalize=False,
        **kwargs,
    )
    _save_report(fig, data, plots_folder, results_folder, f"time_bias_{ifo}{suffix}")

    fig, data = plot_overlap(waveforms, reference, **kwargs)
    _save_report(fig, data, plots_folder, results_folder, f"overlap_{ifo}{suffix}")

    fig, data = plot_time_cumulative_hrss(
        waveforms,
        reference,
        confidence_level=confidence_level,
        percentile_method=ordering,
        **kwargs,
    )
    _save_report(fig, data, plots_folder, results_folder, f"cumulative_hrss_{ifo}{suffix}")

    fig, data = plot_hrss(waveforms, reference)
    _save_report(fig, data, plots_folder, results_folder, f"hrss_{ifo}{suffix}")


def _generate_frequency_domain_reports(
    plots_folder,
    results_folder,
    ifo,
    waveforms,
    reference,
    whitened,
    confidence_level,
    **kwargs,
):
    ordering = kwargs.get("ordering", "percentiles")
    suffix = "_wth" if whitened else ""

    freq_waveforms = [waveform.copy() for waveform in waveforms]
    for waveform in freq_waveforms:
        waveform.fft()
    freq_reference = reference.copy()
    freq_reference.fft()

    fig, data = plot_frequency_waveform_reconstruction(
        freq_waveforms,
        freq_reference,
        confidence_level=confidence_level,
        percentile_method=ordering,
        **kwargs,
    )
    _save_report(
        fig,
        data,
        plots_folder,
        results_folder,
        f"frequency_waveform_reconstruction_{ifo}{suffix}",
    )

    fig, data = plot_frequency_bias(
        freq_waveforms,
        freq_reference,
        confidence_level=confidence_level,
        percentile_method=ordering,
        **kwargs,
    )
    _save_report(fig, data, plots_folder, results_folder, f"frequency_bias_{ifo}{suffix}")

    fig, data = plot_frequency_cumulative_hrss(
        freq_waveforms,
        freq_reference,
        confidence_level=confidence_level,
        percentile_method=ordering,
        **kwargs,
    )
    _save_report(
        fig,
        data,
        plots_folder,
        results_folder,
        f"frequency_cumulative_hrss_{ifo}{suffix}",
    )


def _generate_null_stream_report(
    plots_folder,
    results_folder,
    ifo,
    null_waveforms,
    whitened,
    confidence_level,
    **kwargs,
):
    ordering = kwargs.get("ordering", "percentiles")
    suffix = "_wth" if whitened else ""

    fig, data = plot_time_waveform_reconstruction(
        null_waveforms,
        confidence_level=confidence_level,
        percentile_method=ordering,
        **kwargs,
    )
    _save_report(fig, data, plots_folder, results_folder, f"null_reconstruction_{ifo}{suffix}")


def _save_report(fig, data, plots_folder, results_folder, name):
    fig.savefig(os.path.join(plots_folder, f"{name}.png"), bbox_inches="tight")
    np.savez(os.path.join(results_folder, f"{name}.npz"), **data)
