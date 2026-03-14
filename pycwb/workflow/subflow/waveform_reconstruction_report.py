import os
import logging
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import mode
import numpy as np
from tqdm import tqdm

from pycwb.modules.post_production.waveform_reconstruction_plot import (
    plot_time_waveform_reconstruction,
    plot_time_bias,
    plot_overlap,
    plot_time_cumulative_hrss,
    plot_hrss,
    plot_frequency_waveform_reconstruction,
    plot_frequency_bias,
    plot_frequency_cumulative_hrss,
)
from pycwb.modules.post_production.waveform_reconstruction import (
    sync_waveforms, slice_waveforms, pad_waveforms,
)
from pycwb.types.waveform import Waveform
from pycwb.workflow.subflow.postprocess_and_plots import load_wf_from_wave

logger = logging.getLogger(__name__)


def generate_reconstruction_report(wave_file: str, ifo: str, output_folder: str,
                                    whitened: bool = False, confidence_level: float = 0.95,
                                    **kwargs) -> None:
    """
    Generate aggregate statistical waveform reconstruction report for a given IFO.

    Reads all event waveforms from the consolidated wave.h5 file (written by
    add_wf_to_wave) and produces time-domain, frequency-domain, and null-stream
    plots and result arrays.

    Parameters
    ----------
    wave_file : str
        Path to the wave HDF5 file produced by add_wf_to_wave.
    ifo : str
        Interferometer name, e.g. 'H1'.
    output_folder : str
        Root directory for output; plots go to ``{output_folder}/reports/plots``
        and result arrays to ``{output_folder}/reports/results``.
    whitened : bool
        Use whitened waveforms when True.
    confidence_level : float
        Confidence level for statistical intervals (default 0.95).
    """
    plots_folder = os.path.join(output_folder, "reports/plots")
    results_folder = os.path.join(output_folder, "reports/results")
    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    rec_key = 'wf_REC_whiten' if whitened else 'wf_REC'
    inj_key = 'wf_INJ_whiten' if whitened else 'wf_INJ'
    nul_key = 'wf_NUL_whiten' if whitened else 'wf_NUL'

    logger.info(f"Processing waveform reconstruction report for {ifo} from {wave_file}")

    rec_waveforms, reference = _load_and_sync_waveforms(wave_file, ifo, rec_key, inj_key, **kwargs)
    if not rec_waveforms:
        logger.warning(f"No valid waveforms found for {ifo} in {wave_file}")
        return

    _generate_time_domain_reports(plots_folder, results_folder, ifo,
                                   rec_waveforms, reference, whitened, confidence_level, **kwargs)
    _generate_frequency_domain_reports(plots_folder, results_folder, ifo,
                                        rec_waveforms, reference, whitened, confidence_level, **kwargs)

    null_waveforms, _ = _load_and_sync_waveforms(wave_file, ifo, nul_key, inj_key, **kwargs)
    if null_waveforms:
        _generate_null_stream_report(plots_folder, results_folder, ifo,
                                      null_waveforms, whitened, confidence_level, **kwargs)


def _load_and_sync_waveforms(wave_file: str, ifo: str, rec_key: str, inj_key: str, **kwargs):
    """
    Load paired (reconstructed, injection) waveforms from wave.h5 and synchronize
    each reconstructed waveform to its reference injection waveform.

    Returns the list of synchronized reconstructed waveforms and a single
    representative reference waveform for plotting.
    """
    early_start = kwargs.get('early_start', 0)
    scale = kwargs.get('scale', 0)
    max_workers = kwargs.get('max_workers', os.cpu_count())

    raw = load_wf_from_wave(wave_file, ifo, [rec_key, inj_key])
    pairs = list(zip(raw[rec_key], raw[inj_key]))
    if not pairs:
        return [], None

    def _sync_pair(pair):
        rec_ts, inj_ts = pair
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
        except Exception as e:
            logger.debug(f"Skipping waveform pair due to sync error: {e}")
            return None

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        results = list(tqdm(exe.map(_sync_pair, pairs), total=len(pairs),
                            desc=f"Syncing {ifo} waveforms"))

    valid = [r for r in results if r is not None]
    if not valid:
        return [], None

    rec_list, ref_list = zip(*valid)
    rec_list = list(rec_list)

    # filter to modal length to remove partially-overlapping edge waveforms
    lengths = [len(w) for w in rec_list]
    mode_length = mode(lengths).mode
    filtered = [(r, i) for r, i in zip(rec_list, ref_list) if len(r) == mode_length]
    if len(filtered) < len(rec_list):
        logger.warning(f"Removed {len(rec_list) - len(filtered)} waveforms "
                       f"with non-modal length (mode={mode_length})")

    if not filtered:
        return [], None

    rec_list, ref_list = zip(*filtered)
    reference = next((r for r in ref_list if len(r) == mode_length), None)
    return list(rec_list), reference


def _generate_time_domain_reports(plots_folder, results_folder, ifo, waveforms, reference,
                                   whitened, confidence_level, **kwargs):
    ordering = kwargs.get('ordering', 'percentiles')
    suffix = '_wth' if whitened else ''

    fig, data = plot_time_waveform_reconstruction(
        waveforms, reference, confidence_level=confidence_level,
        percentile_method=ordering, **kwargs)
    _save_report(fig, data, plots_folder, results_folder,
                 f"time_waveform_reconstruction_{ifo}{suffix}")

    fig, data = plot_time_bias(
        waveforms, reference, confidence_level=confidence_level,
        percentile_method=ordering, normalize=False, **kwargs)
    _save_report(fig, data, plots_folder, results_folder, f"time_bias_{ifo}{suffix}")

    fig, data = plot_overlap(waveforms, reference, **kwargs)
    _save_report(fig, data, plots_folder, results_folder, f"overlap_{ifo}{suffix}")

    fig, data = plot_time_cumulative_hrss(
        waveforms, reference, confidence_level=confidence_level,
        percentile_method=ordering, **kwargs)
    _save_report(fig, data, plots_folder, results_folder, f"cumulative_hrss_{ifo}{suffix}")

    fig, data = plot_hrss(waveforms, reference)
    _save_report(fig, data, plots_folder, results_folder, f"hrss_{ifo}{suffix}")


def _generate_frequency_domain_reports(plots_folder, results_folder, ifo, waveforms, reference,
                                        whitened, confidence_level, **kwargs):
    ordering = kwargs.get('ordering', 'percentiles')
    suffix = '_wth' if whitened else ''

    # FFT copies so the caller's time-domain waveforms are not mutated
    freq_waveforms = [w.copy() for w in waveforms]
    [w.fft() for w in freq_waveforms]
    freq_reference = reference.copy()
    freq_reference.fft()

    fig, data = plot_frequency_waveform_reconstruction(
        freq_waveforms, freq_reference, confidence_level=confidence_level,
        percentile_method=ordering, **kwargs)
    _save_report(fig, data, plots_folder, results_folder,
                 f"frequency_waveform_reconstruction_{ifo}{suffix}")

    fig, data = plot_frequency_bias(
        freq_waveforms, reference, confidence_level=confidence_level,
        percentile_method=ordering, **kwargs)
    _save_report(fig, data, plots_folder, results_folder, f"frequency_bias_{ifo}{suffix}")

    fig, data = plot_frequency_cumulative_hrss(
        freq_waveforms, reference, confidence_level=confidence_level,
        percentile_method=ordering, **kwargs)
    _save_report(fig, data, plots_folder, results_folder,
                 f"frequency_cumulative_hrss_{ifo}{suffix}")


def _generate_null_stream_report(plots_folder, results_folder, ifo, null_waveforms,
                                  whitened, confidence_level, **kwargs):
    ordering = kwargs.get('ordering', 'percentiles')
    suffix = '_wth' if whitened else ''

    fig, data = plot_time_waveform_reconstruction(
        null_waveforms, confidence_level=confidence_level,
        percentile_method=ordering, **kwargs)
    _save_report(fig, data, plots_folder, results_folder, f"null_reconstruction_{ifo}{suffix}")


def _save_report(fig, data, plots_folder, results_folder, name):
    fig.savefig(os.path.join(plots_folder, f"{name}.png"), bbox_inches='tight')
    np.savez(os.path.join(results_folder, f"{name}.npz"), **data)

