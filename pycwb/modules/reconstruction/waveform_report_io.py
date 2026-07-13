import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np  # type: ignore
import h5py as h5
from tqdm import tqdm  # type: ignore

from pycwb.types.time_series import TimeSeries
from pycwb.types.waveform import Waveform, load_waveform


logger = logging.getLogger(__name__)


def save(figure, results_dictionary, directory, filename, extension="pdf"):
    """Save a waveform report figure and its numerical results."""
    plots_dir, results_dir = create_save_directories(directory)

    figure_path = os.path.join(plots_dir, f"{filename}.{extension}")
    figure.savefig(figure_path, bbox_inches="tight")

    results_path = os.path.join(results_dir, f"{filename}.npz")
    np.savez(results_path, **results_dictionary)


def create_save_directories(analysis_directory):
    """Create the waveform report output directories."""
    if not os.path.exists(analysis_directory):
        raise FileNotFoundError(
            f"The provided analysis directory {analysis_directory} does not exist."
        )

    plot_directory = os.path.join(
        analysis_directory, "reports", "waveform_reconstruction_plots"
    )
    results_directory = os.path.join(
        analysis_directory, "reports", "waveform_reconstruction_results"
    )

    os.makedirs(plot_directory, exist_ok=True)
    os.makedirs(results_directory, exist_ok=True)
    return plot_directory, results_directory


def load_and_slice(args):
    file, reference, time_shift, phase_shift, early_start, scale = args

    if time_shift is None and phase_shift is None and reference is None:
        raise ValueError(
            "Either one among reference, time_shift or phase_shift is needed, "
            f"while ({reference}, {time_shift}, {phase_shift})"
        )

    waveform = load_waveform(file, skip_nans=True)
    if waveform is None:
        return None
    reference = load_waveform(reference)

    if scale != 0:
        waveform.data *= np.sqrt(2) ** scale

    try:
        waveform, reference = sync_waveforms(waveform, reference, time_shift, phase_shift)
        if len(waveform) != len(reference):
            waveform, reference = pad_waveforms(waveform, reference)
        waveform, reference = slice_waveforms(waveform, reference, early_start)
    except Exception:
        logger.debug("Skipping waveform during load/slice", exc_info=True)
        return None

    return waveform, reference


def load_wf_from_wave(wave_file: str, ifo: str, keys: list[str]) -> dict[str, list]:
    """
    Load IFO waveforms from a consolidated wave HDF5 file.

    This is the paired reader for ``add_wf_to_wave`` in the workflow subflow.
    """
    result: dict[str, list] = {key: [] for key in keys}
    with h5.File(wave_file, "r") as f:
        for event_id in sorted(f.keys()):
            for key in keys:
                full_key = f"{ifo}_{key}"
                if full_key in f[event_id]:
                    dataset = f[event_id][full_key]
                    if "sample_rate" in dataset.attrs and "start_time" in dataset.attrs:
                        value = TimeSeries(
                            dataset[:],
                            delta_t=1.0 / dataset.attrs["sample_rate"],
                            epoch=dataset.attrs["start_time"],
                        )
                    else:
                        value = dataset[:]
                    result[key].append(value)
    return result


def sync_waveforms(waveform, reference, time_shift=None, phase_shift=None):
    """
    Synchronize a single waveform to the reference waveform.

    Returns the synchronized waveform and reference waveform.
    """
    w_copy, r_copy = waveform.copy(), reference.copy()

    if time_shift is None and phase_shift is None:
        w_copy, r_copy = w_copy.syncWaveform(r_copy, sync_phase=True)

    if time_shift:
        w_copy.timeShift(time_shift)

    if phase_shift:
        w_copy.phaseShift(phase_shift)

    return w_copy, r_copy


def pad_waveforms(waveform, reference):
    """
    Pad two waveforms with zeros so they share the same start and end times.

    Returns both padded waveforms.
    """
    w, r = waveform.copy(), reference.copy()
    dt = w.delta_t

    t_start = min(w.sample_times[0], r.sample_times[0])
    t_end = max(w.sample_times[-1], r.sample_times[-1])

    n_pre_w = int(round((w.sample_times[0] - t_start) / dt))
    n_post_w = int(round((t_end - w.sample_times[-1]) / dt))

    if n_pre_w > 0:
        w.prepend_zeros(n_pre_w)
    if n_post_w > 0:
        w.append_zeros(n_post_w)

    n_pre_r = int(round((r.sample_times[0] - t_start) / dt))
    n_post_r = int(round((t_end - r.sample_times[-1]) / dt))
    if n_pre_r > 0:
        r.prepend_zeros(n_pre_r)
    if n_post_r > 0:
        r.append_zeros(n_post_r)

    w._findStartEnd()
    r._findStartEnd()

    return w, r


def slice_waveforms(waveform, reference, early_start=0):
    """
    Slice a waveform pair to the reference waveform's active time range.

    Returns the sliced waveform and reference waveform.
    """
    w, r = waveform.copy(), reference.copy()

    start, stop = r.istart, int(r.iend)
    start = int(max(start - early_start * w.sample_rate, 0))

    w = Waveform(TimeSeries(w[start:stop], dt=1 / w.sample_rate, t0=r.tstart), folder=w.folder)
    r = Waveform(TimeSeries(r[start:stop], dt=1 / r.sample_rate, t0=r.tstart), folder=r.folder)

    w._total_time_shift = waveform._total_time_shift
    w._total_phase_shift = waveform._total_phase_shift

    return w, r


def load_one_waveform_OLD(args):
    """
    Load a single waveform from the specified folder and subfolder.

    Returns a Waveform object or None if loading fails.
    """
    folder, ifo, type_, whitened, file_format, resample, skip_trigger = args

    try:
        file_name = (
            f"{ifo}_wf_{type_}.{file_format}"
            if not whitened
            else f"{ifo}_wf_{type_}_whiten.{file_format}"
        )
        waveform = load_waveform(os.path.join(folder, file_name), resample=resample)

        if np.any(np.isnan(waveform.data)):
            return None

        if skip_trigger and f"{ifo}_wf_INJ.{file_format}" not in folder:
            return None

        return waveform

    except Exception:
        logger.debug("Skipping waveform during legacy load", exc_info=True)
        return None


def _sync_waveform_pair(args):
    waveform, reference, sync_phase, time_shift, phase_shift = args

    try:
        w_copy, r_copy = waveform.copy(), reference.copy()

        if time_shift is None and phase_shift is None:
            w_copy, r_copy = w_copy.syncWaveform(r_copy, sync_phase=sync_phase)

        if time_shift:
            w_copy.timeShift(time_shift)

        if phase_shift:
            w_copy.phaseShift(phase_shift)

        return w_copy, r_copy
    except Exception:
        logger.debug("Skipping waveform during legacy synchronization", exc_info=True)
        return None


def sync_waveforms_OLD(
    waveforms,
    reference,
    sync_phase=True,
    time_shift=None,
    phase_shift=None,
    max_workers=None,
):
    """
    Synchronize all waveforms to a single reference or a paired reference list.
    """
    sync_waveforms_ = []
    reference_waveforms_ = []
    discarded_waveforms = 0

    if isinstance(reference, Waveform):
        args = [(w, reference, sync_phase, time_shift, phase_shift) for w in waveforms]

    elif isinstance(reference, list) and len(reference) == len(waveforms):
        args = [
            (w, r, sync_phase, time_shift, phase_shift)
            for w, r in zip(waveforms, reference)
        ]

    else:
        raise ValueError(
            "Reference must be a single Waveform or a list with the same length as waveforms."
        )

    workers = max_workers or os.cpu_count()
    with ProcessPoolExecutor(max_workers=workers) as exe:
        results = list(
            tqdm(
                exe.map(_sync_waveform_pair, args),
                total=len(args),
                desc="Synchronizing waveforms (parallel)",
            )
        )

    for result in results:
        if result is None:
            discarded_waveforms += 1
            continue

        w_sync, ref = result
        sync_waveforms_.append(w_sync)
        reference_waveforms_.append(ref)

    if discarded_waveforms > 0:
        logger.warning(
            "%s over %s waveforms were discarded during synchronization.",
            discarded_waveforms,
            len(waveforms),
        )

    return sync_waveforms_, reference_waveforms_
