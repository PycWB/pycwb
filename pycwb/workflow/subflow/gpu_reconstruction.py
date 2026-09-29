"""Native whitened reconstruction required by catalog-only background Q-veto.

The native post-processing reconstructs six waveform products per event
through ``reconstruct_waveforms_flow`` even when nothing is saved. Q-veto only
needs the whitened ``REC`` and ``DAT`` waveforms, so :func:`reconstruct`
computes exactly those with the same native ``get_network_MRA_wave`` call and
:func:`make_save` passes it to the native save path for
the parent process. No file is ever written by this module.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any

from pycwb.modules.reconstruction import get_network_MRA_wave
from pycwb.workflow.subflow.postprocess_and_plots import reconstruct_waveforms_flow

from pycwb.constants.gpu_options import gpu_options

if TYPE_CHECKING:
    from collections.abc import Callable

    from pycwb.config import Config
    from pycwb.types.network_cluster import Cluster
    from pycwb.types.network_event import Event
    from pycwb.types.time_series import TimeSeries

logger = logging.getLogger(__name__)

PLOT_OR_SAVE_FLAGS = ("save_waveform", "plot_waveform", "plot_trigger", "plot_sky_map")
"""Config attributes that would require the products this module deliberately skips."""

WHITENED_PRODUCTS = (("signal", "REC"), ("strain", "DAT"))
"""``(get_network_MRA_wave kind, output label)`` pairs consumed by the native Q-veto."""


def reconstruct(
    trigger_folder: str,
    config: Config,
    ifos: list[str],
    event: Event,
    cluster: Cluster,
    epoch: float = 0.0,
    wave_file: str = "",
    save: bool = True,
    plot: bool = False,
    queue: Any = None,
) -> dict[str, TimeSeries]:
    """Compute only the whitened ``REC``/``DAT`` waveforms the native Q-veto reads.

    Drop-in for ``reconstruct_waveforms_flow`` with the same signature. Every
    argument that would imply a file product must be off; the function is
    meant to be reached only through :func:`make_save`.

    Parameters
    ----------
    trigger_folder : str
        Native trigger folder; unused except by the validation reference call.
    config : Config
        Search configuration providing ``rateANA``, ``nIFO`` and ``TDRate``.
    ifos : list of str
        Detector names, in network order.
    event : Event
        Event being post-processed; only ``injection`` and ``hash_id`` are read.
    cluster : Cluster
        Cluster whose pixels are reconstructed.
    epoch : float, optional
        GPS offset added to each waveform ``start_time``.
    wave_file : str, optional
        Accepted for signature compatibility; must not be used for saving.
    save : bool, optional
        Must be ``False``.
    plot : bool, optional
        Must be ``False``.
    queue
        Accepted for signature compatibility; unused.

    Returns
    -------
    dict of str to TimeSeries
        ``{"<ifo>_wf_REC_whiten": ..., "<ifo>_wf_DAT_whiten": ...}`` per detector.

    Raises
    ------
    ValueError
        If ``save``, ``plot``, ``event.injection`` or any of
        ``PLOT_OR_SAVE_FLAGS`` in ``config`` is set.
    AssertionError
        If ``gpu.validate_reconstruction=true`` and the result differs
        bitwise from the complete native ``reconstruct_waveforms_flow``.
    """
    if (
        save
        or plot
        or event.injection
        or any(getattr(config, flag, False) for flag in PLOT_OR_SAVE_FLAGS)
    ):
        raise ValueError(
            "Q-veto reconstruction requires background without saved waveforms or plots"
        )
    expected = None
    if gpu_options(config).validate_reconstruction:
        from pycwb.modules.stage_validation import leaves

        reference = reconstruct_waveforms_flow(
            trigger_folder,
            config,
            ifos,
            copy.deepcopy(event),
            copy.deepcopy(cluster),
            epoch=epoch,
            wave_file=wave_file,
            save=False,
            plot=False,
            queue=None,
        )
        expected = leaves(
            {
                f"{ifo}_wf_{kind}_whiten": reference[f"{ifo}_wf_{kind}_whiten"]
                for ifo in ifos
                for kind in ("REC", "DAT")
            }
        )
        del reference
    data: dict[str, TimeSeries] = {}
    for kind, label in WHITENED_PRODUCTS:
        waves = get_network_MRA_wave(
            config,
            cluster,
            config.rateANA,
            config.nIFO,
            config.TDRate,
            kind,
            0,
            True,
            whiten=True,
        )
        for ifo, wave in zip(ifos, waves, strict=True):
            wave.start_time += epoch
            data[f"{ifo}_wf_{label}_whiten"] = wave
    if expected is not None:
        if leaves(data) != expected:
            raise AssertionError(
                "Q-veto waveforms differ from complete native reconstruction"
            )
        logger.info(
            "GPU Q-veto reconstruction parity: event=%s waveforms=%d exact=1",
            event.hash_id,
            len(data),
        )
    return data


def make_save(context: Any) -> Callable[[Any, Any], None]:
    """Return the native lag save path with an explicit reconstruction callback that reconstructs via :func:`reconstruct`.

    The callback is used by the parent's :class:`~.output_buffer.OutputWriter`
    and writes only what the native save path writes (triggers and progress
    for catalog-only background).

    Parameters
    ----------
    context
        Native ``LagOutputContext``; validated with
        the shared runtime settings validator.

    Returns
    -------
    callable
        ``save(output_context, result)`` with the native signature.

    Raises
    ------
    ValueError
        If the context is not catalog-only background or its runtime settings
        are incompatible with parent Q-veto reconstruction.
    """
    from pycwb.workflow.subflow import job_segment_output, process_job_segment_native

    from pycwb.config.validation import OUTPUT_PRODUCT_FLAGS, validate_runtime_settings
    from functools import partial

    validate_runtime_settings(context.config)
    if context.sub_job_seg.injections or any(getattr(context.config, flag, False) for flag in OUTPUT_PRODUCT_FLAGS):
        raise ValueError("Q-veto reconstruction requires catalog-only background without plots")
    postprocess = partial(
        job_segment_output._postprocess_saved_triggers,
        reconstruct=reconstruct,
    )
    return partial(
        process_job_segment_native._save_lag_outputs,
        postprocess=postprocess,
    )
