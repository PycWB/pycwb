"""Output-finalization helpers for native job-segment processing."""

import gc
import logging
import time

import psutil

from pycwb.modules.qveto.qveto import get_qveto
from pycwb.modules.reconstruction import estimate_snr
from pycwb.modules.workflow_utils import create_single_trigger_folder, save_trigger
from pycwb.types.trigger import Trigger
from pycwb.workflow.subflow.job_segment_progress import _catalog_path, _record_lag_progress
from pycwb.workflow.subflow.job_segment_resources import _free_jax_buffers
from pycwb.workflow.subflow.postprocess_and_plots import (
    plot_skymap_flow,
    plot_trigger_flow,
    reconstruct_injection_waveforms_flow,
    reconstruct_waveforms_flow,
)

logger = logging.getLogger(__name__)


def _create_and_save_trigger_folders(output_context, result) -> list[str | None]:
    config = output_context.config
    trigger_folders = []
    for trigger in result.events_data:
        try:
            trigger_folder = create_single_trigger_folder(
                output_context.working_dir, config.trigger_dir, output_context.sub_job_seg, trigger,
            )
            trigger_folders.append(trigger_folder)
            save_trigger(
                trigger_folder=trigger_folder,
                trigger_data=trigger,
                save_cluster=config.save_cluster,
                save_sky_map=config.save_sky_map,
                save_likelihood_features=getattr(
                    config, "save_likelihood_features", False
                ),
            )
        except Exception:
            event = trigger[0]
            logger.exception(
                "Failed to save trigger folder for event hash_id=%s cluster_id=%s lag=%d - skipping event",
                getattr(event, "hash_id", "?"),
                getattr(event, "cluster_id", "?"),
                result.lag,
            )
            # Keep trigger_folders aligned with events_data by appending None.
            trigger_folders.append(None)

    logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)
    return trigger_folders


def _postprocess_saved_triggers(output_context, result, trigger_folders) -> tuple[float, float, float]:
    config = output_context.config
    sub_job_seg = output_context.sub_job_seg
    reconstruct_elapsed = 0.0
    qveto_elapsed = 0.0
    plot_elapsed = 0.0

    for trigger_folder, trigger in zip(trigger_folders, result.events_data):
        # Skip events whose trigger folder could not be created.
        if trigger_folder is None:
            continue
        event, cluster_out, event_skymap_statistics = trigger

        reconstruct_timer = time.perf_counter()
        reconst_data = reconstruct_waveforms_flow(
            trigger_folder,
            config,
            sub_job_seg.ifos,
            event,
            cluster_out,
            epoch=sub_job_seg.padded_start,
            wave_file=output_context.wave_file,
            save=config.save_waveform,
            plot=config.plot_waveform,
            queue=output_context.queue,
        )
        reconstruct_elapsed += time.perf_counter() - reconstruct_timer

        if event.injection:
            _update_event_from_injection_reconstruction(
                output_context, trigger_folder, event, reconst_data,
            )

        qveto_elapsed += _compute_event_qveto(sub_job_seg.ifos, event, reconst_data)

        plot_timer = time.perf_counter()
        if config.plot_trigger:
            plot_trigger_flow(trigger_folder, event, cluster_out)

        if config.plot_sky_map:
            plot_skymap_flow(trigger_folder, event, event_skymap_statistics)
        plot_elapsed += time.perf_counter() - plot_timer

        del reconst_data

    return reconstruct_elapsed, qveto_elapsed, plot_elapsed


def _update_event_from_injection_reconstruction(output_context, trigger_folder, event, reconst_data) -> None:
    config = output_context.config
    sub_job_seg = output_context.sub_job_seg
    injected_data = reconstruct_injection_waveforms_flow(
        trigger_folder,
        config,
        sub_job_seg.ifos,
        event,
        output_context.unwhitened_injection_strains,
        output_context.whitened_injection_strains,
        config.iwindow / 2,
        config.segEdge,
        config.inRate,
        wave_file=output_context.wave_file,
        save=config.save_injection,
        plot=config.plot_injection,
        queue=output_context.queue,
    )
    event.hrss += injected_data["hrss"]
    event.time += injected_data["central_time"]
    event.iSNR = injected_data["snr"]
    event.frequency += injected_data["central_freq"]
    event.bandwidth += injected_data["bandwidth"]
    event.duration += injected_data["duration"]

    inj_waveforms = injected_data["whitened_injected_waveform"]
    rec_waveforms = [reconst_data[f"{ifo}_wf_REC_whiten"] for ifo in sub_job_seg.ifos]
    event.oSNR = [estimate_snr(rec) for rec in rec_waveforms]
    event.ioSNR = [
        estimate_snr(inj, rec)
        if (inj is not None) and (rec is not None) else None
        for inj, rec in zip(inj_waveforms, rec_waveforms)
    ]
    del injected_data, inj_waveforms, rec_waveforms


def _compute_event_qveto(ifos, event, reconst_data) -> float:
    qveto_timer = time.perf_counter()
    try:
        min_qveto = 1e23
        min_qfactor = 1e23
        for ifo in ifos:
            for a_type in ["DAT", "REC"]:
                [qveto, qfactor] = get_qveto(reconst_data[f"{ifo}_wf_{a_type}_whiten"])
                min_qveto = min(min_qveto, qveto)
                min_qfactor = min(min_qfactor, qfactor)
        event.Qveto = [min_qveto, min_qfactor]
        event.qveto = min_qveto
        event.qfactor = min_qfactor
        logger.info(
            "Qveto for event %s: %s, Qfactor: %s",
            event.hash_id,
            event.qveto,
            event.qfactor,
        )
    except Exception as e:
        logger.error("Error calculating Qveto for event %s: %s", event.hash_id, e)
    finally:
        event_qveto_elapsed = time.perf_counter() - qveto_timer
        logger.info(
            "Qveto/Qfactor computation time for event %s: %.3f s",
            event.hash_id,
            event_qveto_elapsed,
        )
    return event_qveto_elapsed


def _write_trigger_records(output_context, result) -> tuple[float, float]:
    config = output_context.config
    trigger_convert_elapsed = 0.0
    trigger_write_elapsed = 0.0

    for trigger in result.events_data:
        event, _, _ = trigger
        convert_timer = time.perf_counter()
        try:
            trigger_obj = Trigger.from_event(event)
            trigger_obj.time_lag = result.time_lag
            trigger_obj.segment_lag = result.segment_lag
        except Exception:
            logger.exception(
                "Failed to convert event hash_id=%s to Trigger at lag=%d - skipping",
                getattr(event, "hash_id", "?"),
                result.lag,
            )
            continue
        trigger_convert_elapsed += time.perf_counter() - convert_timer

        write_timer = time.perf_counter()
        try:
            if output_context.queue is not None:
                output_context.queue.put({"type": "trigger", "trigger": trigger_obj})
            else:
                catalog_path = _catalog_path(
                    output_context.working_dir,
                    config,
                    output_context.catalog_file,
                )
                if catalog_path:
                    from pycwb.modules.catalog.catalog import Catalog

                    Catalog.open(catalog_path).add_triggers(trigger_obj)
        except Exception:
            logger.exception(
                "Failed to write trigger hash_id=%s to catalog at lag=%d - skipping",
                trigger_obj.hash_id,
                result.lag,
            )
        trigger_write_elapsed += time.perf_counter() - write_timer

    return trigger_convert_elapsed, trigger_write_elapsed


def _record_output_progress(output_context, result) -> float:
    progress_timer = time.perf_counter()
    _record_lag_progress(
        output_context.working_dir,
        output_context.config,
        output_context.catalog_file,
        output_context.queue,
        result.progress_record,
    )
    return time.perf_counter() - progress_timer


def _log_lag_output_timing(
    result,
    reconstruct_elapsed: float,
    qveto_elapsed: float,
    plot_elapsed: float,
    trigger_convert_elapsed: float,
    trigger_write_elapsed: float,
    progress_elapsed: float,
) -> None:
    finalization_elapsed = (
        reconstruct_elapsed
        + qveto_elapsed
        + plot_elapsed
        + trigger_convert_elapsed
        + trigger_write_elapsed
        + progress_elapsed
    )
    if finalization_elapsed >= 0.1:
        logger.info(
            "Lag %d output finalization time: reconstruct_waveforms=%.2f s, "
            "qveto=%.2f s, plot=%.2f s, "
            "trigger_convert=%.2f s, trigger_write=%.2f s, progress=%.2f s",
            result.lag,
            reconstruct_elapsed,
            qveto_elapsed,
            plot_elapsed,
            trigger_convert_elapsed,
            trigger_write_elapsed,
            progress_elapsed,
        )


def _log_lag_completion(result) -> None:
    logger.info("-------------------------------------------")
    logger.info("Lag %d processing time: %.2f s", result.lag, time.perf_counter() - result.lag_timer)
    logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)
    logger.info("-------------------------------------------")


def _cleanup_lag_output_state() -> None:
    gc.collect()
    _free_jax_buffers()
