"""Veto-window and livetime helpers for native job-segment processing."""

from pycwb.config import Config
from pycwb.modules.job_segment import build_injection_veto_windows, intersect_intervals
from pycwb.types.job import WaveSegment


def _effective_veto_windows(config: Config, sub_job_seg: WaveSegment) -> list[tuple[float, float]] | None:
    """Return the CAT2 veto windows for the segment.

    When ``analyze_injection_only`` is enabled, injection windows are NOT
    folded in here. They are applied later inside the lag loop, after the
    ``segTHR`` check, so that the injection time window cannot cause the
    analysis to be skipped.
    """
    return (
        sub_job_seg.cwb_veto_windows
        if getattr(sub_job_seg, "cwb_veto_windows", None) is not None
        else sub_job_seg.veto_windows
    )


def _injection_aware_veto_windows(
    config: Config,
    sub_job_seg: WaveSegment,
    veto_windows: list[tuple[float, float]] | None,
) -> list[tuple[float, float]] | None:
    """Return veto windows intersected with injection envelopes when applicable.

    Called after the ``segTHR`` check so that small injection windows do not
    cause the lag to be skipped.
    """
    if not (getattr(config, "analyze_injection_only", False) and sub_job_seg.injections):
        return veto_windows

    injection_envelopes = [(inj["real_start"], inj["real_end"]) for inj in sub_job_seg.injections]
    inj_windows = build_injection_veto_windows(
        injection_envelopes,
        padding=getattr(config, "injection_padding", 1.0),
        duration=sub_job_seg.duration,
    )
    if veto_windows is not None:
        return intersect_intervals(sorted(veto_windows), sorted(inj_windows))
    return inj_windows


def _lag_livetime(context, lag: int) -> float:
    sub_job_seg = context.sub_job_seg
    if hasattr(sub_job_seg, "circular_livetime"):
        return sub_job_seg.circular_livetime(lag, context.veto_windows)
    return sub_job_seg.livetime(lag)
