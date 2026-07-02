"""Lag-independent setup for native coherence."""

from __future__ import annotations

import logging
import os
import time

from wdm_wavelet.wdm import WDM as WDMWavelet

from pycwb.config import Config
from pycwb.types.job import WaveSegment
from pycwb.types.time_frequency_map import TimeFrequencyMap
from pycwb.types.time_series import TimeSeries

from .projection import _max_energy_backend, _max_energy_backend_label, max_energy
from .selection import _build_selection_cache
from .tf_batch_generation import batch_t2w_detectors
from .veto_threshold import compute_threshold

logger = logging.getLogger(__name__)


def _coherence_timing_enabled(config: Config) -> bool:
    """Return True when detailed coherence setup timing logs are requested."""
    flag = str(os.getenv("PYCWB_COHERENCE_TIMING", "")).strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    return bool(getattr(config, "coherence_timing", False))


def setup_coherence(
    config: Config, strains: list[TimeSeries], job_seg: WaveSegment | None = None
) -> list[dict]:
    """
    Compute all lag-independent coherence data (TF maps after max_energy,
    threshold, lag plan) for every resolution level.

    Call this once per job segment, then pass the returned list to
    :func:`coherence_single_lag` for each lag.

    Parameters
    ----------
    config : Config
        Configuration object.
    strains : list[TimeSeries]
        Whitened strain time series.
    job_seg : WaveSegment or None, optional
        Job segment (provides lag count via ``job_seg.n_lag``).

    Returns
    -------
    list[dict]
        One setup dict per resolution, keyed by ``tf_maps``, ``Eo``,
        ``job_seg``, ``pattern``, ``level``, ``layers``, ``rate``,
        ``select_subrho``, ``select_subnet``, ``segEdge``.
    """
    # Compute upsample factor for max_energy (minimum 1)
    up_n = int(config.rateANA / 1024)
    if up_n < 1:
        up_n = 1

    # Normalize input strains to pycwb TimeSeries objects
    normalized_strains = [TimeSeries.from_input(strain) for strain in strains]

    # Build setups for each resolution level independently
    # (expensive WDM transforms, TF maps, and thresholds are computed once here,
    #  then reused across all lags in coherence_single_lag)
    setups = [
        _setup_coherence_single_res(
            i, config, normalized_strains, up_n, job_seg=job_seg
        )
        for i in range(config.nRES)
    ]

    return setups


def _setup_coherence_single_res(
    i: int,
    config: Config,
    strains: list[TimeSeries],
    up_n: int,
    job_seg: WaveSegment | None = None,
) -> dict:
    """
    Lag-independent coherence setup for one resolution level.

    Builds the WDM wavelet, TF maps, applies max_energy, computes the
    energy threshold, and builds the lag plan.  Nothing here depends on
    which lag is being processed.

    Returns
    -------
    dict
        Keys: ``tf_maps``, ``Eo``, ``job_seg``, ``pattern``, ``level``,
        ``layers``, ``rate``, ``select_subrho``, ``select_subnet``,
        ``segEdge``, ``selection_cache``.
    """
    timer_start = time.perf_counter()
    timing_enabled = _coherence_timing_enabled(config)
    level = config.l_high - i
    layers = 2**level if level > 0 else 0
    rate = config.rateANA // 2**level
    max_energy_backend = _max_energy_backend(config, layers=layers)
    max_energy_backend_log = _max_energy_backend_label(max_energy_backend)

    t_stage = time.perf_counter()
    # Ensure at least one WDM layer for zero-lag case
    wdm_layers = max(1, layers)
    wdm_wavelet = WDMWavelet(
        M=wdm_layers,
        K=wdm_layers,
        beta_order=config.WDM_beta_order,
        precision=config.WDM_precision,
    )
    t_wdm = time.perf_counter() - t_stage

    # Build time-frequency maps via batch WDM transform (preferring fast path)
    t_stage = time.perf_counter()
    try:
        batch_data_list, (dt, df) = batch_t2w_detectors(strains, wdm_wavelet)
        tf_maps = [
            TimeFrequencyMap(
                data=batch_data_list[n],
                is_whitened=True,
                dt=dt,
                df=df,
                start=float(strains[n].t0),
                stop=float(strains[n].end_time),
                f_low=getattr(config, "fLow", None),
                f_high=getattr(config, "fHigh", None),
                edge=getattr(config, "segEdge", None),
                wavelet=wdm_wavelet,
                len_timeseries=len(strains[n].data),
            )
            for n in range(len(strains))
        ]
        t_tf_maps = time.perf_counter() - t_stage
    except (
        Exception
    ) as exc:  # broad catch intentional: batch_t2w_detectors may raise any of
        # TypeError / ValueError / AttributeError / RuntimeError / numpy internals depending on
        # the WDM implementation version; we always want the serial fallback to succeed.
        logger.warning(
            "Batch t2w failed (%s); falling back to serial from_timeseries", exc
        )
        t_stage = time.perf_counter()
        tf_maps = [
            TimeFrequencyMap.from_timeseries(
                ts=strain,
                wavelet=wdm_wavelet,
                is_whitened=True,
                f_low=getattr(config, "fLow", None),
                f_high=getattr(config, "fHigh", None),
                edge=getattr(config, "segEdge", None),
            )
            for strain in strains
        ]
        t_tf_maps = time.perf_counter() - t_stage

    logger.info(
        "level : %d\t rate(hz) : %d\t layers : %d\t df(hz) : %f\t dt(ms) : %f",
        level,
        rate,
        layers,
        config.rateANA / 2.0 / (2**level),
        1000.0 / rate,
    )

    # Apply max_energy skymap projection to decorrelate across lags
    # (computes the optimal coherent energy over sky positions)
    max_delay = config.max_delay
    pattern = config.pattern
    alp = 0.0
    t_max_energy_total = 0.0
    for n, tf_map in enumerate(tf_maps):
        t_stage = time.perf_counter()
        tf_maps[n], alp_n = max_energy(
            tf_map=tf_map,
            max_delay=max_delay,
            up_n=up_n,
            pattern=pattern,
            f_low=config.fLow,
            f_high=config.fHigh,
            backend=max_energy_backend,
        )
        t_ifo = time.perf_counter() - t_stage
        t_max_energy_total += t_ifo
        if timing_enabled:
            logger.info(
                "level %d max_energy backend=%s ifo=%d done in %.3f s (alp=%.6g)",
                level,
                max_energy_backend_log,
                n,
                t_ifo,
                alp_n,
            )
        alp += alp_n
    # Average the Gamma-to-Gauss scaling factor across detectors
    alp = alp / config.nIFO

    # Compute pixel energy threshold based on black-pixel probability
    # (independent of lag since TF maps are lag-independent)
    t_stage = time.perf_counter()
    Eo = compute_threshold(
        tf_maps,
        config.bpp,
        alp=alp if pattern != 0 else None,
        edge=config.segEdge,
    )
    t_threshold = time.perf_counter() - t_stage

    t_stage = time.perf_counter()
    selection_cache = _build_selection_cache(
        tf_maps,
        edge=config.segEdge,
        lag_shifts_by_lag=getattr(job_seg, "lag_shifts", None),
    )
    t_selection_cache = time.perf_counter() - t_stage

    # Extract lag count from job segment for setup dictionary
    n_lag = job_seg.n_lag

    if timing_enabled:
        logger.info(
            "level %d setup timing: wdm=%.3fs tf_maps=%.3fs max_energy=%.3fs "
            "threshold=%.3fs selection_cache=%.3fs total=%.3fs backend=%s",
            level,
            t_wdm,
            t_tf_maps,
            t_max_energy_total,
            t_threshold,
            t_selection_cache,
            time.perf_counter() - timer_start,
            max_energy_backend_log,
        )

    logger.info(
        "level %d setup done: Eo=%.4g, n_lag=%d  (%.2f s)",
        level,
        Eo,
        n_lag,
        time.perf_counter() - timer_start,
    )

    return {
        "tf_maps": tf_maps,
        "Eo": Eo,
        "job_seg": job_seg,
        "pattern": pattern,
        "level": level,
        "layers": layers,
        "rate": rate,
        "select_subrho": config.select_subrho,
        "select_subnet": config.select_subnet,
        "segEdge": config.segEdge,
        "selection_cache": selection_cache,
        "max_energy_backend": max_energy_backend,
    }
