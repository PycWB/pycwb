"""
Auxiliary workflow: build a comprehensive per-simulation summary table.

For every simulated signal recorded in the job segments this module:

1. Generates the waveform (via ``generate_strain_from_injection``) to measure
   the true GPS extent of the signal across all IFOs (``real_start`` /
   ``real_end``).
2. Identifies which job segment(s) each simulation belongs to.
3. Classifies each simulation against three data-quality veto levels:

   * **CAT0** – simulation falls outside all CWB_CAT0-level good-data intervals.
     Only evaluated when the config contains CWB_CAT0 DQ files; otherwise the
     column is ``None``.
   * **CAT1** – simulation GPS time falls outside all job-segment analysis
     windows.  Job segments are already built from CAT1 good-data intervals, so
     a simulation not covered by any segment is CAT1-vetoed.
   * **CAT2** – simulation GPS time falls inside a job segment but is not
     covered by the segment's ``veto_windows`` (the intersected CAT2 keep
     intervals).  ``None`` when no veto windows are present on any segment.

4. Flags simulations whose waveform extent (``real_start`` → ``real_end``) spans
   two or more job segments (``across_segments``).

The result is saved as a Parquet file with one row per simulation.  Every
simulation parameter is stored verbatim in a ``parameters`` struct column, while
derived quantities and veto flags are stored in dedicated top-level columns for
fast filtering.
"""

import logging
import os
from copy import deepcopy
from typing import List, Optional

import numpy as np
import pandas as pd

from pycwb.config import Config
from pycwb.modules.injection import generate_strain_from_injection
from pycwb.types.job import WaveSegment
from pycwb.types.simulation import (
    infer_injection_fields,
    rows_to_flat_table,
    build_fixed_schema,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gps_in_windows(gps: float, windows: list) -> bool:
    """Return True if *gps* falls within any ``(start, stop)`` pair in *windows*."""
    for start, stop in windows:
        if start <= gps < stop:
            return True
    return False


def _seg_list_to_windows(seg_list) -> list:
    """Convert a ``(starts, stops)`` tuple returned by ``read_seg_list`` to a
    sorted list of ``(start, stop)`` float tuples."""
    starts, stops = seg_list
    return sorted(zip([float(s) for s in starts], [float(e) for e in stops]))


def _find_segment(gps: float, job_segments: List[WaveSegment]) -> Optional[WaveSegment]:
    """Return the first job segment whose analysis window contains *gps*, or
    ``None`` if the GPS time falls outside every segment."""
    for seg in job_segments:
        if seg.analyze_start <= gps < seg.analyze_end:
            return seg
    return None


def _build_cat0_windows(config: Config) -> Optional[list]:
    """Build the list of CWB_CAT0-level good-data windows from *config*.

    Returns ``None`` when no CWB_CAT0 DQ files are present (so the caller can
    propagate ``None`` / NaN for the ``vetoed_cat0`` column).
    """
    if not config.dq_files:
        return None

    cat0_files = [dqf for dqf in config.dq_files if dqf.dq_cat == 'CWB_CAT0']
    if not cat0_files:
        return None

    from pycwb.modules.job_segment.dq_segment import read_seg_list, merge_seg_list

    # Intersect CAT0 good-data intervals across all IFOs
    seg_lists = []
    for ifo in config.ifo:
        ifo_files = [dqf for dqf in cat0_files if dqf.ifo == ifo]
        if not ifo_files:
            # This IFO has no CAT0 file; treat as always good for that IFO
            continue
        seg_lists.append(read_seg_list(ifo_files, 'CWB_CAT0'))

    if not seg_lists:
        return None

    merged = seg_lists[0]
    for sl in seg_lists[1:]:
        merged = merge_seg_list(merged, sl)

    return _seg_list_to_windows(merged)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_simulation_summary(
    config: Config,
    job_segments: List[WaveSegment],
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """Build a comprehensive per-simulation summary table and optionally save it
    to a Parquet file.

    Parameters
    ----------
    config : Config
        The run configuration.  Used for IFO list, sample rate, simulation
        block defaults, and DQ files.
    job_segments : list[WaveSegment]
        The complete list of job segments produced by
        ``create_job_segment_from_config``.  Each segment's ``injections``
        attribute and ``veto_windows`` (CAT2) are read directly.
    output_file : str, optional
        Destination path for the Parquet output.  The parent directory is
        created automatically.  Skipped when ``None``.

    Returns
    -------
    pd.DataFrame
        One row per simulation.  Schema:

        ==================  ============  ============================================
        Column              dtype         Description
        ==================  ============  ============================================
        ``sim_idx``         int           Canonical generated-simulation id
        ``trial_idx``       int           Trial index (0 = first/only trial)
        ``gps_time``        float         Requested coalescence GPS time
        ``name``            str           Injection waveform name
        ``approximant``     str           Waveform approximant (e.g. IMRPhenomXPHM)
        ``ra``              float         Right ascension (radians)
        ``dec``             float         Declination (radians)
        ``pol``             float         Polarisation angle (radians)
        ``hrss``            float         Strain amplitude (h_rss)
        ``target_snr``      float         Target network SNR (0 if not set)
        ``real_start``      float         Earliest waveform sample across all IFOs
        ``real_end``        float         Latest waveform sample across all IFOs
        ``real_duration``   float         ``real_end - real_start``
        ``job_id``          int           ``WaveSegment.index`` of the job that
                                          owns this scheduled simulation
        ``shift``           list[float]   Superlag shift vector of the owning job
        ``segment_idx``     int           ``WaveSegment.index`` of the containing
                                          segment, or ``-1`` if none.  For new
                                          summaries this is kept as a
                                          backward-compatible alias of ``job_id``.
        ``vetoed_cat0``     bool/None     GPS not in any CAT0 good interval
                                          (``None`` = no CAT0 DQ files present)
        ``vetoed_cat1``     bool          GPS outside every job-segment window
        ``vetoed_cat2``     bool/None     GPS in job segment but outside CAT2
                                          ``veto_windows`` (``None`` = not set on
                                          any segment)
        ``across_segments`` bool          Waveform spans ≥2 job segments
        ``error``           str/None      Exception message if waveform generation
                                          failed; other columns may be ``None``
        ``parameters``      dict          All raw simulation parameters (verbatim copy)
        ==================  ============  ============================================
    """
    # ── Pre-compute veto structures that are common across all simulations ──
    cat0_windows = _build_cat0_windows(config)
    has_cat0 = cat0_windows is not None

    # Check whether any segment carries CAT2 veto windows
    has_cat2 = any(seg.veto_windows for seg in job_segments)

    # Build a flat set of (start, stop) tuples for all job-segment analysis
    # windows, for fast CAT1 lookup
    seg_analysis_windows = [
        (seg.analyze_start, seg.analyze_end) for seg in job_segments
    ]

    # ── Collect every simulation from all segments (de-duplicate by identity) ──
    # We use the simulation dicts from the segments.  If the same dict
    # appears in multiple segments (e.g. after add_injections_into_job_segments),
    # we process each occurrence independently because the veto status may differ
    # per segment.
    all_simulations = []  # list of (simulation_dict, owning_segment)
    seen_ids = set()
    for seg in job_segments:
        for sim in (seg.injections or []):
            uid = (id(sim), seg.index)
            if uid not in seen_ids:
                seen_ids.add(uid)
                all_simulations.append((deepcopy(sim), seg))

    if not all_simulations:
        logger.warning("No simulations found in any job segment — returning empty summary.")
        return pd.DataFrame()

    logger.info("Building simulation summary for %d simulation(s) across %d segment(s)",
                len(all_simulations), len(job_segments))

    # ── Process each simulation ──
    # Suppress verbose INFO logs from waveform generation sub-modules while the
    # progress bar is active; they would flood the terminal for every injection.
    _noisy_loggers = [
        logging.getLogger("pycwb.modules.injection.strain"),
        logging.getLogger("pycwb.utils.module"),
    ]
    _saved_levels = [lg.level for lg in _noisy_loggers]
    for lg in _noisy_loggers:
        lg.setLevel(logging.WARNING)

    from tqdm import tqdm
    rows = []
    for sim_idx, (simulation, owning_seg) in tqdm(
        enumerate(all_simulations), total=len(all_simulations),
        desc="Building simulation summary", unit="sim",
    ):
        sim_idx_value = int(simulation.get('sim_idx', sim_idx))
        gps_time = float(simulation.get('gps_time', float('nan')))
        trial_idx = int(simulation.get('trial_idx', 0))

        row = {
            'sim_idx':   sim_idx_value,
            'trial_idx': trial_idx,
            'gps_time':  gps_time,
            'name':        str(simulation.get('name', '')),
            'approximant': str(simulation.get('approximant', '')),
            'ra':          float(simulation.get('ra', float('nan'))),
            'dec':         float(simulation.get('dec', float('nan'))),
            'pol':         float(simulation.get('pol', float('nan'))),
            'hrss':        float(simulation.get('hrss', float('nan'))),
            'target_snr':  float(simulation.get('target_snr',
                                simulation.get('targeted_snr', float('nan')))),
            # to be filled after waveform generation
            'real_start':      None,
            'real_end':        None,
            'real_duration':   None,
            'job_id':          int(simulation.get('job_id', owning_seg.index)),
            'shift':           list(simulation.get(
                'shift',
                owning_seg.shift if owning_seg.shift is not None else [0.0 for _ in owning_seg.ifos],
            )),
            'segment_idx':     int(simulation.get('job_id', owning_seg.index)),
            'vetoed_cat0':     None,
            'vetoed_cat1':     None,
            'vetoed_cat2':     None,
            'across_segments': None,
            'error':           None,
            # verbatim copy of raw injection dict — expanded to flat columns
            # when saving (see rows_to_flat_table below)
            'parameters':      {k: v for k, v in simulation.items()
                                if k not in ('real_start', 'real_end', 'job_id',
                                             'shift', 'sim_idx')},
        }

        # ── Step 1 : generate waveform → real_start / real_end ────────────
        try:
            # Use the original input sample rate for waveform generation
            sample_rate = config.inRate
            sim_strains = generate_strain_from_injection(
                simulation, config, sample_rate, config.ifo
            )
            n_ifo = len(config.ifo)
            real_start = min(float(sim_strains[i].t0) for i in range(n_ifo))
            real_end = max(
                float(sim_strains[i].t0) + len(sim_strains[i].data) * float(sim_strains[i].dt)
                for i in range(n_ifo)
            )
            del sim_strains  # free memory immediately
        except Exception as exc:
            logger.error("Waveform generation failed for simulation %d (gps=%.3f): %s",
                         sim_idx_value, gps_time, exc)
            row['error'] = str(exc)
            rows.append(row)
            continue

        row['real_start'] = real_start
        row['real_end'] = real_end
        row['real_duration'] = real_end - real_start

        # ── Step 2 : identify containing job segment ───────────────────────
        containing_seg = _find_segment(gps_time, job_segments)

        # ── Step 3a : CAT0 veto ────────────────────────────────────────────
        if has_cat0:
            row['vetoed_cat0'] = not _gps_in_windows(gps_time, cat0_windows)
        # else remains None

        # ── Step 3b : CAT1 veto (= not in any job segment) ────────────────
        row['vetoed_cat1'] = not _gps_in_windows(gps_time, seg_analysis_windows)

        # ── Step 3c : CAT2 veto ────────────────────────────────────────────
        if has_cat2:
            if not owning_seg.veto_windows:
                # Owning segment carries no CAT2 windows → not vetoed
                row['vetoed_cat2'] = False
            else:
                row['vetoed_cat2'] = not _gps_in_windows(
                    gps_time, owning_seg.veto_windows
                )
        # else remains None

        # ── Step 4 : check if waveform spans multiple segments ─────────────
        start_seg = _find_segment(real_start, job_segments)
        end_seg = _find_segment(real_end, job_segments)
        if start_seg is None and end_seg is None:
            # Entirely outside all segments
            row['across_segments'] = False
        elif start_seg is None or end_seg is None:
            # One end inside, other outside → effectively across a boundary
            row['across_segments'] = True
        else:
            row['across_segments'] = start_seg.index != end_seg.index

        rows.append(row)
        logger.debug(
            "sim %d  gps=%.3f  seg=%s  vetoed(0=%s,1=%s,2=%s)  across=%s",
            sim_idx_value, gps_time, row['segment_idx'],
            row['vetoed_cat0'], row['vetoed_cat1'], row['vetoed_cat2'],
            row['across_segments'],
        )

    # Restore log levels suppressed during the loop
    for lg, lvl in zip(_noisy_loggers, _saved_levels):
        lg.setLevel(lvl)

    # ── Build flat Arrow table ─────────────────────────────────────────────
    # Collect all raw injection dicts to infer the waveform-specific schema
    all_injection_dicts = [row['parameters'] for row in rows if row.get('parameters')]
    extra_fields = infer_injection_fields(all_injection_dicts)
    logger.debug(
        "Inferred %d extra injection fields: %s",
        len(extra_fields), list(extra_fields.keys()),
    )

    table = rows_to_flat_table(rows, extra_fields, fixed_schema=build_fixed_schema())

    # ── Save to Parquet ────────────────────────────────────────────────────
    if output_file is not None:
        import pyarrow.parquet as pq
        parent = os.path.dirname(output_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        pq.write_table(
            table, output_file,
            compression='snappy',
            write_statistics=True,
            store_schema=True,
        )
        logger.info("Simulation summary saved to %s  (%d rows, %d columns)",
                    output_file, len(table), len(table.schema))

    # Return as pandas DataFrame for convenience
    df = table.to_pandas()
    return df
