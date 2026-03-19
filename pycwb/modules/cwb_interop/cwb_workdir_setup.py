"""
Create a standalone cWB (ROOT) working directory for each sub_job_seg so that
the pycwb analysis can be compared directly with an equivalent cWB run.

The module is intended to be called **before** ``check_and_resample_py`` so
that the time-series data still lives at the original input sample rate
(``config.inRate``) and has not yet been DC-corrected, down-sampled, or
rescaled by pycwb.

Directory layout produced for each sub_job_seg::

    {cwb_compare_dir}/
    └── job{index}_trail{trail_idx}/          ← cWB working directory
        ├── config/
        │   └── user_parameters.C              ← equivalent cWB configuration
        ├── run.sh                             ← runs cwb_inet 0 1 true
        ├── wdmXTalk -> <filter_dir>/<top>     ← symlink so HOME_WAT_FILTERS
        │                                         can point to this dir
        └── input/
            ├── frames/
            │   ├── L1-PYCWB_DATA-{GPS}-{dur}.gwf
            │   └── H1-PYCWB_DATA-{GPS}-{dur}.gwf
            ├── L1_frames.in                   ← frame-file list for cWB
            ├── H1_frames.in
            ├── L1_cat0.txt                    ← full segment (valid times)
            ├── H1_cat0.txt
            ├── L1_cat1.txt                    ← empty (no bad times known)
            ├── H1_cat1.txt
            ├── L1_cat2.txt
            ├── H1_cat2.txt
            └── segment.period                 ← analysis period

DQ semantics used
-----------------
* **cat0** (``inverse=false``): lists *good* time intervals → the full IFO
  window is listed so cWB uses all saved data.
* **cat1 / cat2** (``inverse=true``): lists *bad* time intervals → files are
  empty, meaning nothing is flagged as bad.
* **segment.period** (``CWB_CAT0``, ``inverse=false``): single interval
  spanning the union of all IFO windows, used by cWB to define the analysis
  chunk.

wdmXTalk resolution
-------------------
cWB builds the full xtalk path as ``$HOME_WAT_FILTERS / wdmXTalk``.  This
module creates a symlink ``<cwb_workdir>/<xtalk_top_dir> → config.filter_dir /
<xtalk_top_dir>`` so that when ``HOME_WAT_FILTERS=<cwb_workdir>`` the relative
``wdmXTalk`` path resolves correctly without modifying the user environment.

The generated ``run.sh`` sets ``HOME_WAT_FILTERS`` automatically.
"""

import logging
import os
import pathlib
import textwrap

import numpy as np

logger = logging.getLogger(__name__)

# Channel suffix written into the GWF files and referenced in user_parameters.C
_EXPORT_CHANNEL = "PYCWB_DATA"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def create_cwb_workdir(
    working_dir: str,
    config,
    sub_job_seg,
    data: list,
    cwb_compare_dir: str = None,
) -> str:
    """Create a standalone cWB ROOT working directory for *sub_job_seg*.

    Parameters
    ----------
    working_dir:
        The pycwb working directory (parent of ``cwb_compare_dir``).
    config:
        The pycwb :class:`~pycwb.config.Config` object.
    sub_job_seg:
        The :class:`~pycwb.types.job.WaveSegment` being processed
        (``trail_idx`` must already be set).
    data:
        List of ``pycwb.types.time_series.TimeSeries`` (one per IFO) at ``config.inRate``
        Hz, *before* ``check_and_resample_py`` has been applied.
    cwb_compare_dir:
        Base directory for all cWB comparison working directories.
        Defaults to ``{working_dir}/cwb_compare``.

    Returns
    -------
    str
        Absolute path to the created cWB working directory.
    """
    if cwb_compare_dir is None:
        cwb_compare_dir = os.path.join(working_dir, "cwb_compare")

    cwb_workdir = os.path.abspath(
        os.path.join(
            cwb_compare_dir,
            f"job{sub_job_seg.index}_trail{sub_job_seg.trail_idx}",
        )
    )
    config_dir = os.path.join(cwb_workdir, "config")
    input_dir = os.path.join(cwb_workdir, "input")
    frames_dir = os.path.join(input_dir, "frames")
    tmp_dir = os.path.join(cwb_workdir, "tmp")
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    logger.info("Creating cWB compare workdir: %s", cwb_workdir)

    # 1. Write GWF files (one per IFO)
    physical_starts = sub_job_seg.physical_analyze_starts
    physical_ends = sub_job_seg.physical_analyze_ends

    frame_paths = {}   # {ifo: absolute_path_to_gwf}
    for i, ifo in enumerate(sub_job_seg.ifos):
        gps_start = physical_starts[ifo]
        gps_end = physical_ends[ifo]
        duration = int(round(gps_end - gps_start))

        gwf_name = f"{ifo}-PYCWB_DATA-{int(gps_start)}-{duration}.gwf"
        gwf_path = os.path.join(frames_dir, gwf_name)

        _save_timeseries_to_gwf(data[i], ifo, _EXPORT_CHANNEL, gwf_path)
        frame_paths[ifo] = gwf_path
        logger.info("  Saved GWF for %s → %s", ifo, gwf_path)

    # 2. Frame-file lists (.in files)
    _create_frame_lists(sub_job_seg, input_dir, frame_paths)

    # 3. DQ files — pass config so we know which categories are actually used
    has_cat1, has_cat2 = _create_dq_files(sub_job_seg, input_dir, config)

    # 4. wdmXTalk symlink
    _symlink_xtalk(config, cwb_workdir)

    # 5. user_parameters.C  (placed under config/ to mirror pycwb layout)
    _write_user_parameters_c(config, sub_job_seg, config_dir, has_cat1, has_cat2)

    # 6. run.sh helper
    _write_run_sh(cwb_workdir)

    logger.info("cWB compare workdir ready: %s", cwb_workdir)
    return cwb_workdir


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _save_timeseries_to_gwf(ts, ifo: str, channel_suffix: str, gwf_path: str) -> None:
    """Write a pycwb TimeSeries to a GWF file via gwpy."""
    from gwpy.timeseries import TimeSeries as GWpyTimeSeries

    channel_name = f"{ifo}:{channel_suffix}"
    # Accept both pycbc.TimeSeries (has .data, .start_time, .delta_t) and
    # pycwb TimeSeries (same interface).
    strain = GWpyTimeSeries(
        data=np.array(ts.data, dtype=np.float64),
        t0=float(ts.start_time),
        dt=float(ts.delta_t),
        channel=channel_name,
        name=channel_name,
    )
    strain.write(gwf_path)


def _create_frame_lists(sub_job_seg, input_dir: str, frame_paths: dict) -> None:
    """Write one ``{IFO}_frames.in`` file per IFO containing the absolute GWF path."""
    for ifo in sub_job_seg.ifos:
        list_path = os.path.join(input_dir, f"{ifo}_frames.in")
        with open(list_path, "w") as fh:
            fh.write(os.path.abspath(frame_paths[ifo]) + "\n")


def _create_dq_files(sub_job_seg, input_dir: str, config) -> tuple:
    """Create minimal DQ files that declare the full segment as valid.

    * ``{IFO}_cat0.txt`` – GPS start / GPS end of that IFO's window (always).
    * ``{IFO}_cat1.txt`` – empty bad-time list, only if pycwb config has CAT1.
    * ``{IFO}_cat2.txt`` – empty bad-time list, only if pycwb config has CAT2.
    * ``segment.period`` – union interval covering all IFOs.

    Returns
    -------
    (has_cat1, has_cat2) : (bool, bool)
        Whether CAT1 / CAT2 files were found in the pycwb config and written.
    """
    physical_starts = sub_job_seg.physical_analyze_starts
    physical_ends = sub_job_seg.physical_analyze_ends

    # Overall analysis period: union of all IFO windows
    period_start = int(min(physical_starts.values()))
    period_end = int(max(physical_ends.values()))

    # Determine which DQ categories are configured in pycwb
    dq_files = getattr(config, "dq_files", []) or []
    config_cats = {dqf.dq_cat for dqf in dq_files}
    has_cat1 = "CWB_CAT1" in config_cats
    has_cat2 = "CWB_CAT2" in config_cats

    for ifo in sub_job_seg.ifos:
        ifo_start = int(physical_starts[ifo])
        ifo_end = int(physical_ends[ifo])

        # cat0: good-time interval (inverse=false → list valid stretches)
        with open(os.path.join(input_dir, f"{ifo}_cat0.txt"), "w") as fh:
            fh.write(f"{ifo_start} {ifo_end}\n")

        # cat1 / cat2: only write if the pycwb config actually uses them
        if has_cat1:
            with open(os.path.join(input_dir, f"{ifo}_cat1.txt"), "w") as fh:
                pass  # empty = no bad times flagged
        if has_cat2:
            with open(os.path.join(input_dir, f"{ifo}_cat2.txt"), "w") as fh:
                pass  # empty = no bad times flagged

    # Period file (shared by all IFOs)
    with open(os.path.join(input_dir, "segment.period"), "w") as fh:
        fh.write(f"{period_start} {period_end}\n")

    return has_cat1, has_cat2


def _symlink_xtalk(config, cwb_workdir: str) -> None:
    """Symlink the top-level xtalk directory into the cWB workdir.

    cWB constructs the catalog path as ``$HOME_WAT_FILTERS / wdmXTalk``.
    If ``wdmXTalk = "wdmXTalk/OverlapCatalog.bin"`` and the symlink
    ``cwb_workdir/wdmXTalk → config.filter_dir/wdmXTalk`` is created,
    running cWB with ``HOME_WAT_FILTERS=cwb_workdir`` resolves correctly.

    If ``config.wdmXTalk`` is an absolute path (no top-level component) or
    the source directory does not exist, a warning is emitted and the caller's
    run.sh will contain a reminder to set ``HOME_WAT_FILTERS`` manually.
    """
    xtalk_rel = pathlib.PurePosixPath(config.wdmXTalk)
    if not xtalk_rel.parts:
        logger.warning("config.wdmXTalk is empty; skipping wdmXTalk symlink.")
        return

    xtalk_top = xtalk_rel.parts[0]
    xtalk_src = os.path.join(config.filter_dir, xtalk_top)
    xtalk_dst = os.path.join(cwb_workdir, xtalk_top)

    if os.path.exists(xtalk_dst):
        return  # already present (e.g. re-run)

    if os.path.exists(xtalk_src):
        os.symlink(os.path.abspath(xtalk_src), xtalk_dst)
    else:
        logger.warning(
            "wdmXTalk source '%s' not found. Set HOME_WAT_FILTERS manually before running cWB.",
            xtalk_src,
        )


# ---------------------------------------------------------------------------
# user_parameters.C generator
# ---------------------------------------------------------------------------

def _write_user_parameters_c(config, sub_job_seg, config_dir: str,
                              has_cat1: bool, has_cat2: bool) -> None:
    """Generate ``user_parameters.C`` inside *config_dir* (the ``config/`` subdirectory)."""

    nIFO = len(sub_job_seg.ifos)

    # ---- segment timing ------------------------------------------------
    physical_starts = sub_job_seg.physical_analyze_starts
    physical_ends = sub_job_seg.physical_analyze_ends
    seg_start = int(min(physical_starts.values()))
    seg_end = int(max(physical_ends.values()))

    seg_duration = seg_end - seg_start
    seg_len = seg_duration
    seg_edge = float(getattr(config, "segEdge", 8.0))
    seg_mls = max(int(seg_duration - 2 * seg_edge), int(seg_duration // 2))
    seg_thr = 10

    # ---- helpers -------------------------------------------------------
    def _cfg(attr, default):
        val = getattr(config, attr, None)
        return default if val is None else val

    def _boolstr(v):
        return "true" if v else "false"

    cfg_search  = _cfg("cfg_search", "r")
    optim       = _cfg("optim", False)
    ref_ifo     = _cfg("refIFO", sub_job_seg.ifos[0])
    lag_step    = _cfg("lagStep", 1.0)
    f_low       = _cfg("fLow", 64.0)
    f_high      = _cfg("fHigh", 2048.0)
    level_r     = _cfg("levelR", 2)
    l_low       = _cfg("l_low", 3)
    l_high      = _cfg("l_high", 8)
    wdm_xtalk   = _cfg("wdmXTalk", "")
    healpix     = _cfg("healpix", 7)
    bpp         = _cfg("bpp", 0.001)
    subnet      = _cfg("subnet", 0.7)
    subcut      = _cfg("subcut", 0.33)
    net_rho     = _cfg("netRHO", 4.0)
    net_cc      = _cfg("netCC", 0.5)
    a_core      = _cfg("Acore", 1.4142135623730951)
    t_gap       = _cfg("Tgap", 3.0)
    f_gap       = _cfg("Fgap", 130.0)
    delta       = _cfg("delta", 0.5)
    cfg_gamma   = _cfg("cfg_gamma", 0.5)
    loud        = _cfg("LOUD", 200)
    pattern     = _cfg("pattern", 0)
    precision   = _cfg("precision", 0.0)
    in_rate     = _cfg("inRate", 16384)
    f_resample  = _cfg("fResample", 0)
    up_tdf      = _cfg("upTDF", 4)
    td_size     = _cfg("TDSize", 12)
    white_win   = _cfg("whiteWindow", 60.0)
    white_str   = _cfg("whiteStride", 20.0)
    dc_cal      = _cfg("dcCal", [1.0] * nIFO)
    efec        = _cfg("EFEC", True)

    f_resample_line = (
        f"fResample = {int(f_resample)};"
        if f_resample and int(f_resample) > 0
        else "// fResample = 0;  // no resampling before levelR reduction"
    )

    # Pre-compute multi-statement blocks (can't use backslash in f-string expr)
    ifo_stmts     = [f'strcpy(ifo[{i}], "{ifo}");'
                     for i, ifo in enumerate(sub_job_seg.ifos)]
    channel_stmts = [f'sprintf(channelNamesRaw[{i}], "{ifo}:{_EXPORT_CHANNEL}");'
                     for i, ifo in enumerate(sub_job_seg.ifos)]
    frfiles_stmts = [f'sprintf(frFiles[{i}], "input/{ifo}_frames.in");'
                     for i, ifo in enumerate(sub_job_seg.ifos)]
    dc_cal_stmts  = [f"dcCal[{i}] = {float(dc_cal[i] if i < len(dc_cal) else 1.0)};"
                     for i in range(nIFO)]

    # ---- DQ file entries -----------------------------------------------
    # Only include categories that are actually used in the pycwb config.
    dqf_rows = []
    for ifo in sub_job_seg.ifos:
        dqf_rows.append(f'  {{"{ifo}", "input/{ifo}_cat0.txt", CWB_CAT0, 0., false, false}}')
    if has_cat1:
        for ifo in sub_job_seg.ifos:
            dqf_rows.append(f'  {{"{ifo}", "input/{ifo}_cat1.txt", CWB_CAT1, 0., true,  false}}')
    if has_cat2:
        for ifo in sub_job_seg.ifos:
            dqf_rows.append(f'  {{"{ifo}", "input/{ifo}_cat2.txt", CWB_CAT2, 0., true,  false}}')
    for ifo in sub_job_seg.ifos:
        dqf_rows.append(f'  {{"{ifo}", "input/segment.period", CWB_CAT0, 0., false, false}}')

    n_dqf = len(dqf_rows)
    dqf_block = ",\n".join(dqf_rows)

    # ---- assemble the macro body as a list of lines -------------------
    # Written without textwrap.dedent so embedded multi-line blocks keep
    # their indentation regardless of function scope indentation.
    I = "  "  # 2-space indent for the C macro body

    def _join(*stmts):
        """Join C statements at the same indent level."""
        return ("\n" + I).join(stmts)

    lines = [
        "{",
        f"{I}// ----------------------------------------------------------------",
        f"{I}// user_parameters.C — generated by pycwb/modules/cwb_interop",
        f"{I}// pycwb job_id : {sub_job_seg.index}",
        f"{I}// trail_idx    : {sub_job_seg.trail_idx}",
        f"{I}// GPS window   : {seg_start} – {seg_end}",
        f"{I}// IFOs         : {', '.join(sub_job_seg.ifos)}",
        f"{I}//",
        f"{I}// Lag/superlag forced to zero – compare against pycwb lag-0.",
        f"{I}// Run: cd <workdir> && bash run.sh",
        f"{I}// ----------------------------------------------------------------",
        "",
        f"{I}strcpy(analysis, \"2G\");",
        "",
        f"{I}nIFO = {nIFO};",
        f"{I}cfg_search = '{cfg_search}';",
        f"{I}optim = {_boolstr(optim)};",
        "",
        f"{I}{_join(*ifo_stmts)}",
        f"{I}strcpy(refIFO, \"{ref_ifo}\");",
        "",
        f"{I}// --- lags: zero-lag only for direct comparison -----------------",
        f"{I}lagSize = 1;",
        f"{I}lagStep = {lag_step};",
        f"{I}lagOff  = 0;",
        f"{I}lagMax  = 0;",
        "",
        f"{I}// --- superlags: disabled ---------------------------------------",
        f"{I}slagSize = 0;",
        f"{I}slagMin  = 0;",
        f"{I}slagMax  = 0;",
        f"{I}slagOff  = 0;",
        "",
        f"{I}// --- job-segment parameters ------------------------------------",
        f"{I}segLen  = {seg_len};",
        f"{I}segMLS  = {seg_mls};",
        f"{I}segTHR  = {seg_thr};",
        f"{I}segEdge = {seg_edge};",
        "",
        f"{I}// --- frequency range -------------------------------------------",
        f"{I}fLow  = {f_low};",
        f"{I}fHigh = {f_high};",
        "",
        f"{I}// --- wavelet parameters ----------------------------------------",
        f"{I}levelR = {level_r};",
        f"{I}l_low  = {l_low};",
        f"{I}l_high = {l_high};",
        f"{I}upTDF  = {up_tdf};",
        f"{I}TDSize = {td_size};",
        "",
        f"{I}strcpy(wdmXTalk, \"{wdm_xtalk}\");",
        "",
        f"{I}healpix = {healpix};",
        "",
        f"{I}// --- analysis thresholds ---------------------------------------",
        f"{I}bpp       = {bpp};",
        f"{I}subnet    = {subnet};",
        f"{I}subcut    = {subcut};",
        f"{I}netRHO    = {net_rho};",
        f"{I}netCC     = {net_cc};",
        f"{I}Acore     = {a_core};",
        f"{I}Tgap      = {t_gap};",
        f"{I}Fgap      = {f_gap};",
        f"{I}delta     = {delta};",
        f"{I}cfg_gamma = {cfg_gamma};",
        f"{I}LOUD      = {loud};",
        f"{I}pattern   = {pattern};",
        f"{I}precision = {float(precision)};",
        "",
        f"{I}// --- whitening -------------------------------------------------",
        f"{I}whiteWindow = {white_win};",
        f"{I}whiteStride = {white_str};",
        "",
        f"{I}// --- input data rate -------------------------------------------",
        f"{I}inRate = {in_rate};",
        f"{I}{f_resample_line}",
        "",
        f"{I}// --- DC calibration corrections --------------------------------",
        f"{I}{_join(*dc_cal_stmts)}",
        "",
        f"{I}EFEC = {_boolstr(efec)};",
        "",
        f"{I}// --- simulation: disabled (compare on actual saved data) -------",
        f"{I}simulation = 0;",
        f"{I}nfactor    = 1;",
        "",
        f"{I}// --- channel names (match the exported GWF channel) ------------",
        f"{I}{_join(*channel_stmts)}",
        "",
        f"{I}// --- frame-file lists ------------------------------------------",
        f"{I}{_join(*frfiles_stmts)}",
        "",
        f"{I}// --- DQ file list ----------------------------------------------",
        f"{I}nDQF = {n_dqf};",
        f"{I}dqfile dqf[{n_dqf}] = {{",
        dqf_block,
        f"{I}}};",
        f"{I}for (int i = 0; i < {n_dqf}; i++) DQF[i] = dqf[i];",
        "}",
        "",
    ]

    out_path = os.path.join(config_dir, "user_parameters.C")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# run.sh helper
# ---------------------------------------------------------------------------

def _write_run_sh(cwb_workdir: str) -> None:
    """Write a ``run.sh`` that sets ``HOME_WAT_FILTERS`` and launches cWB."""
    script = textwrap.dedent(f"""\
        #!/usr/bin/env bash
        # Helper script to launch a cWB 2G analysis from this comparison workdir.
        # The wdmXTalk symlink inside this directory requires HOME_WAT_FILTERS
        # to point here so that cWB can locate the cross-talk catalog.
        set -euo pipefail

        WORKDIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
        export HOME_WAT_FILTERS="${{WORKDIR}}"

        echo "HOME_WAT_FILTERS=${{HOME_WAT_FILTERS}}"
        echo "Workdir: ${{WORKDIR}}"

        cd "${{WORKDIR}}"
        cwb_inet 1 1 true
        """)
    run_sh = os.path.join(cwb_workdir, "run.sh")
    with open(run_sh, "w") as fh:
        fh.write(script)
    os.chmod(run_sh, 0o755)
