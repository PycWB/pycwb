"""Efficiency vs hrss plots — workflow-compatible.

Computes efficiency curves (fraction of injections recovered vs. hrss) at a
given IFAR threshold.  The IFAR threshold is determined by ranking background
events by the explicitly selected statistic (default: XGBoost probability).
Use ranking_par="rhor" and scored_file to share the background ranking and cuts.

Workflow actions
----------------
``postprocess.plot_efficiency.plot_efficiency_vs_hrss``
    Plot efficiency vs hrss at a fixed IFAR threshold and save the figure.
    Also computes the hrss at 50% efficiency.

``postprocess.plot_efficiency.compute_hrss50``
    Compute the hrss value at 50% recovery for a given IFAR.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from pycwb.post_production.action_spec import action_spec
from pycwb.modules.postprocess.efficiency_metrics import (
    _matched_ranking_scores,
    _ranking_metadata,
    _IFAR_PRESETS as _IFAR_PRESETS,
    _parse_ifar_seconds,
    _validate_unique_simulations,
    _validate_fixed_hrss_population,
    _empirical_probability_detection,
    _compute_efficiency_by_waveform_matched,
    _compute_efficiency_vs_hrss_by_waveform_matched,
    _fit_efficiency_curve as _fit_efficiency_curve,
    _interpolate_hrss50,
    _interpolate_hrss50_curve,
    _parse_waveform_q_frequency,
)
from pycwb.modules.postprocess.efficiency_plots import (
    _plot_efficiency_by_waveform_panels as _plot_efficiency_by_waveform_panels,
    _plot_efficiency_curve,
    _plot_waveform_efficiency as _plot_waveform_efficiency,
)

logger = logging.getLogger(__name__)


@action_spec(
    outputs=['output_file'],
    inputs=['sim_catalog', 'bkg_catalog', 'model_file', 'config_file', 'scored_file'],
    description='Compute hrss at 50% efficiency for a given IFAR threshold',
)
def compute_hrss50(
    work_dir: str,
    sim_catalog: str,
    bkg_catalog: str,
    livetime: float,
    ifar: str = "1yr",
    output_file: Optional[str] = None,
    **kwargs,
) -> dict:
    """Compute hrss at 50% efficiency for a given IFAR threshold.

    Parameters
    ----------
    work_dir : str
        Base directory.
    sim_catalog : str
        Path to scored SIM parquet (with ``xgb_prob`` column).
    bkg_catalog : str
        Path to BKG parquet (used to compute prob threshold from FAR).
    livetime : float
        Background live time in seconds.
    ranking_par : str, optional (via kwargs)
        Statistic shared with the background calibration, default ``xgb_prob``.
    scored_file : str, optional (via kwargs)
        Pre-scored SIM catalog. Scores are joined by ID; events removed by
        prediction cuts remain misses. Takes precedence over model_file.
    ifar : str
        IFAR threshold: ``"1yr"``, ``"6mo"``, ``"1mo"``, ``"1wk"``, ``"1day"``.
    output_file : str, optional
        Save plot to this path.
    matched_right_file : str, required (via kwargs)
        Path to ``matched_right.parquet``. Efficiency is
        computed by counting **unique simulations** (``sim_sim_idx``) rather
        than individual triggers.  One injection → one trial.
    exclude_vetoed : bool, optional (via kwargs)
        If True and *matched_right_file* provided, exclude vetoed simulations
        (cat0|cat1|cat2|across_segments) from the denominator.

    Returns
    -------
    dict
        ``hrss50``, ``ifar_sec``, ``prob_threshold``, ``efficiency_curve``.
    """
    def resolve(path):
        return path if os.path.isabs(path) else os.path.join(work_dir, path)

    matched_file = kwargs.get("matched_right_file")
    if not matched_file:
        raise ValueError("matched_right_file is required to retain missed injections in the denominator")
    mr = pd.read_parquet(resolve(matched_file))
    _validate_unique_simulations(mr)
    _validate_fixed_hrss_population(mr)
    ranking_par = kwargs.get("ranking_par", "xgb_prob")
    mr["_ranking"] = _matched_ranking_scores(
        mr, work_dir, ranking_par, kwargs.get("scored_file"), kwargs.get("model_file"),
        kwargs.get("nifo", 2), kwargs.get("search", "blf"), kwargs.get("config_file"),
    )
    exclude_vetoed = kwargs.get("exclude_vetoed", False)
    if exclude_vetoed:
        columns = ["sim_vetoed_cat0", "sim_vetoed_cat1", "sim_vetoed_cat2", "sim_across_segments"]
        mr = mr[~mr[columns].fillna(False).astype(bool).any(axis=1)].copy()
    ifar_sec = _parse_ifar_seconds(ifar)
    background = pd.read_parquet(resolve(bkg_catalog))
    detected, threshold = _empirical_probability_detection(mr["_ranking"], background[ranking_par], livetime, ifar_sec)
    mr["detected"] = mr.id.notna() & detected
    mr["amplitude"] = pd.to_numeric(mr.sim_hrss, errors="coerce")
    if not (np.isfinite(mr.amplitude) & (mr.amplitude > 0)).all():
        raise ValueError("Each injection must have a finite positive sim_hrss")
    curve = [dict(hrss=float(h),efficiency=float(g.detected.mean()),n_total=len(g),
                  n_recovered=int(g.detected.sum())) for h,g in mr.groupby("amplitude",sort=True)]
    crossing = _interpolate_hrss50(curve)
    if output_file:
        path = resolve(output_file)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _plot_efficiency_curve(curve,crossing,ifar,ifar_sec,threshold,path, ranking_par=ranking_par)
    return dict(hrss50=crossing,ifar_sec=ifar_sec,**_ranking_metadata(ranking_par, threshold),
                efficiency_curve=curve,method="unique_simulation",exclude_vetoed=exclude_vetoed)


# ---------------------------------------------------------------------------
# plot_efficiency_vs_hrss (alias with plot output)
# ---------------------------------------------------------------------------

@action_spec(
    outputs=['output_file'],
    inputs=['sim_catalog', 'bkg_catalog', 'model_file', 'config_file', 'scored_file'],
    description='Plot efficiency vs hrss at a given IFAR threshold',
)
def plot_efficiency_vs_hrss(
    work_dir: str,
    sim_catalog: str,
    bkg_catalog: str,
    livetime: float,
    ifar: str = "1yr",
    output_file: str = "efficiency_vs_hrss.png",
    **kwargs,
) -> dict:
    """Plot efficiency vs hrss at a given IFAR threshold.

    See :func:`compute_hrss50` for parameter details.
    """
    return compute_hrss50(
        work_dir=work_dir,
        sim_catalog=sim_catalog,
        bkg_catalog=bkg_catalog,
        livetime=livetime,
        ifar=ifar,
        output_file=output_file,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Waveform-level efficiency using matched_right.parquet
# ---------------------------------------------------------------------------

@action_spec(
    outputs=['output_file'],
    inputs=['sim_catalog', 'matched_file', 'bkg_catalog', 'model_file', 'config_file', 'scored_file'],
    description='Compute per-waveform efficiency using matched_right cross-match',
)
def compute_efficiency_by_waveform(
    work_dir: str,
    sim_catalog: str,
    matched_file: str,
    bkg_catalog: str,
    livetime: float,
    model_file: str,
    search: str = "blf",
    nifo: int = 2,
    config_file: Optional[str] = None,
    ifar: str = "1mo",
    output_file: Optional[str] = None,
    **kwargs,
) -> dict:
    """Score unique matched injections, retaining unrecovered sources in the denominator.

    Uses ``matched_right.parquet`` (from ``pycwb match-simulations``) to
    identify which injections were detected by the cWB pipeline.  The
    XGBoost IFAR threshold provides an additional cut.

    Parameters
    ----------
    work_dir : str
        Base directory.
    sim_catalog : str
        Retained for call compatibility; the matched injection table supplies the denominator.
    matched_file : str
        Path to ``matched_right.parquet``.
    bkg_catalog : str
        Path to scored BKG catalog (with ``xgb_prob`` column).
    livetime : float
        Background live time in seconds.
    model_file : str
        Path to trained XGBoost model.
    ranking_par : str, optional (via kwargs)
        Statistic shared with the background calibration, default ``xgb_prob``.
    scored_file : str, optional (via kwargs)
        Pre-scored SIM catalog. Scores are joined by ID; events removed by
        prediction cuts remain misses. Takes precedence over model_file.
    ifar : str
        IFAR threshold: ``"1yr"``, ``"1mo"``, ``"1wk"``, ``"1day"``.
    output_file : str, optional
        Save per-waveform efficiency bar chart to this path.

    Returns
    -------
    dict
        ``efficiency_by_waveform`` list, ``prob_threshold``, ``ifar_sec``.
    """
    if kwargs.get("use_unique_sim", True) is not True:
        raise ValueError("Trigger counts are not an injection denominator; use_unique_sim must be True")
    return _compute_efficiency_by_waveform_matched(
        work_dir, matched_file, bkg_catalog, livetime, model_file,
        search, nifo, config_file, ifar, _parse_ifar_seconds(ifar), output_file,
        kwargs.get("exclude_vetoed", False),
        kwargs.get("ranking_par", "xgb_prob"), kwargs.get("scored_file"),
    )


# ---------------------------------------------------------------------------
# Efficiency vs hrss by waveform (per-waveform curves)
# ---------------------------------------------------------------------------

@action_spec(
    outputs=['output_file', 'fit_parameters_file'],
    inputs=['sim_catalog', 'matched_file', 'bkg_catalog', 'model_file', 'config_file', 'scored_file'],
    description='Efficiency vs hrss curves for each waveform, grouped by Q-factor',
)
def compute_efficiency_vs_hrss_by_waveform(
    work_dir: str,
    sim_catalog: str,
    matched_file: str,
    bkg_catalog: str,
    livetime: float,
    model_file: str,
    search: str = "blf",
    nifo: int = 2,
    config_file: Optional[str] = None,
    ifar: str = "1mo",
    output_file: Optional[str] = None,
    **kwargs,
) -> dict:
    """Efficiency vs hrss curves for each waveform, grouped by Q-factor.

    Produces a multi-panel plot with efficiency vs hrss for all waveforms,
    organized by Q-factor (Q3, Q9, Q100) in separate subplots.

    Parameters
    ----------
    work_dir : str
        Base directory.
    sim_catalog : str
        Retained for call compatibility; the matched injection table supplies the denominator.
    matched_file : str
        Path to ``matched_right.parquet``.
    bkg_catalog : str
        Path to scored BKG catalog.
    livetime : float
        Background live time in seconds.
    model_file : str
        Trained XGBoost model path.
    ranking_par : str, optional (via kwargs)
        Statistic shared with the background calibration, default ``xgb_prob``.
    scored_file : str, optional (via kwargs)
        Pre-scored SIM catalog. Scores are joined by ID; events removed by
        prediction cuts remain misses. Takes precedence over model_file.
    ifar : str
        IFAR threshold (``"1yr"``, ``"1mo"``, ``"1wk"``, ``"1day"``).
    output_file : str, optional
        Save plot to this path.

    Returns
    -------
    dict
        ``curves`` list of per-waveform efficiency data,
        ``prob_threshold``, ``ifar_sec``.
    """
    if kwargs.get("use_unique_sim", True) is not True:
        raise ValueError("Trigger counts are not an injection denominator; use_unique_sim must be True")
    return _compute_efficiency_vs_hrss_by_waveform_matched(
        work_dir, matched_file, bkg_catalog, livetime, model_file,
        search, nifo, config_file, ifar, output_file,
        kwargs.get("exclude_vetoed", False), kwargs.get("fit_parameters_file"),
        kwargs.get("ranking_par", "xgb_prob"), kwargs.get("scored_file"),
    )


# ---------------------------------------------------------------------------
# Per-waveform hrss50 CSV report (multiple IFARs)
# ---------------------------------------------------------------------------

@action_spec(
    outputs=['output_csv'],
    inputs=['sim_catalog', 'matched_file', 'bkg_catalog', 'model_file', 'config_file', 'scored_file'],
    description='Compute hrss50 for each waveform at multiple IFARs, save CSV',
)
def compute_hrss50_by_waveform_csv(
    work_dir: str,
    sim_catalog: str,
    matched_file: str,
    bkg_catalog: str,
    livetime: float,
    model_file: str,
    search: str = "blf",
    nifo: int = 2,
    config_file: Optional[str] = None,
    ifars: str = "1mo,1yr,10yr",
    output_csv: Optional[str] = None,
    **kwargs,
) -> dict:
    """Compute hrss50 for each waveform at multiple IFAR thresholds, save CSV.

    Parameters
    ----------
    ifars : str
        Comma-separated IFAR labels, e.g. ``"1mo,1yr,10yr"``.
    output_csv : str, optional
        Save CSV to this path.
    """
    rows = []
    for label in [v.strip() for v in ifars.split(",") if v.strip()]:
        result = compute_efficiency_vs_hrss_by_waveform(
            work_dir, sim_catalog, matched_file, bkg_catalog, livetime, model_file,
            search, nifo, config_file, label, None,
            exclude_vetoed=kwargs.get("exclude_vetoed", False),
            ranking_par=kwargs.get("ranking_par", "xgb_prob"), scored_file=kwargs.get("scored_file"),
        )
        for fit in result["fit_parameters"]:
            row = dict(fit)
            row["fit_status"] = row.pop("status", None)
            rows.append(row)
    table = pd.DataFrame(rows)
    if output_csv:
        path = output_csv if os.path.isabs(output_csv) else os.path.join(work_dir, output_csv)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        table.to_csv(path,index=False)
    return {"hrss50_csv": table.to_dict(orient="records")}
