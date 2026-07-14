"""Interactive sky-angle comparison plots for multiple injection runs."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from astropy import units as u
from plotly.colors import qualitative

from pycwb.post_production.action_spec import action_spec


def _resolve(work_dir: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(work_dir, path)


def _prepare_output(work_dir: str, path: str) -> str:
    resolved = _resolve(work_dir, path)
    os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
    return resolved


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "as_py"):
        converted = value.as_py()
        return converted if isinstance(converted, dict) else {}
    return {}


def _run_name(values: dict[str, Any]) -> str:
    """Return the canonical run name with legacy manifest compatibility."""
    return str(values.get("name") or values.get("label") or "run")


def _row_run_name(values: dict[str, Any]) -> str:
    """Return a run name from combined tables written by either schema."""
    return str(values.get("run_name") or values.get("run_label") or "run")


def _parameters(injection: dict[str, Any]) -> dict[str, Any]:
    value = injection.get("parameters", {})
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _event_key(values: dict[str, Any]) -> tuple[int, int]:
    parameters = _parameters(values)
    sim_idx = values.get("sim_idx", parameters.get("sim_idx"))
    trial_idx = values.get("trial_idx", parameters.get("trial_idx", 0))
    if sim_idx is None:
        raise ValueError("Injection is missing sim_idx in both its row and parameters")
    return int(sim_idx), int(trial_idx or 0)


def _event_id(sim_idx: Any, trial_idx: Any) -> str:
    """Return the stable browser-side identifier used for linked selection."""
    return f"{int(sim_idx)}:{int(trial_idx)}"


def _angle_deg(value: Any, unit: str) -> float:
    try:
        angle = value if isinstance(value, u.Quantity) else float(value) * u.Unit(unit)
        return float(angle.to_value(u.deg))
    except (TypeError, ValueError, u.UnitConversionError) as exc:
        raise ValueError(
            f"Angle value {value!r} with unit {unit!r} is not convertible to degrees"
        ) from exc


def _separation_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    a1, d1, a2, d2 = np.radians([ra1, dec1, ra2, dec2])
    cosine = (
        np.sin(d1) * np.sin(d2)
        + np.cos(d1) * np.cos(d2) * np.cos(a1 - a2)
    )
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _plot_lon(ra_deg: float) -> float:
    wrapped = (float(ra_deg) + 180.0) % 360.0 - 180.0
    return -wrapped


def _unit_vector(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra, dec = np.radians([ra_deg, dec_deg])
    return np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])


def _great_circle_path(
    start: tuple[float, float],
    stop: tuple[float, float],
    samples: int = 32,
) -> tuple[list[Optional[float]], list[Optional[float]]]:
    first = _unit_vector(*start)
    second = _unit_vector(*stop)
    omega = float(np.arccos(np.clip(np.dot(first, second), -1.0, 1.0)))
    sin_omega = math.sin(omega)
    lons: list[Optional[float]] = []
    lats: list[Optional[float]] = []
    previous: Optional[float] = None
    for fraction in np.linspace(0.0, 1.0, samples + 1):
        if abs(sin_omega) < 1e-12:
            point = first
        else:
            point = (
                math.sin((1.0 - fraction) * omega) / sin_omega * first
                + math.sin(fraction * omega) / sin_omega * second
            )
        point /= np.linalg.norm(point)
        ra = math.degrees(math.atan2(point[1], point[0])) % 360.0
        dec = math.degrees(math.asin(float(point[2])))
        lon = _plot_lon(ra)
        if previous is not None and abs(lon - previous) > 180.0:
            lons.append(None)
            lats.append(None)
        lons.append(lon)
        lats.append(dec)
        previous = lon
    return lons, lats


def _scheduled_truth(
    injections: pd.DataFrame,
    *,
    injection_angle_unit: str,
    strict_truth: bool,
    tolerance_deg: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in injections.iterrows():
        values = row.to_dict()
        key = _event_key(values)
        run_name = _row_run_name(values)
        records.append({
            "run_index": int(values["run_index"]),
            "run_name": run_name,
            "run_label": run_name,
            "sim_idx": key[0],
            "trial_idx": key[1],
            "injected_ra": _angle_deg(values["ra"], injection_angle_unit) % 360.0,
            "injected_dec": _angle_deg(values["dec"], injection_angle_unit),
            "gps_time": float(values.get("gps_time", np.nan)),
        })
    truth = pd.DataFrame(records)
    if truth.empty:
        raise ValueError("The combined injection table is empty")

    key_columns = ["sim_idx", "trial_idx"]
    n_runs = truth["run_index"].nunique()
    for key, group in truth.groupby(key_columns, sort=False):
        ra = group["injected_ra"].to_numpy(dtype=float)
        dec = group["injected_dec"].to_numpy(dtype=float)
        gps = group["gps_time"].to_numpy(dtype=float)
        if strict_truth:
            if group["run_index"].nunique() != n_runs or len(group) != n_runs:
                raise ValueError(
                    f"Injection event {key} is not scheduled exactly once in every run"
                )
            ra_delta = np.abs(((ra - ra[0] + 180.0) % 360.0) - 180.0)
            gps_delta = np.abs(gps - gps[0])
            finite_gps_delta = gps_delta[np.isfinite(gps_delta)]
            if (
                np.nanmax(ra_delta) > tolerance_deg
                or np.nanmax(np.abs(dec - dec[0])) > tolerance_deg
                or (
                    finite_gps_delta.size
                    and np.max(finite_gps_delta) > 1e-6
                )
            ):
                raise ValueError(f"Injection truth differs between runs for event {key}")
    return truth


def _recovered_events(
    triggers: pd.DataFrame,
    *,
    injection_angle_unit: str,
    recovered_angle_unit: str,
    ranking_column: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in triggers.iterrows():
        injection = _mapping(row.get("injection"))
        if not injection:
            continue
        key = _event_key(injection)
        injected_ra = _angle_deg(injection["ra"], injection_angle_unit) % 360.0
        injected_dec = _angle_deg(injection["dec"], injection_angle_unit)
        recovered_ra = _angle_deg(row["ra"], recovered_angle_unit) % 360.0
        recovered_dec = _angle_deg(row["dec"], recovered_angle_unit)
        ranking_value = row.get(ranking_column, np.nan)
        run_name = _row_run_name(row.to_dict())
        records.append({
            "run_index": int(row["run_index"]),
            "run_name": run_name,
            "run_label": run_name,
            "sim_idx": key[0],
            "trial_idx": key[1],
            "injected_ra": injected_ra,
            "injected_dec": injected_dec,
            "recovered_ra": recovered_ra,
            "recovered_dec": recovered_dec,
            "angular_error_deg": _separation_deg(
                injected_ra, injected_dec, recovered_ra, recovered_dec
            ),
            "rho": float(row.get("rho", np.nan)),
            "net_cc": float(row.get("net_cc", np.nan)),
            "ranking_value": (
                float(ranking_value) if pd.notna(ranking_value) else -np.inf
            ),
        })
    recovered = pd.DataFrame(records)
    if recovered.empty:
        raise ValueError("No recovered triggers with injection metadata were found")
    recovered = recovered.sort_values(
        ["run_index", "sim_idx", "trial_idx", "ranking_value"],
        ascending=[True, True, True, False],
    )
    return recovered.drop_duplicates(
        ["run_index", "sim_idx", "trial_idx"], keep="first"
    ).reset_index(drop=True)


def _run_summaries(
    manifest: dict[str, Any],
    truth: pd.DataFrame,
    recovered: pd.DataFrame,
) -> list[dict[str, Any]]:
    n_truth = int(truth[["sim_idx", "trial_idx"]].drop_duplicates().shape[0])
    summaries = []
    for run in manifest["runs"]:
        run_index = int(run["run_index"])
        group = recovered[recovered["run_index"] == run_index]
        errors = group["angular_error_deg"].to_numpy(dtype=float)
        run_name = _run_name(run)
        summaries.append({
            "run_index": run_index,
            "name": run_name,
            "label": run_name,
            "metadata": run.get("metadata", {}),
            "n_injections": n_truth,
            "n_recovered": int(len(group)),
            "median_error_deg": float(np.median(errors)) if len(errors) else None,
            "mean_error_deg": float(np.mean(errors)) if len(errors) else None,
            "p90_error_deg": float(np.percentile(errors, 90)) if len(errors) else None,
            "max_error_deg": float(np.max(errors)) if len(errors) else None,
            "within_1_deg": int(np.sum(errors <= 1.0)),
            "within_5_deg": int(np.sum(errors <= 5.0)),
            "within_10_deg": int(np.sum(errors <= 10.0)),
            "above_30_deg": int(np.sum(errors > 30.0)),
            "median_rho": float(np.nanmedian(group["rho"])) if len(group) else None,
            "median_net_cc": float(np.nanmedian(group["net_cc"])) if len(group) else None,
        })
    return summaries


def _sky_figure(
    manifest: dict[str, Any],
    truth: pd.DataFrame,
    recovered: pd.DataFrame,
) -> go.Figure:
    figure = go.Figure()
    unique_truth = truth.drop_duplicates(["sim_idx", "trial_idx"])
    recovered_by_key = recovered.groupby(["sim_idx", "trial_idx"])["run_index"].nunique()
    injection_text = []
    for _, row in unique_truth.iterrows():
        key = (int(row["sim_idx"]), int(row["trial_idx"]))
        recovered_count = int(recovered_by_key.get(key, 0))
        status = f"recovered in {recovered_count}/{len(manifest['runs'])} runs"
        injection_text.append(
            f"sim {key[0]}, trial {key[1]}<br>injected RA={row['injected_ra']:.3f}°, "
            f"Dec={row['injected_dec']:.3f}°<br>{status}"
        )
    figure.add_trace(go.Scattergeo(
        lon=[_plot_lon(value) for value in unique_truth["injected_ra"]],
        lat=unique_truth["injected_dec"],
        mode="markers",
        name="shared injections",
        meta={"role": "shared-injection"},
        customdata=[
            _event_id(row.sim_idx, row.trial_idx)
            for row in unique_truth.itertuples()
        ],
        marker={"symbol": "circle-open", "size": 8, "color": "#1b9e77", "line": {"width": 2}},
        text=injection_text,
        hovertemplate="%{text}<extra></extra>",
    ))

    symbols = ["diamond", "triangle-up", "square", "cross", "star"]
    colors = qualitative.Safe
    for run in manifest["runs"]:
        run_index = int(run["run_index"])
        name = _run_name(run)
        group = recovered[recovered["run_index"] == run_index]
        color = colors[run_index % len(colors)]
        path_lon: list[Optional[float]] = []
        path_lat: list[Optional[float]] = []
        path_event_ids: list[Optional[str]] = []
        for _, row in group.iterrows():
            lon, lat = _great_circle_path(
                (row["injected_ra"], row["injected_dec"]),
                (row["recovered_ra"], row["recovered_dec"]),
            )
            path_lon.extend(lon + [None])
            path_lat.extend(lat + [None])
            path_event_ids.extend(
                [_event_id(row["sim_idx"], row["trial_idx"])] * len(lon)
                + [None]
            )
        figure.add_trace(go.Scattergeo(
            lon=path_lon,
            lat=path_lat,
            customdata=path_event_ids,
            mode="lines",
            name=f"{name} paths",
            legendgroup=f"run-{run_index}",
            meta={"role": "recovery-path", "default_opacity": 0.55},
            line={"color": color, "width": 1.5},
            opacity=0.55,
            hoverinfo="skip",
        ))
        figure.add_trace(go.Scattergeo(
            lon=[],
            lat=[],
            mode="lines",
            name=f"{name} selected path",
            legendgroup=f"run-{run_index}",
            showlegend=False,
            meta={"role": "selected-recovery-path"},
            line={"color": color, "width": 3.5},
            opacity=0.0,
            hoverinfo="skip",
        ))
        hover = [
            f"sim {int(row.sim_idx)}, trial {int(row.trial_idx)}<br>"
            f"injected RA={row.injected_ra:.3f}°, Dec={row.injected_dec:.3f}°<br>"
            f"recovered RA={row.recovered_ra:.3f}°, Dec={row.recovered_dec:.3f}°<br>"
            f"error={row.angular_error_deg:.3f}°; rho={row.rho:.2f}; net_cc={row.net_cc:.3f}"
            for row in group.itertuples()
        ]
        figure.add_trace(go.Scattergeo(
            lon=[_plot_lon(value) for value in group["recovered_ra"]],
            lat=group["recovered_dec"],
            mode="markers",
            name=f"{name} recovery",
            legendgroup=f"run-{run_index}",
            meta={"role": "recovery"},
            customdata=[
                _event_id(row.sim_idx, row.trial_idx)
                for row in group.itertuples()
            ],
            marker={
                "symbol": symbols[run_index % len(symbols)],
                "size": 8,
                "color": color,
                "line": {"width": 1, "color": color},
            },
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    figure.update_geos(
        projection_type="mollweide",
        showframe=True,
        showcoastlines=False,
        showcountries=False,
        showland=False,
        showocean=False,
        bgcolor="rgba(0,0,0,0)",
        lonaxis={
            "showgrid": True,
            "gridwidth": 1,
            "dtick": 60,
        },
        lataxis={"showgrid": True, "gridwidth": 1, "dtick": 30},
    )
    figure.update_layout(
        title={
            "text": (
                "Injected and recovered ICRS positions (RA increases right-to-left)"
                "<br><sup>Click a shared injection to highlight its recoveries; "
                "click it again or double-click to reset.</sup>"
            ),
        },
        margin={"l": 25, "r": 25, "t": 55, "b": 70},
        height=640,
        legend={"orientation": "h", "y": -0.10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _histogram_figure(
    manifest: dict[str, Any],
    recovered: pd.DataFrame,
    bin_size_deg: float,
) -> go.Figure:
    figure = go.Figure()
    colors = qualitative.Safe
    for run in manifest["runs"]:
        run_index = int(run["run_index"])
        group = recovered[recovered["run_index"] == run_index]
        figure.add_trace(go.Histogram(
            x=group["angular_error_deg"],
            name=_run_name(run),
            marker_color=colors[run_index % len(colors)],
            opacity=0.58,
            histnorm="percent",
            xbins={"start": 0.0, "end": 180.0, "size": float(bin_size_deg)},
            hovertemplate="error bin %{x}°<br>%{y:.2f}%<extra>%{fullData.name}</extra>",
        ))
    figure.update_layout(
        title="Angular-error distribution",
        barmode="overlay",
        xaxis_title="Injected-to-recovered great-circle error (deg)",
        yaxis_title="Recovered events (%)",
        xaxis_range=[0, 180],
        height=470,
        margin={"l": 60, "r": 25, "t": 55, "b": 55},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


_SKY_SELECTION_POST_SCRIPT = r"""
(function () {
  const plot = document.getElementById('{plot_id}');
  const dimmedPointOpacity = 0.12;
  const dimmedPathOpacity = 0.08;

  function roleOf(trace) {
    return trace.meta && trace.meta.role ? trace.meta.role : '';
  }

  function resetSelection() {
    plot.__pycwbSelectedInjection = null;
    plot.data.forEach(function (trace, traceIndex) {
      const role = roleOf(trace);
      if (role === 'shared-injection' || role === 'recovery') {
        const count = (trace.customdata || []).length;
        Plotly.restyle(plot, {
          'marker.opacity': [Array(count).fill(1.0)],
          'marker.size': [Array(count).fill(8)]
        }, [traceIndex]);
      } else if (role === 'recovery-path') {
        const defaultOpacity = trace.meta.default_opacity || 0.55;
        Plotly.restyle(plot, {'opacity': defaultOpacity}, [traceIndex]);
      } else if (role === 'selected-recovery-path') {
        Plotly.restyle(plot, {
          'lon': [[]],
          'lat': [[]],
          'opacity': 0.0
        }, [traceIndex]);
      }
    });
  }

  function selectInjection(eventId) {
    plot.__pycwbSelectedInjection = eventId;
    plot.data.forEach(function (trace, traceIndex) {
      const role = roleOf(trace);
      if (role === 'shared-injection' || role === 'recovery') {
        const eventIds = trace.customdata || [];
        const opacity = eventIds.map(function (candidate) {
          return candidate === eventId ? 1.0 : dimmedPointOpacity;
        });
        const size = eventIds.map(function (candidate) {
          return candidate === eventId ? 12 : 7;
        });
        Plotly.restyle(plot, {
          'marker.opacity': [opacity],
          'marker.size': [size]
        }, [traceIndex]);
      } else if (role === 'recovery-path') {
        Plotly.restyle(plot, {'opacity': dimmedPathOpacity}, [traceIndex]);
      } else if (role === 'selected-recovery-path') {
        const source = plot.data.find(function (candidate) {
          return roleOf(candidate) === 'recovery-path'
            && candidate.legendgroup === trace.legendgroup;
        });
        const selectedLon = [];
        const selectedLat = [];
        if (source) {
          const eventIds = source.customdata || [];
          eventIds.forEach(function (candidate, pointIndex) {
            if (candidate === eventId) {
              selectedLon.push(source.lon[pointIndex]);
              selectedLat.push(source.lat[pointIndex]);
            }
          });
        }
        Plotly.restyle(plot, {
          'lon': [selectedLon],
          'lat': [selectedLat],
          'opacity': selectedLon.length ? 1.0 : 0.0
        }, [traceIndex]);
      }
    });
  }

  plot.on('plotly_click', function (event) {
    const point = event.points && event.points[0];
    if (!point || roleOf(point.data) !== 'shared-injection') {
      return;
    }
    const eventId = point.customdata;
    if (plot.__pycwbSelectedInjection === eventId) {
      resetSelection();
    } else {
      selectInjection(eventId);
    }
  });

  plot.on('plotly_doubleclick', resetSelection);
})();
"""


def _write_figure(
    figure: go.Figure,
    filename: str,
    *,
    post_script: Optional[str] = None,
) -> None:
    figure.write_html(
        filename,
        include_plotlyjs="cdn",
        full_html=True,
        post_script=post_script,
        config={
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )


@action_spec(
    outputs=["map_file", "histogram_file", "summary_file", "data_file"],
    inputs=["triggers_file", "injections_file", "manifest_file"],
    display_name="Compare sky-angle recovery",
    description="Create interactive multi-run sky recovery and angle-error plots",
)
def plot_angle_error_comparison(
    work_dir: str,
    triggers_file: str,
    injections_file: str,
    manifest_file: str,
    output_dir: str = "public/angle_error_comparison/plots",
    map_filename: str = "sky_recovery.html",
    histogram_filename: str = "angle_error_histogram.html",
    summary_filename: str = "angle_error_summary.json",
    data_filename: str = "angle_error_events.csv",
    injection_angle_unit: str = "rad",
    recovered_angle_unit: str = "deg",
    ranking_column: str = "rho",
    strict_truth: bool = True,
    truth_tolerance_deg: float = 1e-6,
    histogram_bin_size_deg: float = 5.0,
    **kwargs,
) -> dict[str, Any]:
    """Plot shared injection truth against recovery from any number of runs."""
    work_dir = os.path.abspath(str(work_dir))
    triggers = pd.read_parquet(_resolve(work_dir, triggers_file))
    injections = pd.read_parquet(_resolve(work_dir, injections_file))
    with open(_resolve(work_dir, manifest_file), encoding="utf-8") as handle:
        manifest = json.load(handle)

    truth = _scheduled_truth(
        injections,
        injection_angle_unit=injection_angle_unit,
        strict_truth=bool(strict_truth),
        tolerance_deg=float(truth_tolerance_deg),
    )
    recovered = _recovered_events(
        triggers,
        injection_angle_unit=injection_angle_unit,
        recovered_angle_unit=recovered_angle_unit,
        ranking_column=ranking_column,
    )
    summaries = _run_summaries(manifest, truth, recovered)

    base = _prepare_output(work_dir, output_dir)
    os.makedirs(base, exist_ok=True)
    map_path = os.path.join(base, map_filename)
    histogram_path = os.path.join(base, histogram_filename)
    summary_path = os.path.join(base, summary_filename)
    data_path = os.path.join(base, data_filename)

    _write_figure(
        _sky_figure(manifest, truth, recovered),
        map_path,
        post_script=_SKY_SELECTION_POST_SCRIPT,
    )
    _write_figure(
        _histogram_figure(manifest, recovered, histogram_bin_size_deg),
        histogram_path,
    )
    recovered.to_csv(data_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({"runs": summaries}, handle, indent=2)

    def relative(path: str) -> str:
        return os.path.relpath(path, work_dir)

    plots = [
        {
            "title": "Injected and recovered sky positions",
            "html_file": relative(map_path),
            "height": 700,
            "description": (
                "Shared injections and per-run great-circle recovery paths. "
                "Click a shared injection to highlight its matching recoveries "
                "and paths."
            ),
        },
        {
            "title": "Angular-error distribution",
            "html_file": relative(histogram_path),
            "height": 530,
            "description": "Normalized recovered-event histograms on common bins.",
        },
    ]
    return {
        "map_file": relative(map_path),
        "histogram_file": relative(histogram_path),
        "summary_file": relative(summary_path),
        "data_file": relative(data_path),
        "plots": plots,
        "summary": summaries,
        "n_runs": len(summaries),
    }


__all__ = ["plot_angle_error_comparison"]
