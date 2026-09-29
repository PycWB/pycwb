"""Read cWB simulation truth and reconstructed events for shared postproduction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pycwb.modules.catalog.convert_root import (
    _ARRAY_BRANCHES,
    _SCALAR_BRANCHES,
    _row_to_trigger,
)
from pycwb.modules.postprocess.root_adapter import _rows
from pycwb.post_production.action_spec import action_spec
from pycwb.types.trigger import Trigger


@action_spec(
    inputs=["wave_file", "mdc_file"],
    outputs=["output_dir"],
    description="Import cWB injection truth and uniquely matched triggers",
)
def import_cwb_simulation(
    work_dir,
    wave_file,
    ifo_list,
    output_dir,
    mdc_file=None,
    ranking_par="rho_alt",
    time_window=0.1,
    batch_size=10000,
    **kwargs,
):
    """Adapt a merged simulation tree, retaining *all* MDC rows in the denominator.

    cWB's injection association (time[nIFO]) is used, rather than matching
    the reconstructed time to the nearest injection. Unique selection occurs
    before the recovery-time cut, like Toolbox::setUniqueEvents followed by
    cwb_report_sim. The supplied file must be a single simulation campaign.
    Unmatched or ambiguous truth associations are errors.
    """

    from pycwb.types.simulation import InjectionParams

    if not ifo_list or len(set(ifo_list)) != len(ifo_list):
        raise ValueError("Supply a unique, nonempty detector order")
    if ranking_par not in ("rho", "rho_alt"):
        raise ValueError("Unique-event ranking must be rho or rho_alt")
    if not np.isfinite(time_window) or time_window < 0:
        raise ValueError("time_window must be finite and nonnegative")
    wave_path = Path(work_dir) / wave_file
    truth_path = Path(work_dir) / (mdc_file or wave_file)
    out = Path(work_dir) / output_dir
    out.mkdir(parents=True, exist_ok=True)
    nifo = len(ifo_list)
    truth = []
    by_key = {}
    required = ["time", "type", "name", "strain", "factor"]
    for entry, row in _rows(
        truth_path, "mdc", required + ["run"], required, batch_size
    ):
        key = (float(row["time"][0]), float(row["factor"]), int(row["type"]))
        if key in by_key:
            raise ValueError(f"Ambiguous MDC injection key: {key}")
        if not np.isfinite(row["strain"]) or row["strain"] <= 0:
            raise ValueError("MDC strain must be finite and positive")
        sid = len(truth)
        by_key[key] = sid
        truth.append(
            {
                "sim_idx": sid,
                "name": row["name"],
                "hrss": float(row["strain"]),
                "gps_time": key[0],
                "root_factor": key[1],
                "root_type": key[2],
                "root_entry": entry,
                "root_file": str(truth_path.resolve()),
            }
        )
    rows = []
    extra = []
    requested = list(dict.fromkeys(_SCALAR_BRANCHES + _ARRAY_BRANCHES + ["name"]))
    for entry, row in _rows(
        wave_path,
        "waveburst",
        requested,
        ["ndim", "time", "type", "factor", "rho", "slag"],
        batch_size,
    ):
        if row["ndim"] != nifo or len(row["time"]) < 2 * nifo:
            raise ValueError("ROOT simulation detector count/time layout mismatch")
        key = (float(row["time"][nifo]), float(row["factor"]), int(row["type"][1]))
        if key not in by_key:
            raise ValueError(f"Recovered trigger has no exact MDC truth key: {key}")
        sid = by_key[key]
        sim = truth[sid]
        trigger = _row_to_trigger(row, int(row.get("run", 0)))
        trigger.ifo_list = list(ifo_list)
        trigger.id = hashlib.sha256(
            f"{wave_path.resolve()}:{entry}".encode()
        ).hexdigest()[:32]
        trigger.injection = InjectionParams(
            name=sim["name"],
            hrss=sim["hrss"],
            gps_time=sim["gps_time"],
            time=row["time"][nifo : 2 * nifo],
        )
        rows.append(trigger.to_arrow_dict(ifo_list))
        item = {
            "sim_sim_idx": sid,
            "root_file": str(wave_path.resolve()),
            "root_entry": entry,
            "root_injection_time": key[0],
            "root_factor": key[1],
            "root_type": key[2],
            "root_slag_idx": int(row["slag"][nifo]),
        }
        # Keep ROOT noise precision alongside the canonical float32 fields.
        for branch in ["Qveto", "Lveto", "noise"]:
            for i, value in enumerate(row.get(branch, [])):
                item[f"{branch}{i}"] = value
        extra.append(item)
    table = pa.Table.from_pylist(rows, schema=Trigger.arrow_schema(ifo_list))
    events = table.to_pandas()
    for key in [
        "sim_sim_idx",
        "root_file",
        "root_entry",
        "root_injection_time",
        "root_factor",
        "root_type",
        "root_slag_idx",
    ]:
        events[key] = [row[key] for row in extra]
    for key in sorted({k for row in extra for k in row} - set(events.columns)):
        events[key] = [row.get(key, np.nan) for row in extra]
    if len(events):
        # Stable ties retain the first source row. Cluster times within 1 ms,
        # separately for each superlag/factor, as in cWB's unique operation.
        ordered = events.sort_values(
            ["root_slag_idx", "root_factor", "root_injection_time"], kind="stable"
        )
        selected = []
        for _, group in ordered.groupby(["root_slag_idx", "root_factor"], sort=True):
            start = None
            best = None
            for index, row in group.iterrows():
                if start is None or abs(row.root_injection_time - start) > 0.001:
                    if best is not None:
                        selected.append(best)
                    start = row.root_injection_time
                    best = index
                elif row[ranking_par] > events.loc[best, ranking_par]:
                    best = index
            if best is not None:
                selected.append(best)
        unique = events.loc[selected].copy()
    else:
        unique = events.copy()
    recovered = unique[
        (unique.gps_time - unique.root_injection_time).abs() <= time_window
    ].copy()
    if recovered.sim_sim_idx.duplicated().any():
        raise ValueError(
            "Multiple recoveries per truth row; split superlag campaigns before efficiency"
        )
    sims = pd.DataFrame(
        truth,
        columns=[
            "sim_idx",
            "name",
            "hrss",
            "gps_time",
            "root_factor",
            "root_type",
            "root_entry",
            "root_file",
        ],
    )
    right = sims.add_prefix("sim_").merge(
        recovered, on="sim_sim_idx", how="left", validate="one_to_one"
    )
    metadata = {b"config": json.dumps({"ifo": list(ifo_list)}).encode(), b"jobs": b"[]"}
    paths = {}
    for key, frame in [
        ("catalog", recovered),
        ("raw", events),
        ("unique", unique),
        ("simulations", sims),
        ("matched", right),
    ]:
        path = out / f"{key}.parquet"
        pq.write_table(
            pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
                metadata
            ),
            path,
        )
        paths[f"{key}_file"] = str(path)
    summary = {
        "injected": len(sims),
        "raw": len(events),
        "unique": len(unique),
        "recovered": len(recovered),
        "missed": len(sims) - len(recovered),
        "ranking_par": ranking_par,
        "time_window": time_window,
    }
    (out / "import_summary.json").write_text(json.dumps(summary, indent=2))
    return {**paths, **summary}
