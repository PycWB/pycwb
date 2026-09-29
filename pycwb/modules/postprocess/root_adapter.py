"""Adapt cWB background ROOT results to native postproduction tables.

ROOT is an input format only: selection, ranking and rates use the ordinary
postprocess functions. Requires uproot, not PyROOT. No livetime is inferred
from event counts. Detector order must be supplied in cWB network order.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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
from pycwb.post_production.action_spec import action_spec
from pycwb.types.trigger import Trigger


@dataclass
class RootResults:
    """Common trigger/progress tables, with source provenance and job metadata."""

    triggers: pa.Table
    progress: pa.Table
    jobs: list[dict]

    def write(self, directory: str | Path) -> dict[str, str]:
        """Write trigger/progress Parquet tables and return their paths."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, table in (("catalog", self.triggers), ("progress", self.progress)):
            path = directory / f"{name}.parquet"
            pq.write_table(table, path)
            paths[f"{name}_file"] = str(path)
        return paths


def _rows(path, tree_name, branches, required, batch_size):
    import uproot

    with uproot.open(path) as source:
        tree = source[tree_name]
        missing = set(required) - set(tree.keys())
        if missing:
            raise ValueError(f"{path}:{tree_name}: missing branches {sorted(missing)}")
        selected = [b for b in branches if b in tree]
        entry = 0
        for batch in tree.iterate(selected, library="ak", step_size=batch_size):
            for row in batch.to_list():
                yield entry, row
                entry += 1


def _indices(row, nifo):
    for name in ("lag", "slag"):
        values = row[name]
        if len(values) < nifo + 1 or not np.all(np.isfinite(values[: nifo + 1])):
            raise ValueError(
                f"Invalid cWB {name}: expected {nifo} offsets and an index"
            )
        if values[nifo] != int(values[nifo]):
            raise ValueError(f"Noninteger cWB {name} index")
        if (values[nifo] == 0) != all(abs(value) <= 1e-12 for value in values[:nifo]):
            raise ValueError(
                f"cWB {name} index and offsets disagree on zero-lag status"
            )
    return int(row["run"]), int(row["slag"][nifo]), int(row["lag"][nifo])


def read_cwb_root(wave_files, ifo_list, *, live_files=None, batch_size=10000):
    """Read background waveburst/liveTime trees, retaining zero-event exposure.

    ``live_files=None`` reads liveTime from each wave file. Separate merged
    live files may be supplied. Duplicate exposure keys are rejected, never
    silently summed. Use one campaign per call; run numbers must identify
    jobs within that campaign. Simulation truth/efficiency are not inferred.
    """
    if not ifo_list or len(set(ifo_list)) != len(ifo_list):
        raise ValueError("ifo_list must be a nonempty, unique detector order")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    def paths(value):
        if isinstance(value, (str, Path)):
            value = [value]
        result = [str(Path(p).resolve()) for p in value]
        if not result or len(set(result)) != len(result):
            raise ValueError(
                "Input file list must be nonempty and contain no duplicates"
            )
        return result

    waves = paths(wave_files)
    lives = waves if live_files is None else paths(live_files)
    nifo = len(ifo_list)
    # Validate network size before interpreting liveTime's padded arrays.
    import uproot

    for path in waves:
        with uproot.open(path) as source:
            tree = source["waveburst"]
            if "ndim" not in tree:
                raise ValueError(f"{path}: missing ndim branch")
            for batch in tree.iterate(["ndim"], library="ak", step_size=batch_size):
                if any(value != nifo for value in batch["ndim"].to_list()):
                    raise ValueError(
                        "ROOT ndim does not match the supplied detector order"
                    )
    exposure = {}
    shifts = {}
    for path in lives:
        for entry, row in _rows(
            path,
            "liveTime",
            ["run", "live", "lag", "slag", "start", "stop"],
            ["run", "live", "lag", "slag"],
            batch_size,
        ):
            key = _indices(row, nifo)
            if key in exposure:
                raise ValueError(f"Duplicate liveTime key (run, slag, lag): {key}")
            if not np.isfinite(row["live"]) or row["live"] < 0:
                raise ValueError(f"Invalid livetime for {key}")
            shift = tuple(row["slag"][:nifo])
            if key[:2] in shifts and shifts[key[:2]] != shift:
                raise ValueError(f"Inconsistent superlag offsets for {key[:2]}")
            shifts[key[:2]] = shift
            exposure[key] = (path, entry, row)
    if not exposure:
        raise ValueError("No liveTime rows; exposure cannot be inferred from triggers")
    job_keys = sorted(shifts)
    # Keep cWB run numbers when unique; split superlags into distinct native jobs.
    unique_runs = len({k[0] for k in job_keys}) == len(job_keys)
    job_ids = {key: key[0] if unique_runs else i + 1 for i, key in enumerate(job_keys)}
    jobs = [
        {
            "index": job_ids[k],
            "shift": list(shifts[k]),
            "root_run": k[0],
            "root_slag_idx": k[1],
        }
        for k in job_keys
    ]
    progress_rows = []
    for key, (path, entry, row) in exposure.items():
        item = {
            "job_id": job_ids[key[:2]],
            "lag_idx": key[2],
            "livetime": float(row["live"]),
            "status": "completed",
            "root_run": key[0],
            "root_slag_idx": key[1],
            "root_file": path,
            "root_entry": entry,
        }
        for i, ifo in enumerate(ifo_list):
            item[f"time_lag_{ifo}"] = row["lag"][i]
            item[f"segment_lag_{ifo}"] = row["slag"][i]
        progress_rows.append(item)
    rows, extras = [], []
    for path in waves:
        for entry, row in _rows(
            path,
            "waveburst",
            _SCALAR_BRANCHES + _ARRAY_BRANCHES,
            ["run", "ndim", "eventID", "nevent", "rho", "netcc", "time", "lag", "slag"],
            batch_size,
        ):
            if row["ndim"] != nifo:
                raise ValueError("ROOT ndim does not match the supplied detector order")
            key = _indices(row, nifo)
            if key not in exposure:
                raise ValueError(f"Trigger has no matching liveTime row: {key}")
            live = exposure[key][2]
            if (
                row["lag"][:nifo] != live["lag"][:nifo]
                or row["slag"][:nifo] != live["slag"][:nifo]
            ):
                raise ValueError(f"Trigger/liveTime offsets disagree for {key}")
            trigger = _row_to_trigger(row, job_ids[key[:2]])
            trigger.ifo_list = list(ifo_list)
            trigger.id = hashlib.sha256(
                f"{path}:waveburst:{entry}".encode()
            ).hexdigest()[:32]
            item = trigger.to_arrow_dict(ifo_list)
            # Keep missing measurements null instead of converter defaults.
            sources = {
                "likelihood": ["likelihood"],
                "ecor": ["coherent_energy"],
                "ECOR": ["coherent_energy_norm"],
                "norm": ["packet_norm"],
                "penalty": ["penalty"],
                "gnet": ["network_sensitivity"],
                "anet": ["network_alignment_factor"],
                "inet": ["network_index"],
                "neted": [
                    "net_energy_disb",
                    "net_null",
                    "net_energy",
                    "like_sky",
                    "energy_sky",
                ],
                "chirp": [
                    "mchirp",
                    "mchirp_err",
                    "chirp_ellip",
                    "chirp_pfrac",
                    "chirp_efrac",
                ],
            }
            per_ifo = {
                "frequency": "central_freq",
                "low": "freq_low",
                "high": "freq_high",
                "bandwidth": "bandwidth",
                "duration": "duration",
                "noise": "noise_rms",
                "snr": "data_energy",
                "sSNR": "signal_energy",
                "xSNR": "cross_energy",
                "null": "null_energy",
                "nill": "residual_energy",
                "hrss": "hrss",
                "bp": "fp",
                "bx": "fx",
                "rate": "sample_rate",
            }
            for branch, target in per_ifo.items():
                sources[branch] = [f"{target}_{ifo}" for ifo in ifo_list]
            for branch, fields in sources.items():
                if branch not in row:
                    for field in fields:
                        item[field] = None
            # Absent optional plugin vetoes must not masquerade as measured zero.
            if "Qveto" not in row:
                item["q_veto"] = item["q_factor"] = None
            rows.append(item)
            extra = {
                "root_file": path,
                "root_entry": entry,
                "root_run": key[0],
                "root_slag_idx": key[1],
            }
            # Preserve plugin components and double-precision noise RMS. The
            # canonical noise fields are float32, which can move ML splits.
            for branch in ("Qveto", "Lveto", "noise"):
                for i, value in enumerate(row.get(branch, [])):
                    extra[f"{branch}{i}"] = value
            extras.append(extra)
    triggers = pa.Table.from_pylist(rows, schema=Trigger.arrow_schema(ifo_list))
    if extras:
        extra_table = pa.Table.from_pandas(pd.DataFrame(extras), preserve_index=False)
    else:
        extra_table = pa.table(
            {
                "root_file": pa.array([], type=pa.string()),
                "root_entry": pa.array([], type=pa.int64()),
                "root_run": pa.array([], type=pa.int64()),
                "root_slag_idx": pa.array([], type=pa.int64()),
            }
        )
    for name in extra_table.column_names:
        triggers = triggers.append_column(name, extra_table[name])
    progress = pa.Table.from_pylist(progress_rows)
    metadata = {
        b"config": json.dumps({"ifo": list(ifo_list)}).encode(),
        b"jobs": json.dumps(jobs).encode(),
        b"cwb_root_adapter": json.dumps(
            {"version": 1, "wave_files": waves, "live_files": lives}
        ).encode(),
    }
    return RootResults(
        triggers.replace_schema_metadata(metadata),
        progress.replace_schema_metadata(metadata),
        jobs,
    )


@action_spec(
    inputs=["wave_files", "live_files"],
    outputs=["output_dir"],
    description="Adapt cWB ROOT events and exposure for shared postproduction",
)
def import_cwb_root(
    work_dir,
    wave_files,
    ifo_list,
    live_files=None,
    output_dir=None,
    batch_size=10000,
    **kwargs,
):
    """Workflow action; return tables in memory and optionally write Parquet."""

    def resolve(value):
        if value is None:
            return None
        if isinstance(value, (str, Path)):
            return str(Path(work_dir) / value)
        return [str(Path(work_dir) / p) for p in value]

    result = read_cwb_root(
        resolve(wave_files),
        ifo_list,
        live_files=resolve(live_files),
        batch_size=batch_size,
    )
    output = {
        "triggers": result.triggers.to_pandas(),
        "progress": result.progress.to_pandas(),
        "jobs": result.jobs,
    }
    if output_dir is not None:
        output.update(result.write(Path(work_dir) / output_dir))
    return output
