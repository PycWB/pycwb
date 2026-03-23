"""
JSON-backed catalog for pycWB (demonstration of :class:`~pycwb.types.base_catalog.BaseCatalog`).

Storage layout
--------------
A single ``.json`` file (serialised with :mod:`orjson`) with the top-level structure::

    {
        "pycwb_version": "<str>",
        "config":   { ... },
        "jobs":     [ ... ],
        "triggers": [ ... ]
    }

Each entry in ``"triggers"`` is the flat dict produced by
:meth:`~pycwb.types.trigger.Trigger.to_json_dict`.

This backend is intentionally simple and human-readable, making it convenient
for small runs, debugging, and interactive inspection.  For production runs
with many triggers prefer the Parquet-backed :class:`~pycwb.modules.catalog.Catalog`.

Primary interface
-----------------
:class:`JSONCatalog` implements :class:`~pycwb.types.base_catalog.BaseCatalog`.
Typical usage::

    # --- run time ---
    cat = JSONCatalog.create("catalog/catalog.json", config, job_segments)
    cat.add_triggers(trigger)
    cat.add_events(event)

    # --- analysis time ---
    cat = JSONCatalog.open("catalog/catalog.json")
    rows = cat.triggers()            # list[dict]
    rows = cat.filter("rho > 5")
    rows = cat.query("SELECT id, rho FROM triggers WHERE rho > 5")
    livetimes = cat.live_time()
"""
from __future__ import annotations

import dataclasses
import logging
import math
import os
from typing import Optional, Union

import numpy as np
import orjson
from filelock import SoftFileLock

import pycwb
from pycwb.types.base_catalog import BaseCatalog
from pycwb.types.trigger import Trigger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _jobs_to_serialisable(jobs: list) -> list:
    result = []
    for j in jobs:
        if dataclasses.is_dataclass(j) and not isinstance(j, type):
            result.append(dataclasses.asdict(j))
        elif isinstance(j, dict):
            result.append(j)
        else:
            result.append(vars(j))
    return result


def _trigger_to_dict(trigger: Trigger) -> dict:
    """Flatten a :class:`~pycwb.types.trigger.Trigger` to a JSON-serialisable dict.

    Per-IFO lists are kept as lists.  :class:`~pycwb.types.trigger.InjectionParams`
    is nested under the key ``"injection"`` (``null`` for background triggers).
    """
    d = dataclasses.asdict(trigger)
    # numpy scalars / arrays are not JSON-serialisable by default; orjson handles
    # them via OPT_SERIALIZE_NUMPY, so we just return the raw dict here and let
    # orjson deal with it at write time.
    return d


def _read_file(filename: str) -> dict:
    with open(filename, "rb") as fh:
        return orjson.loads(fh.read())


def _write_file(filename: str, data: dict) -> None:
    raw = orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2)
    with open(filename, "wb") as fh:
        fh.write(raw)


# ---------------------------------------------------------------------------
# JSONCatalog class
# ---------------------------------------------------------------------------

class JSONCatalog(BaseCatalog):
    """Human-readable JSON catalog for a pycWB run.

    Implements :class:`~pycwb.types.base_catalog.BaseCatalog` using
    :mod:`orjson` for fast JSON serialisation.

    The entire catalog is a single ``.json`` file — ideal for small runs,
    debugging, and cases where human readability matters.  For large-scale
    production runs with many triggers consider the Parquet-backed
    :class:`~pycwb.modules.catalog.Catalog`.

    Instantiation
    -------------
    * :meth:`JSONCatalog.create` – initialise a new catalog on disk.
    * :meth:`JSONCatalog.open`   – open an existing catalog for reading/appending.

    Thread / process safety
    -----------------------
    Write operations use a :class:`~filelock.SoftFileLock`.
    """

    DEFAULT_EXTENSION: str = ".json"
    DEFAULT_FILENAME: str = "catalog.json"

    def __init__(self, filename: str):
        self.filename = os.path.abspath(filename)
        self._meta_cache: Optional[dict] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, filename: str, config, jobs: list) -> "JSONCatalog":
        """Create an empty JSON catalog and return a :class:`JSONCatalog` for it.

        Parameters
        ----------
        filename : str
            Destination path, e.g. ``"catalog/catalog.json"``.
        config : Config
            Pipeline configuration.
        jobs : list
            All job segments for this run.
        """
        data = {
            "pycwb_version": pycwb.__version__,
            "config": config.__dict__,
            "jobs": _jobs_to_serialisable(jobs),
            "triggers": [],
        }
        with SoftFileLock(filename + ".lock", timeout=10):
            _write_file(filename, data)
        logger.info("Created JSON catalog: %s", filename)
        return cls(filename)

    @classmethod
    def open(cls, filename: str) -> "JSONCatalog":
        """Open an existing JSON catalog for reading or appending."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Catalog not found: {filename}")
        return cls(filename)

    # ------------------------------------------------------------------
    # Metadata properties (cached, invalidated on write)
    # ------------------------------------------------------------------

    def _load_meta(self) -> dict:
        if self._meta_cache is None:
            data = _read_file(self.filename)
            self._meta_cache = {
                "version": data.get("pycwb_version", ""),
                "config":  data.get("config", {}),
                "jobs":    data.get("jobs", []),
            }
        return self._meta_cache

    @property
    def version(self) -> str:
        """pycwb version string recorded when the catalog was created."""
        return self._load_meta()["version"]

    @property
    def config(self) -> dict:
        """Pipeline configuration dict stored in the catalog."""
        return self._load_meta()["config"]

    @property
    def jobs(self) -> list:
        """List of job-segment dicts stored in the catalog."""
        return self._load_meta()["jobs"]

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add_triggers(self, triggers: Union[Trigger, list[Trigger]]) -> None:
        """Append one or more :class:`~pycwb.types.trigger.Trigger` objects.

        Uses a read–modify–write cycle under a :class:`~filelock.SoftFileLock`.
        """
        if not isinstance(triggers, list):
            triggers = [triggers]
        if not triggers:
            return

        new_rows = [_trigger_to_dict(t) for t in triggers]

        with SoftFileLock(self.filename + ".lock", timeout=30):
            data = _read_file(self.filename)
            data.setdefault("triggers", []).extend(new_rows)
            _write_file(self.filename, data)

        self._meta_cache = None  # invalidate after write
        logger.info("Appended %d trigger(s) to %s", len(new_rows), self.filename)

    def add_events(self, events) -> None:
        """Convert legacy :class:`~pycwb.types.network_event.Event` objects and append.

        Accepts a single ``Event`` or a list.
        """
        if not isinstance(events, list):
            events = [events]
        self.add_triggers([Trigger.from_event(ev) for ev in events])

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def triggers(self, deduplicate: bool = True) -> list[dict]:
        """Return all trigger records as a ``list[dict]``.

        Parameters
        ----------
        deduplicate : bool
            If ``True`` (default), duplicate rows sharing the same
            ``(job_id, id)`` key are removed, keeping the last occurrence.

        Returns
        -------
        list[dict]
            One dict per trigger, matching the fields of
            :class:`~pycwb.types.trigger.Trigger`.
        """
        rows: list[dict] = _read_file(self.filename).get("triggers", [])

        if deduplicate and rows:
            seen: dict[str, int] = {}
            for i, row in enumerate(rows):
                key = f"{row.get('job_id')}__{row.get('id')}"
                seen[key] = i
            keep = sorted(seen.values())
            if len(keep) < len(rows):
                removed = len(rows) - len(keep)
                logger.info("Removed %d duplicate trigger(s)", removed)
                rows = [rows[i] for i in keep]

        return rows

    # ------------------------------------------------------------------
    # Filtering / searching
    # ------------------------------------------------------------------

    def filter(self, *conditions) -> list[dict]:
        """Filter triggers by one or more string expression conditions.

        Each condition is a Python boolean expression evaluated against the
        trigger's fields, e.g. ``"rho > 5"``, ``"net_cc > 0.5"``.

        For injection sub-fields, reference the nested dict:
        ``"injection is not None and injection['hrss'] > 1e-22"``.

        Returns
        -------
        list[dict]
        """
        rows = self.triggers()
        for cond in conditions:
            rows = [
                row for row in rows
                if bool(eval(cond, {"__builtins__": None, "math": math}, row))  # noqa: S307
            ]
        logger.info("filter() returned %d row(s)", len(rows))
        return rows

    def query(self, sql: str) -> object:
        """Run a DuckDB SQL query against the trigger records.

        The trigger list is exposed as the relation ``"triggers"`` inside the
        query.  Because the data is a list of dicts, DuckDB automatically
        handles nested ``injection`` fields with dot notation.

        Requires ``duckdb`` (``pip install duckdb``).

        Parameters
        ----------
        sql : str
            DuckDB SQL statement.  Reference the table as ``triggers``.

        Returns
        -------
        pyarrow.Table
            Results as an Arrow table (DuckDB native output).

        Example::

            rows = cat.query(
                "SELECT id, rho, injection.name, injection.approximant FROM triggers"
                " WHERE injection IS NOT NULL AND injection.hrss > 1e-22"
            )
        """
        try:
            import duckdb
        except ImportError as exc:
            raise ImportError(
                "duckdb is required for JSONCatalog.query(); install with: pip install duckdb"
            ) from exc

        triggers = self.triggers()  # noqa: F841  (used by DuckDB via local scope)
        return duckdb.query(sql).arrow()

    # ------------------------------------------------------------------
    # Live time
    # ------------------------------------------------------------------

    def live_time(self, filters: Optional[list] = None) -> list:
        """Return per-lag livetime dicts for background estimation.

        Each dict contains ``"shift"``, ``"livetime"`` (seconds), and ``"lag"``.

        Parameters
        ----------
        filters : list of str, optional
            Python boolean expressions applied to each livetime dict,
            e.g. ``["lag == 0"]``.

        Returns
        -------
        list of dict
        """
        cfg  = self.config
        jobs = self.jobs
        lags = np.arange(cfg.get("lagSize", 1))

        livetimes = []
        for job in jobs:
            livetime_single = job["analyze_end"] - job["analyze_start"]
            for lag in lags:
                livetimes.append({
                    "shift":    job.get("shift"),
                    "livetime": livetime_single,
                    "lag":      int(lag),
                })

        if filters:
            before = len(livetimes)
            fstr = " and ".join(filters)
            livetimes = [
                lt for lt in livetimes
                if eval(fstr, {"__builtins__": None}, lt)  # noqa: S307
            ]
            logger.info("live_time filter removed %d entries", before - len(livetimes))

        total = sum(lt["livetime"] for lt in livetimes)
        logger.info(
            "Total live time: %.1f s (%.2f days, %.2f years)",
            total, total / 86400, total / 86400 / 365,
        )
        return livetimes

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        try:
            n = len(_read_file(self.filename).get("triggers", []))
        except Exception:
            n = "?"
        return f"JSONCatalog('{self.filename}', triggers={n})"
