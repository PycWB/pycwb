"""
Arrow/Parquet-based catalog for pycWB.

Storage layout
--------------
A single Parquet file per run (or per batch job).  The file has two parts:

1. **Schema key-value metadata** – JSON-encoded fields stored under the keys:
   ``b"pycwb_version"``, ``b"config"``, ``b"jobs"``.
   These hold the run configuration and the list of
   :class:`~pycwb.types.job.WaveSegment` job descriptors, keeping the file
   self-contained.

2. **Row data** – one row per reconstructed trigger, using the schema defined by
   :meth:`~pycwb.types.trigger.Trigger.arrow_schema`.  Per-IFO quantities are
   stored as ``list<float>`` columns; injection parameters live in a nullable
   ``struct`` column.

Primary interface
-----------------
:class:`Catalog` is the canonical interface.  Typical usage::

    # --- run time ---
    cat = Catalog.create("catalog/catalog.parquet", config, job_segments)
    cat.add_triggers(trigger)          # Trigger object
    cat.add_events(event)              # legacy Event object

    # --- analysis time ---
    cat = Catalog.open("catalog/catalog.parquet")
    print(cat.version, cat.config, cat.jobs)
    table = cat.triggers()             # pyarrow.Table of all triggers
    table = cat.filter("rho > 5", "net_cc > 0.5")
    table = cat.query("SELECT id, rho FROM triggers WHERE injection.mchirp > 10")
    livetimes = cat.live_time()

Module-level functions are kept as thin shims for backwards compatibility.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import os
from typing import Optional, Union

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from filelock import SoftFileLock

import pycwb
from pycwb.config import Config
from pycwb.types.base_catalog import BaseCatalog
from pycwb.types.job import WaveSegment
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


def _build_schema_metadata(config: Config, jobs: list) -> dict:
    return {
        b"pycwb_version": pycwb.__version__.encode(),
        b"config": orjson.dumps(config.__dict__, option=orjson.OPT_SERIALIZE_NUMPY),
        b"jobs": orjson.dumps(_jobs_to_serialisable(jobs), option=orjson.OPT_SERIALIZE_NUMPY),
    }


def _empty_table(schema: pa.Schema) -> pa.Table:
    return pa.table(
        {f.name: pa.array([], type=f.type) for f in schema},
        schema=schema,
    )


# ---------------------------------------------------------------------------
# Catalog class
# ---------------------------------------------------------------------------

class Catalog(BaseCatalog):
    """Self-contained Arrow/Parquet catalog for a pycWB run.

    A :class:`Catalog` wraps a single ``.parquet`` file.  It stores:

    * **Run metadata** (config, job list, pycwb version) in the Parquet schema's
      key-value metadata so the file is fully self-describing.
    * **Trigger rows** using :meth:`~pycwb.types.trigger.Trigger.arrow_schema`.

    Instantiation
    -------------
    Use the class-methods rather than ``__init__`` directly:

    * :meth:`Catalog.create` – initialise a new catalog on disk.
    * :meth:`Catalog.open`   – open an existing catalog for reading/appending.

    Thread / process safety
    -----------------------
    Write operations are protected by a :class:`~filelock.SoftFileLock`.
    Standard ``fcntl`` locks are unreliable on LIGO cluster filesystems (CIT,
    LDAS), hence the soft variant.
    """

    #: File extension for Parquet-backed catalogs.
    DEFAULT_EXTENSION: str = ".parquet"

    #: Default bare filename used when callers do not specify one.
    DEFAULT_FILENAME: str = "catalog.parquet"

    def __init__(self, filename: str):
        self.filename = os.path.abspath(filename)
        # Lazy-loaded cache; invalidated after every write
        self._meta_cache: Optional[dict] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, filename: str, config: Config,
               jobs: list[WaveSegment]) -> "Catalog":
        """Create an empty Parquet catalog and return a :class:`Catalog` for it.

        Parameters
        ----------
        filename : str
            Destination path, e.g. ``"catalog/catalog.parquet"``.
        config : Config
            Pipeline configuration.
        jobs : list[WaveSegment]
            All job segments for this run.
        """
        ifo_list = getattr(config, "ifo", [])
        schema = Trigger.arrow_schema(ifo_list=ifo_list).with_metadata(
            _build_schema_metadata(config, jobs)
        )
        table = _empty_table(schema)
        with SoftFileLock(filename + ".lock", timeout=10):
            pq.write_table(table, filename, compression="snappy")
        logger.info("Created Arrow catalog: %s", filename)
        return cls(filename)

    @classmethod
    def open(cls, filename: str) -> "Catalog":
        """Open an existing Parquet catalog for reading or appending."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Catalog not found: {filename}")
        return cls(filename)

    # ------------------------------------------------------------------
    # Metadata properties (cached, invalidated on write)
    # ------------------------------------------------------------------

    def _load_meta(self) -> dict:
        if self._meta_cache is None:
            raw = pq.read_schema(self.filename).metadata or {}
            self._meta_cache = {
                "version": raw.get(b"pycwb_version", b"").decode(),
                "config": orjson.loads(raw[b"config"]) if b"config" in raw else {},
                "jobs":   orjson.loads(raw[b"jobs"])   if b"jobs"   in raw else [],
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

    @property
    def ifo_list(self) -> list:
        """IFO names from the catalog's stored config, e.g. ``["H1", "L1"]``.

        Used automatically by :meth:`add_triggers` to produce flat per-IFO
        columns (``time_H1``, ``central_freq_L1``, etc.) matching the schema
        created by :meth:`create`.
        """
        return self.config.get("ifo", [])

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

        ifo_list = self.ifo_list
        schema   = Trigger.arrow_schema(ifo_list=ifo_list)
        new_rows = pa.Table.from_pylist(
            [t.to_arrow_dict(ifo_list=ifo_list) for t in triggers],
            schema=schema,
        )

        with SoftFileLock(self.filename + ".lock", timeout=30):
            existing = pq.read_table(self.filename)
            meta = existing.schema.metadata or {}
            combined = pa.concat_tables([existing, new_rows], promote_options="default")
            combined = combined.replace_schema_metadata(meta)
            pq.write_table(combined, self.filename, compression="snappy")

        self._meta_cache = None  # invalidate after write

    def add_events(self, events) -> None:
        """Convert legacy :class:`~pycwb.types.network_event.Event` objects and append.

        Accepts a single ``Event`` or a list.  This shim lets existing pipeline
        code that still produces ``Event`` objects write to the new Arrow catalog
        without modification.
        """
        if not isinstance(events, list):
            events = [events]
        self.add_triggers([Trigger.from_event(ev) for ev in events])

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def triggers(self, deduplicate: bool = True) -> pa.Table:
        """Return all trigger rows as a :class:`pyarrow.Table`.

        Parameters
        ----------
        deduplicate : bool
            If ``True`` (default), duplicate rows sharing the same
            ``(job_id, id)`` key are removed, keeping the last occurrence.

        Returns
        -------
        pyarrow.Table
            Trigger table using :meth:`~pycwb.types.trigger.Trigger.arrow_schema`.
        """
        table = pq.read_table(self.filename)

        if deduplicate and table.num_rows > 0:
            job_col = pc.cast(table["job_id"], pa.string())
            key_col = pc.binary_join_element_wise(job_col, table["id"], "_")
            seen: dict[str, int] = {}
            for i, k in enumerate(key_col.to_pylist()):
                seen[k] = i
            keep = sorted(seen.values())
            if len(keep) < table.num_rows:
                removed = table.num_rows - len(keep)
                logger.info("Removed %d duplicate trigger(s)", removed)
                table = table.take(keep)

        return table
    
    def total_number(self) -> int:
        """Return the total number of triggers in the catalog."""
        return pq.read_metadata(self.filename).num_rows if os.path.exists(self.filename) else 0

    # ------------------------------------------------------------------
    # Filtering / searching
    # ------------------------------------------------------------------

    def filter(self, *conditions) -> pa.Table:
        """Filter triggers by one or more conditions.

        Each condition can be:

        * A **string** – a Python boolean expression referencing column names,
          e.g. ``"rho > 5"``, ``"net_cc > 0.5"``.  Evaluated row-by-row (safe
          subset of builtins only).  Good for quick interactive use.
        * A **PyArrow expression** – returned by :func:`pyarrow.compute`
          functions; efficient, operates on full arrays.

        For struct sub-fields (injection parameters) use :meth:`query` with
        DuckDB's dot notation, which is more ergonomic.

        Example::

            table = cat.filter("rho > 5", "net_cc > 0.5")
            table = cat.filter(pc.greater(pc.field("rho"), 5))

        Returns
        -------
        pyarrow.Table
        """
        table = self.triggers()
        for cond in conditions:
            if isinstance(cond, str):
                rows = table.to_pydict()
                mask = [
                    bool(eval(cond, {"__builtins__": None, "math": math},  # noqa: S307
                              {k: v[i] for k, v in rows.items()}))
                    for i in range(table.num_rows)
                ]
                table = table.filter(pa.array(mask, type=pa.bool_()))
            else:
                table = table.filter(cond)
        logger.info("filter() returned %d row(s)", table.num_rows)
        return table

    def query(self, sql: str) -> pa.Table:
        """Run an arbitrary DuckDB SQL query against the trigger rows.

        The trigger table is exposed as the relation ``"triggers"`` inside the
        query.  Struct sub-fields (injection parameters) can be accessed with
        dot notation.

        Requires ``duckdb`` to be installed (``pip install duckdb``).

        Example::

            table = cat.query('''
                SELECT id, rho, injection.mchirp, injection.distance
                FROM   triggers
                WHERE  injection IS NOT NULL
                  AND  injection.mchirp > 10
                  AND  rho > 5
            ''')

        Parameters
        ----------
        sql : str
            DuckDB SQL statement.  Use ``triggers`` as the table name.

        Returns
        -------
        pyarrow.Table
        """
        try:
            import duckdb
        except ImportError as exc:
            raise ImportError(
                "duckdb is required for Catalog.query(); install with: pip install duckdb"
            ) from exc

        triggers = self.triggers()  # noqa: F841  (used by DuckDB via local scope)
        return duckdb.query(sql).arrow()

    # ------------------------------------------------------------------
    # Live time
    # ------------------------------------------------------------------

    def live_time(self, filters: Optional[list] = None) -> list:
        """Return a list of per-lag livetime dicts for background estimation.

        Each dict has keys ``"shift"``, ``"livetime"`` (seconds), ``"lag"``.

        Parameters
        ----------
        filters : list of str, optional
            Python boolean expressions evaluated against each livetime dict,
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
            livetime_single = job["end_time"] - job["start_time"]
            for lag in lags:
                livetimes.append({
                    "shift":    job.get("shift"),
                    "livetime": livetime_single,
                    "lag":      int(lag),
                })

        if filters:
            before = len(livetimes)
            fstr = " and ".join(filters)
            livetimes = [lt for lt in livetimes
                         if eval(fstr, {"__builtins__": None}, lt)]  # noqa: S307
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
        n = pq.read_metadata(self.filename).num_rows if os.path.exists(self.filename) else "?"
        return f"Catalog('{self.filename}', triggers={n})"


# ---------------------------------------------------------------------------
# Module-level shims (backwards compatibility)
# ---------------------------------------------------------------------------

def create_catalog(filename: str, config: Config,
                   jobs: list[WaveSegment]) -> Catalog:
    """Create a new catalog; returns the :class:`Catalog` object.

    .. deprecated::
        Prefer ``Catalog.create(filename, config, jobs)``.
    """
    return Catalog.create(filename, config, jobs)


def add_triggers_to_catalog(filename: str, triggers) -> None:
    """Append trigger(s) to an existing catalog file.

    .. deprecated::
        Prefer ``Catalog.open(filename).add_triggers(triggers)``.
    """
    Catalog.open(filename).add_triggers(triggers)


def add_events_to_catalog(filename: str, events) -> None:
    """Convert legacy Event(s) and append to the catalog.

    .. deprecated::
        Prefer ``Catalog.open(filename).add_events(events)``.
    """
    Catalog.open(filename).add_events(events)


def read_catalog_metadata(filename: str) -> dict:
    """Return ``{"version", "config", "jobs"}`` from the catalog metadata.

    .. deprecated::
        Prefer ``Catalog.open(filename).config`` / ``.jobs`` / ``.version``.
    """
    cat = Catalog.open(filename)
    return {"version": cat.version, "config": cat.config, "jobs": cat.jobs}


def read_catalog_triggers(filename: str) -> pa.Table:
    """Return all trigger rows as a :class:`pyarrow.Table`.

    .. deprecated::
        Prefer ``Catalog.open(filename).triggers()``.
    """
    return Catalog.open(filename).triggers()

