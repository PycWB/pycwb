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
    table = cat.query("SELECT id, rho FROM triggers WHERE injection.approximant = 'WNB' AND rho > 5")
    livetimes = cat.live_time()

Module-level functions are kept as thin shims for backwards compatibility.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import os
import tempfile
import time
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
# Progress schema
# ---------------------------------------------------------------------------

PROGRESS_SCHEMA = pa.schema([
    ("job_id",     pa.int32()),
    ("trial_idx",  pa.int32()),
    ("lag_idx",    pa.int32()),
    ("n_triggers", pa.int32()),
    ("livetime",   pa.float64()),
    ("timestamp",  pa.float64()),
    ("status",     pa.string()),
])


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


def _validate_catalog_file(path: str) -> None:
    """Validate that the on-disk parquet file exists and is non-empty."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Catalog not found: {path}")
    size = os.path.getsize(path)
    if size == 0:
        raise ValueError(
            f"Catalog file is empty (0 bytes): {path}. "
            "This usually indicates an interrupted/failed previous write. "
            "Remove the empty file and recreate the catalog."
        )


def _write_table_atomic(table: pa.Table, filename: str, compression: str = "snappy") -> None:
    """Write parquet to a temp file and atomically replace the destination."""
    target_dir = os.path.dirname(os.path.abspath(filename)) or "."
    os.makedirs(target_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".catalog_tmp_", suffix=".parquet", dir=target_dir)
    os.close(fd)
    try:
        pq.write_table(table, tmp_path, compression=compression)
        if os.path.getsize(tmp_path) == 0:
            raise ValueError(f"Temporary parquet write produced empty file: {tmp_path}")
        os.replace(tmp_path, filename)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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
            _write_table_atomic(table, filename, compression="snappy")
        logger.info("Created Arrow catalog: %s", filename)
        return cls(filename)

    @classmethod
    def open(cls, filename: str) -> "Catalog":
        """Open an existing Parquet catalog for reading or appending."""
        _validate_catalog_file(filename)
        return cls(filename)

    # ------------------------------------------------------------------
    # Metadata properties (cached, invalidated on write)
    # ------------------------------------------------------------------

    def _load_meta(self) -> dict:
        if self._meta_cache is None:
            _validate_catalog_file(self.filename)
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
            _validate_catalog_file(self.filename)
            existing = pq.read_table(self.filename)
            meta = existing.schema.metadata or {}
            combined = pa.concat_tables([existing, new_rows], promote_options="default")
            combined = combined.replace_schema_metadata(meta)
            _write_table_atomic(combined, self.filename, compression="snappy")

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

    def remove_stale_triggers(self, job_id: int, completed_lags: dict) -> int:
        """Remove triggers belonging to incomplete (lag_idx, trial_idx) pairs.

        A (trial_idx, lag_idx) pair is considered *incomplete* when it is not
        present in *completed_lags* but a trigger with that combination already
        exists in the catalog (left over from a previous interrupted run).

        Parameters
        ----------
        job_id : int
            The job segment index whose triggers should be inspected.
        completed_lags : dict[int, set[int]]
            ``{trial_idx: {lag_idx, ...}}`` as returned by
            :meth:`get_completed_lags`.  Pairs present here are kept; all
            other pairs for *job_id* are removed.

        Returns
        -------
        int
            Number of stale trigger rows removed.
        """
        with SoftFileLock(self.filename + ".lock", timeout=30):
            _validate_catalog_file(self.filename)
            table = pq.read_table(self.filename)
            if table.num_rows == 0:
                return 0

            # Build a boolean keep-mask: keep rows that are NOT for this job_id,
            # or that belong to a completed (trial_idx, lag_idx) pair.
            keep_mask = []
            job_id_col   = table["job_id"].to_pylist()
            lag_idx_col  = table["lag_idx"].to_pylist()
            trial_idx_col = table["trial_idx"].to_pylist()

            for jid, lid, tid in zip(job_id_col, lag_idx_col, trial_idx_col):
                if jid != job_id:
                    keep_mask.append(True)
                else:
                    # Keep only if this (trial_idx, lag_idx) is in completed_lags
                    keep_mask.append(
                        tid in completed_lags and lid in completed_lags[tid]
                    )

            n_stale = keep_mask.count(False)
            if n_stale == 0:
                return 0

            stale_pairs: set[tuple[int, int]] = set()
            for i, k in enumerate(keep_mask):
                if not k:
                    stale_pairs.add((trial_idx_col[i], lag_idx_col[i]))

            keep_indices = [i for i, k in enumerate(keep_mask) if k]
            if keep_indices:
                filtered = table.take(keep_indices)
            else:
                filtered = _empty_table(table.schema.remove_metadata())
            meta = table.schema.metadata or {}
            filtered = filtered.replace_schema_metadata(meta)
            _write_table_atomic(filtered, self.filename, compression="snappy")

        pairs_str = ", ".join(
            f"trial={trial} lag={lag_id}" for trial, lag_id in sorted(stale_pairs)
        )
        logger.info(
            "Removed %d stale trigger(s) for job %d [%s]",
            n_stale, job_id, pairs_str,
        )
        return n_stale
    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    @property
    def progress_file(self) -> str:
        """Path to the companion progress Parquet file.

        Derived by replacing the ``catalog`` prefix with ``progress``:
        ``catalog/catalog_1-5.parquet`` → ``catalog/progress_1-5.parquet``

        Returns
        -------
        str
            Absolute path to the progress Parquet file.
        """
        dirname = os.path.dirname(self.filename)
        basename = os.path.basename(self.filename).replace("catalog", "progress", 1)
        return os.path.join(dirname, basename)

    def add_lag_progress(self, job_id: int, trial_idx: int, lag_idx: int,
                         n_triggers: int, livetime: float,
                         status: str = "completed") -> None:
        """Record the completion of a single lag.

        Appends one row to the progress Parquet file (read-modify-write with
        atomic replacement).

        Parameters
        ----------
        job_id : int
            Job segment index.
        trial_idx : int
            Trial (injection) index; 0 for no injections.
        lag_idx : int
            0-based lag index.
        n_triggers : int
            Number of triggers found in this lag.
        livetime : float
            Effective post-veto analysed duration in seconds.
        status : str
            Lag outcome: ``"completed"`` (default) or ``"skipped_segTHR"``.
        """
        pf = self.progress_file
        new_row = pa.table(
            {
                "job_id":     [job_id],
                "trial_idx":  [trial_idx],
                "lag_idx":    [lag_idx],
                "n_triggers": [n_triggers],
                "livetime":   [livetime],
                "timestamp":  [time.time()],
                "status":     [status],
            },
            schema=PROGRESS_SCHEMA,
        )

        with SoftFileLock(pf + ".lock", timeout=30):
            if os.path.exists(pf) and os.path.getsize(pf) > 0:
                existing = pq.read_table(pf, schema=PROGRESS_SCHEMA)
                combined = pa.concat_tables([existing, new_row])
            else:
                combined = new_row
            _write_table_atomic(combined, pf, compression="snappy")

    def get_completed_lags(self, job_id: int) -> dict[int, set[int]]:
        """Return completed lags for a given job.

        All statuses (including ``"skipped_segTHR"``) count as processed
        so the lag is not re-run on restart.

        Returns
        -------
        dict[int, set[int]]
            ``{trial_idx: {lag_0, lag_1, ...}}`` for all completed lags
            belonging to *job_id*.  Empty dict if no progress file exists.
        """
        pf = self.progress_file
        if not os.path.exists(pf) or os.path.getsize(pf) == 0:
            return {}

        # Read with promote_options to handle old files missing the status column
        try:
            table = pq.read_table(pf, schema=PROGRESS_SCHEMA)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            table = pq.read_table(pf)
        mask = pc.equal(table["job_id"], job_id)
        rows = table.filter(mask)

        result: dict[int, set[int]] = {}
        for row in rows.to_pylist():
            result.setdefault(row["trial_idx"], set()).add(row["lag_idx"])
        return result

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
            # Build composite key: job_id + id, plus trial_idx / lag_idx when present
            key_cols = ["job_id", "id"] + [c for c in ("trial_idx", "lag_idx") if c in table.schema.names]
            parts = [pc.cast(table["job_id"], pa.string()), table["id"]]
            for col in ("trial_idx", "lag_idx"):
                if col in table.schema.names:
                    parts.append(pc.cast(table[col], pa.string()))
            key_col = pc.binary_join_element_wise(*parts, "_")
            seen: dict[str, int] = {}
            for i, k in enumerate(key_col.to_pylist()):
                seen[k] = i
            keep = sorted(seen.values())
            if len(keep) < table.num_rows:
                removed = table.num_rows - len(keep)
                logger.info("Removed %d duplicate trigger(s) (dedup keys: %s)", removed, key_cols)
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
                SELECT id, rho, injection.name, injection.hrss, injection.approximant
                FROM   triggers
                WHERE  injection IS NOT NULL
                  AND  injection.approximant = 'WNB'
                  AND  rho > 5
            ''')

            # Access waveform-specific parameters stored in the JSON blob:
            table = cat.query(
                "SELECT injection.name,"
                " json_extract(injection.parameters, '$.frequency')::FLOAT AS freq,"
                " json_extract(injection.parameters, '$.Q')::FLOAT AS Q"
                " FROM triggers WHERE injection IS NOT NULL"
            )

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
        result = duckdb.query(sql).arrow()
        # DuckDB ≥ 1.0 returns a RecordBatchReader; materialise to Table.
        if isinstance(result, pa.RecordBatchReader):
            result = result.read_all()
        return result

    # ------------------------------------------------------------------
    # Live time
    # ------------------------------------------------------------------

    def live_time(self, filters: Optional[list] = None) -> list:
        """Return a list of per-lag livetime dicts for background estimation.

        If a progress file exists, uses the stored post-veto livetimes (one
        entry per completed lag).  Otherwise falls back to the legacy
        ``end_time - start_time`` estimate for all jobs × lags.

        Each dict has keys ``"job_id"``, ``"shift"``, ``"livetime"``
        (seconds), ``"lag"``.

        Parameters
        ----------
        filters : list of str, optional
            Python boolean expressions evaluated against each livetime dict,
            e.g. ``["lag == 0"]``.

        Returns
        -------
        list of dict
        """
        pf = self.progress_file
        if os.path.exists(pf) and os.path.getsize(pf) > 0:
            # ── Progress-based (accurate post-veto livetimes) ──
            try:
                progress = pq.read_table(pf, schema=PROGRESS_SCHEMA)
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                progress = pq.read_table(pf)
            jobs_by_index = {j.get("index", i): j for i, j in enumerate(self.jobs)}

            livetimes = []
            for row in progress.to_pylist():
                # Exclude lags that were skipped (e.g. segTHR) from livetime sums
                status = row.get("status", "completed")
                if status != "completed":
                    continue
                job = jobs_by_index.get(row["job_id"], {})
                livetimes.append({
                    "job_id":   row["job_id"],
                    "shift":    job.get("shift"),
                    "livetime": row["livetime"],
                    "lag":      row["lag_idx"],
                })
        else:
            # ── Legacy fallback (raw segment duration) ──
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

