"""
pycwb.modules.catalog.matching
================================

Trigger-to-simulation matching utilities.

Two complementary strategies are provided:

1. :func:`match_triggers_to_simulations` — pure-Python sort+bisect approach
   for matching in-memory :class:`~pycwb.types.trigger.Trigger` lists against
   simulation row dicts.  O((N + M) log M) per trial group.

2. :func:`match_simulations_parquet` — DuckDB interval join for Parquet files.
   Suitable when both catalogs are on disk and ``duckdb`` is installed.

Matching criterion
------------------
A trigger **T** and simulation **S** match when all three conditions hold:

* ``T.trial_idx == S.trial_idx``  — same injection trial (no cross-trial matches)
* ``T.job_id == S.job_id`` when simulation summaries carry ``job_id`` —
  same scheduled job/superlag owner
* ``min(T.event_start + T.segment_lag) < S.real_end``   — trigger starts before injection ends
* ``max(T.event_stop  + T.segment_lag) > S.real_start`` — trigger ends after injection starts

i.e. the trigger's reconstructed time window overlaps the injected waveform's
true GPS extent.  An optional ``window_buffer`` (seconds) can be added
symmetrically to the trigger window to account for reconstruction bias.

Algorithm (sort+bisect)
-----------------------
::

    sims sorted by real_start per trial_idx
    ─────────────────────────────────────────────────────────────────
    real_start:  s0   s1   s2   s3   s4   s5   s6   s7
                 │                        │
                 ▼                        ▼
    for trigger [t_start ────────── t_stop]:

        hi = bisect_left(real_starts, t_stop)
             → index of first sim that starts at or after t_stop
               (those definitely cannot overlap)

        scan sims[0:hi] and keep those with real_end > t_start

This is O(M log M + N (log M + k)) where M = simulations, N = triggers,
k = average matches per trigger (typically ≤ 1 for well-separated injections).
"""

from __future__ import annotations

import bisect
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa
    from pycwb.types.trigger import Trigger

logger = logging.getLogger(__name__)


def _trigger_window(trig: "Trigger") -> tuple[float, float]:
    """Return a trigger window in the common analysis-time frame."""
    starts = list(trig.event_start) if trig.event_start else [trig.gps_time]
    stops = list(trig.event_stop) if trig.event_stop else [trig.gps_time]
    segment_lag = list(getattr(trig, "segment_lag", []) or [])
    if segment_lag and len(segment_lag) == len(starts) == len(stops):
        starts = [start + lag for start, lag in zip(starts, segment_lag)]
        stops = [stop + lag for stop, lag in zip(stops, segment_lag)]
    starts.append(trig.gps_time)
    stops.append(trig.gps_time)
    return min(starts), max(stops)


def _parquet_trigger_window_exprs(trig_schema: "pa.Schema") -> tuple[str, str]:
    """Build DuckDB trigger-window expressions in common analysis time."""
    names = set(trig_schema.names)

    def quoted(name: str) -> str:
        return f't."{name}"'

    def aligned_expr(prefix: str, ifo: str) -> str:
        col = f"{prefix}_{ifo}"
        lag_col = f"segment_lag_{ifo}"
        if lag_col in names:
            return f"({quoted(col)} + COALESCE({quoted(lag_col)}, 0.0))"
        return quoted(col)

    start_terms = [
        f"COALESCE({aligned_expr('event_start', name.removeprefix('event_start_'))}, t.gps_time)"
        for name in trig_schema.names
        if name.startswith("event_start_")
    ]
    stop_terms = [
        f"COALESCE({aligned_expr('event_stop', name.removeprefix('event_stop_'))}, t.gps_time)"
        for name in trig_schema.names
        if name.startswith("event_stop_")
    ]
    if not start_terms or not stop_terms:
        return "t.gps_time", "t.gps_time"
    return (
        f"LEAST({', '.join(start_terms + ['t.gps_time'])})",
        f"GREATEST({', '.join(stop_terms + ['t.gps_time'])})",
    )


# ---------------------------------------------------------------------------
# In-memory sort+bisect matcher
# ---------------------------------------------------------------------------

def match_triggers_to_simulations(
    triggers: list["Trigger"],
    simulations: list[dict],
    *,
    window_buffer: float = 0.0,
    how: str = "inner",
) -> list[tuple["Trigger | None", "dict | None"]]:
    """Match in-memory Triggers to simulation summary rows.

    Parameters
    ----------
    triggers:
        List of :class:`~pycwb.types.trigger.Trigger` objects.
    simulations:
        List of simulation row dicts as produced by ``build_simulation_summary``
        (must have keys ``trial_idx``, ``real_start``, ``real_end``).
    window_buffer:
        Optional time buffer (seconds) added symmetrically to the trigger
        window before comparing.  Useful to account for reconstruction timing
        bias; 0.1 – 0.5 s is typical.
    how : {"inner", "left", "right", "outer"}
        Join type:

        * ``"inner"`` *(default)* — only matched ``(Trigger, sim)`` pairs.
        * ``"left"``  — all triggers; ``sim=None`` for unmatched triggers.
        * ``"right"`` — all simulations; ``trigger=None`` for unmatched sims.
        * ``"outer"`` — all triggers and all simulations; ``None`` on the
          unmatched side of each row.

    Returns
    -------
    list of ``(Trigger | None, dict | None)`` pairs
        For matched pairs both elements are non-None.  For unmatched triggers
        (``how`` in ``{"left", "outer"}``) the sim element is ``None``.  For
        unmatched simulations (``how`` in ``{"right", "outer"}``) the trigger
        element is ``None``.
    """
    if how not in ("inner", "left", "right", "outer"):
        raise ValueError(f"how must be 'inner', 'left', 'right', or 'outer'; got {how!r}")

    # ── 1. Group and sort simulations by trial ──────────────────────────
    sims_by_trial: dict[tuple[int, int | None], list[dict]] = defaultdict(list)
    use_job_id = any("job_id" in sim for sim in simulations)
    for sim in simulations:
        key = (
            int(sim.get("trial_idx", 0)),
            int(sim["job_id"]) if use_job_id and sim.get("job_id") is not None else None,
        )
        sims_by_trial[key].append(sim)

    # Pre-sort each trial group once; keep parallel arrays for fast bisect.
    # Two sorted arrays are maintained:
    #   real_starts — used for both lo (bisect_right) and hi (bisect_left)
    #   real_ends   — used to compute lo; assumed roughly monotone when
    #                 injection gaps >> waveform duration (true for GW searches).
    sorted_sims: dict[tuple[int, int | None], tuple[list[float], list[float], list[dict]]] = {}
    for trial, sims in sims_by_trial.items():
        sims.sort(key=lambda s: s["real_start"])
        sorted_sims[trial] = (
            [s["real_start"] for s in sims],  # bisect key array (lower + upper bound)
            [s["real_end"]   for s in sims],
            sims,
        )

    # Track matched simulations by object identity
    matched_sim_ids: set[int] = set()

    # ── 2. Match each trigger ────────────────────────────────────────────
    matches: list[tuple] = []
    n_unmatched_triggers = 0

    for trig in triggers:
        trigger_start, trigger_stop = _trigger_window(trig)
        t_start = trigger_start - window_buffer
        t_stop  = trigger_stop + window_buffer
        trial   = int(trig.trial_idx)
        trig_job = int(getattr(trig, "job_id")) if use_job_id and getattr(trig, "job_id", None) is not None else None
        key = (trial, trig_job)

        found = False
        if key in sorted_sims:
            real_starts, real_ends, sims = sorted_sims[key]

            # Upper bound: sims with real_start >= t_stop cannot overlap.
            hi = bisect.bisect_left(real_starts, t_stop)
            # Lower bound: sims with real_start > t_start definitely have
            # real_start > t_start so we only need to back up by 1 to catch
            # any sim that starts just before t_start but ends after it.
            lo = max(0, bisect.bisect_right(real_starts, t_start) - 1)

            for j in range(lo, hi):
                if real_ends[j] > t_start:
                    matches.append((trig, sims[j]))
                    matched_sim_ids.add(id(sims[j]))
                    found = True

        if not found:
            n_unmatched_triggers += 1
            if how in ("left", "outer"):
                matches.append((trig, None))

    # ── 3. Append unmatched simulations ─────────────────────────────────
    n_unmatched_sims = 0
    if how in ("right", "outer"):
        for sim in simulations:
            if id(sim) not in matched_sim_ids:
                matches.append((None, sim))
                n_unmatched_sims += 1

    logger.debug(
        "match_triggers_to_simulations (how=%s): %d triggers, %d sims → "
        "%d matched pairs, %d unmatched triggers, %d unmatched sims",
        how, len(triggers), len(simulations),
        sum(1 for t, s in matches if t is not None and s is not None),
        n_unmatched_triggers, n_unmatched_sims,
    )
    return matches


# ---------------------------------------------------------------------------
# Parquet / DuckDB interval join
# ---------------------------------------------------------------------------

def match_simulations_parquet(
    catalog_parquet: str,
    sim_parquet: str,
    *,
    window_buffer: float = 0.0,
    extra_sim_columns: list[str] | None = None,
    how: str = "inner",
    output_parquet: str | None = None,
) -> "pa.Table":
    """Join a trigger catalog with a simulation summary via a DuckDB interval join.

    Both tables are read directly from Parquet — no data is loaded into Python
    memory before the join.  DuckDB's columnar reader pushes the trial equality
    predicate as a partition filter and uses a sort-merge join on the interval
    overlap predicates.

    Matching SQL (simplified)::

        SELECT t.*, s.col1 AS sim_col1, s.col2 AS sim_col2, ...
        FROM   triggers   t
        <JOIN> simulations s
          ON   t.trial_idx = s.trial_idx
         AND   trigger_start - buffer  <  s.real_end
         AND   trigger_stop  + buffer  >  s.real_start

    where ``<JOIN>`` is ``INNER``, ``LEFT``, ``RIGHT``, or ``FULL OUTER``
    depending on *how*.

    Parameters
    ----------
    catalog_parquet:
        Path to the trigger catalog Parquet file.
    sim_parquet:
        Path to the simulation summary Parquet file
        (produced by ``pycwb simulation-summary``).
    window_buffer:
        Optional time buffer (seconds) added symmetrically to the trigger
        window before comparing.
    extra_sim_columns:
        Deprecated — all simulation columns are now included automatically
        with a ``sim_`` prefix.  This argument is silently ignored.
    how : {"inner", "left", "right", "outer"}
        Join type:

        * ``"inner"`` *(default)* — only matched pairs.
        * ``"left"``  — all triggers; simulation columns are ``NULL`` for
          unmatched triggers.
        * ``"right"`` — all simulations; trigger columns are ``NULL`` for
          unmatched simulations.
        * ``"outer"`` — all triggers and all simulations; ``NULL`` on the
          unmatched side.
    output_parquet : str, optional
        If provided, the result table is written to this path as a Parquet
        file (snappy-compressed) in addition to being returned.

    Returns
    -------
    pyarrow.Table
        One row per (trigger, simulation) pair according to *how*.  Contains
        all trigger columns plus every simulation column prefixed with
        ``sim_`` (e.g. ``sim_idx``, ``sim_real_start``, ``sim_trial_idx``, …).

    Raises
    ------
    ImportError
        When ``duckdb`` is not installed.
    """
    if how not in ("inner", "left", "right", "outer"):
        raise ValueError(f"how must be 'inner', 'left', 'right', or 'outer'; got {how!r}")

    try:
        import duckdb
    except ImportError as exc:
        raise ImportError(
            "duckdb is required for match_simulations_parquet(); "
            "install with: pip install duckdb"
        ) from exc

    import pyarrow as _pa
    import pyarrow.parquet as _pq
    import pyarrow.compute as _pc

    buf = float(window_buffer)

    # Read schemas to build SELECT lists and typed NULL columns later.
    trig_schema = _pq.read_schema(catalog_parquet)
    sim_schema  = _pq.read_schema(sim_parquet)

    sim_cols   = [f"s.{c} AS sim_{c}" for c in sim_schema.names]
    sim_select = ", ".join(sim_cols)
    trigger_start_expr, trigger_stop_expr = _parquet_trigger_window_exprs(trig_schema)
    job_join = ""
    if "job_id" in trig_schema.names:
        if "job_id" in sim_schema.names:
            job_join = "AND   t.job_id     =  s.job_id"
        elif "segment_idx" in sim_schema.names:
            logger.warning(
                "Simulation summary %s has segment_idx but no job_id; matching "
                "will use trial/window only. Regenerate it with job_id/shift for "
                "shift-aware matching.",
                sim_parquet,
            )

    # ── Step 1: inner join with 1-to-1 deduplication via QUALIFY ────────────
    #
    # The interval join is inherently many-to-many when the simulation table
    # contains multiple rows at the same injection GPS time (e.g. different
    # hrss amplitude levels) or when a large window_buffer causes a trigger to
    # overlap more than one injection window.
    #
    # We resolve this with two QUALIFY passes:
    #   a) Per sim  → keep the trigger with the highest rho (most significant
    #                 detection); ties broken by minimum |gps_time| distance,
    #                 then by trigger id for determinism.
    #   b) Per trigger → keep the sim with the minimum |gps_time| distance;
    #                 ties broken by sim_idx for determinism.
    #
    # This guarantees that every trigger and every simulation appears at most
    # once in the matched set, so the outer-join row count equals
    #   N_triggers + N_simulations − N_matched_pairs.
    inner_sql = f"""
        WITH step1 AS (
            SELECT t.*, {sim_select},
                   ABS(t.gps_time - s.gps_time) AS _gps_dist
            FROM   read_parquet('{catalog_parquet}') t
            INNER JOIN read_parquet('{sim_parquet}') s
              ON   t.trial_idx  =  s.trial_idx
             {job_join}
             AND   {trigger_start_expr} - {buf}  <  s.real_end
             AND   {trigger_stop_expr} + {buf}  >  s.real_start
            QUALIFY
                ROW_NUMBER() OVER (
                    PARTITION BY sim_sim_idx
                    ORDER BY rho DESC NULLS LAST, _gps_dist, id
                ) = 1
        ),
        step2 AS (
            SELECT *
            FROM   step1
            QUALIFY
                ROW_NUMBER() OVER (
                    PARTITION BY id
                    ORDER BY _gps_dist, sim_sim_idx
                ) = 1
        )
        SELECT * EXCLUDE (_gps_dist) FROM step2
    """
    logger.debug("match_simulations_parquet inner SQL:\n%s", inner_sql)

    matched = duckdb.query(inner_sql).arrow()
    if hasattr(matched, "read_all"):
        matched = matched.read_all()

    n_matched = matched.num_rows
    logger.debug("match_simulations_parquet: %d 1-to-1 matched pairs", n_matched)

    if how == "inner":
        result = matched
    else:
        # ── Step 2: assemble unmatched rows ─────────────────────────────────
        #
        # Build the full output schema once from the matched table (trigger cols
        # first, then sim_* cols).  Unmatched rows get typed NULL arrays for
        # the side that has no partner.
        out_schema = matched.schema

        def _null_cols_for(schema_side: "_pa.Schema", prefix: str = "") -> list:
            """Return a list of (name, null_array) for columns not in out_schema."""
            cols = []
            for field in schema_side:
                col_name = f"{prefix}{field.name}"
                if col_name not in out_schema.names:
                    continue
                out_field = out_schema.field(col_name)
                cols.append((col_name, _pa.array([None] * 0, type=out_field.type)))
            return cols

        parts = [matched]

        if how in ("left", "outer"):
            cat_table = _pq.read_table(catalog_parquet)
            if n_matched > 0:
                matched_t_ids = matched.column("id")
                mask = _pc.invert(_pc.is_in(cat_table.column("id"),
                                             value_set=matched_t_ids))
                unmatched_t = cat_table.filter(mask)
            else:
                unmatched_t = cat_table

            # Add null sim_* columns to unmatched trigger rows.
            for field in sim_schema:
                col_name = f"sim_{field.name}"
                if col_name in out_schema.names:
                    out_type = out_schema.field(col_name).type
                    unmatched_t = unmatched_t.append_column(
                        col_name,
                        _pa.array([None] * len(unmatched_t), type=out_type),
                    )
            unmatched_t = unmatched_t.select(out_schema.names)
            parts.append(unmatched_t)
            logger.debug("match_simulations_parquet: %d unmatched triggers", len(unmatched_t))

        if how in ("right", "outer"):
            sim_table = _pq.read_table(sim_parquet)
            if n_matched > 0:
                matched_s_idxs = matched.column("sim_sim_idx")
                mask = _pc.invert(_pc.is_in(sim_table.column("sim_idx"),
                                             value_set=matched_s_idxs))
                unmatched_s = sim_table.filter(mask)
            else:
                unmatched_s = sim_table

            # Rename sim columns with sim_ prefix, then add null trigger columns.
            unmatched_s = unmatched_s.rename_columns(
                [f"sim_{c}" for c in unmatched_s.schema.names]
            )
            for field in trig_schema:
                if field.name in out_schema.names:
                    out_type = out_schema.field(field.name).type
                    unmatched_s = unmatched_s.append_column(
                        field.name,
                        _pa.array([None] * len(unmatched_s), type=out_type),
                    )
            unmatched_s = unmatched_s.select(out_schema.names)
            parts.append(unmatched_s)
            logger.debug("match_simulations_parquet: %d unmatched sims", len(unmatched_s))

        result = _pa.concat_tables(parts)

    if output_parquet is not None:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(output_parquet)) or ".", exist_ok=True)
        _pq.write_table(result, output_parquet, compression="snappy")
        logger.info("match_simulations_parquet: wrote %d rows to %s",
                    result.num_rows, output_parquet)

    logger.info(
        "match_simulations_parquet (how=%s): %d triggers, %d sims → "
        "%d matched pairs, %d total rows",
        how,
        _pq.read_metadata(catalog_parquet).num_rows,
        _pq.read_metadata(sim_parquet).num_rows,
        n_matched,
        result.num_rows,
    )
    return result
