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
* ``min(T.event_start) < S.real_end``   — trigger starts before injection ends
* ``max(T.event_stop)  > S.real_start`` — trigger ends after injection starts

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
    sims_by_trial: dict[int, list[dict]] = defaultdict(list)
    for sim in simulations:
        sims_by_trial[int(sim.get("trial_idx", 0))].append(sim)

    # Pre-sort each trial group once; keep parallel arrays for fast bisect.
    # Two sorted arrays are maintained:
    #   real_starts — used for both lo (bisect_right) and hi (bisect_left)
    #   real_ends   — used to compute lo; assumed roughly monotone when
    #                 injection gaps >> waveform duration (true for GW searches).
    sorted_sims: dict[int, tuple[list[float], list[float], list[dict]]] = {}
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
        t_start = (min(trig.event_start) if trig.event_start else trig.gps_time) - window_buffer
        t_stop  = (max(trig.event_stop)  if trig.event_stop  else trig.gps_time) + window_buffer
        trial   = int(trig.trial_idx)

        found = False
        if trial in sorted_sims:
            real_starts, real_ends, sims = sorted_sims[trial]

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
         AND   t.gps_time - buffer  <  s.real_end
         AND   t.gps_time + buffer  >  s.real_start

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

    _join_keywords = {
        "inner": "INNER JOIN",
        "left":  "LEFT JOIN",
        "right": "RIGHT JOIN",
        "outer": "FULL OUTER JOIN",
    }
    join_kw = _join_keywords[how]
    buf = float(window_buffer)

    # Read sim schema to build a full sim_* SELECT list automatically.
    import pyarrow.parquet as _pq
    sim_schema_names = _pq.read_schema(sim_parquet).names

    # Include every sim column, prefixed with sim_ to avoid clashes with trigger columns.
    sim_cols = [f"s.{c} AS sim_{c}" for c in sim_schema_names]
    sim_select = ", ".join(sim_cols)

    sql = f"""
        SELECT t.*, {sim_select}
        FROM   read_parquet('{catalog_parquet}')  t
        {join_kw}
               read_parquet('{sim_parquet}')       s
          ON   t.trial_idx  =  s.trial_idx
         AND   t.gps_time - {buf}  <  s.real_end
         AND   t.gps_time + {buf}  >  s.real_start
    """
    logger.debug("match_simulations_parquet SQL:\n%s", sql)

    result = duckdb.query(sql).arrow()
    # DuckDB ≥ 1.0 returns a RecordBatchReader; materialise to Table
    if hasattr(result, "read_all"):
        result = result.read_all()

    if output_parquet is not None:
        import pyarrow.parquet as pq
        import os
        os.makedirs(os.path.dirname(os.path.abspath(output_parquet)) or ".", exist_ok=True)
        pq.write_table(result, output_parquet, compression="snappy")
        logger.info("match_simulations_parquet: wrote %d rows to %s", result.num_rows, output_parquet)

    logger.info(
        "match_simulations_parquet: %d matched (trigger, simulation) pairs",
        result.num_rows,
    )
    return result
