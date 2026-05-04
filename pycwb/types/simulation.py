"""
pycwb.types.simulation
=======================

Simulation / injection-related types and Parquet schema utilities.

This module consolidates:

* :class:`InjectionParams` — the dataclass describing a single injected signal
  (previously in ``pycwb.types.trigger``).
* Schema-inference helpers for writing flat simulation-summary Parquet tables.

Design rationale
----------------
Different waveform families carry completely different fields — WNB has
``frequency``, ``bandwidth``, ``duration``; BBH has ``mass1``, ``mass2``,
``spin1z``, ``distance``, etc.  Instead of hard-coding these fields or burying
them in an opaque JSON string, :func:`infer_injection_fields` scans the actual
injection dicts at runtime and produces a flat Arrow schema where every
discovered field becomes its own typed column.  Rows that lack a field get a
null for that column, which Parquet encodes efficiently.

Typical usage
-------------
::

    from pycwb.types.simulation import (
        InjectionParams,
        infer_injection_fields,
        rows_to_flat_table,
        build_fixed_schema,
    )

    # Build an InjectionParams from a raw injection dict
    inj = InjectionParams.from_injection_dict(inj_dict)

    # Collect raw injection dicts (from config.injection or seg.injections)
    raw_dicts = [sim for seg in job_segments for sim in (seg.injections or [])]

    # Infer Arrow types from the actual values — returns {field: pa.DataType}
    extra_fields = infer_injection_fields(raw_dicts)

    # Build a flat PyArrow Table (one row per injection, one column per field)
    table = rows_to_flat_table(rows, extra_fields, build_fixed_schema())

    import pyarrow.parquet as pq
    pq.write_table(table, "simulation_summary.parquet")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa


# ---------------------------------------------------------------------------
# InjectionParams
# ---------------------------------------------------------------------------

@dataclass
class InjectionParams:
    """Simulation-only parameters describing the injected signal.

    Storage is two-tier:

    * **Common typed fields** (``name``, ``hrss``, ``target_snr``, ``ra``, ``dec``,
      ``gps_time``, ``pol``, ``approximant``) — always-present typed Arrow columns
      that enable fast Parquet predicate pushdown and work with :meth:`Catalog.filter`.
    * **Per-IFO injection-matched fields** (``snr_sq``, ``rec_snr_sq``,
      ``overlap_snr``, ``d_eff``, ``fp``, ``fx``, ``time``, ``hrss_det``) — the
      injection-side counterparts of the same-named ``Trigger`` per-IFO columns,
      sourced from the second half (``ifo+NIFO``) of the legacy Event arrays and
      from cWB's injection-comparison output (``iSNR``, ``oSNR``, ``ioSNR``,
      ``Deff``).  These have **no equivalent** in the ``Trigger`` for background
      triggers.
    * **Flexible parameters** (``parameters``) — JSON-encoded string of the
      **complete** original user injection dict.  Preserves all waveform-specific
      fields (``frequency``, ``Q``, ``bandwidth`` for WNB/SGE; ``mass1``,
      ``mass2``, ``spin1z`` for CBC, …).  Queryable via DuckDB
      ``json_extract`` / ``json_extract_string``::

          cat.query(\"\"\"
              SELECT injection.name, injection.hrss,
                     json_extract(injection.parameters, '$.frequency')::FLOAT AS freq
              FROM   triggers
              WHERE  injection IS NOT NULL
          \"\"\")

    ``None`` for background triggers.
    """

    # --- Common typed fields (fast Parquet filtering) --------------------

    name: str = ""
    """Injection waveform name, e.g. ``"WNB17b_0_0"``, ``"SGE_Q9_70Hz"``."""

    hrss: float = 0.0
    """Network total injected hrss: sqrt(Σ hrss²) over detectors."""

    target_snr: float = 0.0
    """Target network SNR (0 if not applicable)."""

    ra: float = 0.0
    """Injected Right Ascension (radians)."""

    dec: float = 0.0
    """Injected Declination (radians)."""

    gps_time: float = 0.0
    """Injection GPS time (geocentric end time)."""

    pol: float = 0.0
    """Injected polarisation angle (radians)."""

    approximant: str = ""
    """Waveform approximant, e.g. ``"WNB"``, ``"SGE"``, ``"IMRPhenomTPHM"``."""

    # --- Per-IFO injection-matched fields --------------------------------
    # Injection-side counterparts of the Trigger's own per-IFO columns,
    # computed at the *injected* sky position / arrival time, not the
    # reconstructed one.

    snr_sq: list[float] = field(default_factory=list)
    """Injected SNR² per detector  (legacy ``iSNR[ifo]``)."""

    rec_snr_sq: list[float] = field(default_factory=list)
    """Reconstructed SNR² inside injection window per detector
    (legacy ``oSNR[ifo]``)."""

    overlap_snr: list[float] = field(default_factory=list)
    """Cross-correlation (injected ∩ reconstructed) per detector
    (legacy ``ioSNR[ifo]``).  Residual energy ∝ ``snr_sq + rec_snr_sq − 2·overlap_snr``."""

    d_eff: list[float] = field(default_factory=list)
    """Effective distance per detector (Mpc)  (legacy ``Deff[ifo]``)."""

    fp: list[float] = field(default_factory=list)
    """Antenna response F₊ at the *injected* sky position per detector
    (legacy ``bp[ifo+NIFO]``)."""

    fx: list[float] = field(default_factory=list)
    """Antenna response F× at the *injected* sky position per detector
    (legacy ``bx[ifo+NIFO]``)."""

    time: list[float] = field(default_factory=list)
    """GPS arrival time of the *injection* at each detector
    (legacy ``time[ifo+NIFO]``)."""

    hrss_det: list[float] = field(default_factory=list)
    """Injected hrss at each detector."""

    # --- Flexible parameters (JSON blob) ---------------------------------

    parameters: str = ""
    """JSON-encoded dict of the complete original user injection parameters.

    Preserves all waveform-specific fields that do not fit a fixed schema
    (e.g. ``frequency``, ``Q``, ``bandwidth``, ``duration`` for WNB/SGE;
    ``mass1``, ``mass2``, ``spin1z`` for CBC; legacy cWB scalar fields for
    ROOT-converted triggers).  Query with DuckDB::

        cat.query(\"\"\"
            SELECT injection.name,
                   json_extract(injection.parameters, '$.frequency')::FLOAT AS freq,
                   json_extract(injection.parameters, '$.Q')::FLOAT         AS Q
            FROM   triggers
            WHERE  injection.approximant = 'SGE'
        \"\"\")
    """

    def to_dict(self) -> dict:
        """Return a flat dict for Arrow struct serialisation."""
        return {
            "name":        self.name,
            "hrss":        float(self.hrss),
            "target_snr":  float(self.target_snr),
            "ra":          float(self.ra),
            "dec":         float(self.dec),
            "gps_time":    float(self.gps_time),
            "pol":         float(self.pol),
            "approximant": self.approximant,
            "snr_sq":      [float(v) for v in self.snr_sq],
            "rec_snr_sq":  [float(v) for v in self.rec_snr_sq],
            "overlap_snr": [float(v) for v in self.overlap_snr],
            "d_eff":       [float(v) for v in self.d_eff],
            "fp":          [float(v) for v in self.fp],
            "fx":          [float(v) for v in self.fx],
            "time":        [float(v) for v in self.time],
            "hrss_det":    [float(v) for v in self.hrss_det],
            "parameters":  self.parameters,
        }

    @staticmethod
    def arrow_struct() -> pa.StructType:
        """Return the PyArrow struct type for :class:`InjectionParams`."""
        lf32 = pa.list_(pa.float32())
        lf64 = pa.list_(pa.float64())
        return pa.struct([
            # common typed fields
            pa.field("name",         pa.string()),
            pa.field("hrss",         pa.float32()),
            pa.field("target_snr",   pa.float32()),
            pa.field("ra",           pa.float32()),
            pa.field("dec",          pa.float32()),
            pa.field("gps_time",     pa.float64()),
            pa.field("pol",          pa.float32()),
            pa.field("approximant",  pa.string()),
            # per-IFO injection-matched fields
            pa.field("snr_sq",       lf32),
            pa.field("rec_snr_sq",   lf32),
            pa.field("overlap_snr",  lf32),
            pa.field("d_eff",        lf32),
            pa.field("fp",           lf32),
            pa.field("fx",           lf32),
            pa.field("time",         lf64),
            pa.field("hrss_det",     lf32),
            # flexible parameters
            pa.field("parameters",   pa.string()),
        ])

    @classmethod
    def from_injection_dict(cls, inj_dict: dict) -> "InjectionParams":
        """Create an :class:`InjectionParams` from a raw user injection dict.

        Extracts the common typed fields by well-known key names; serialises
        the **complete** dict as a JSON string into :attr:`parameters`.

        Parameters
        ----------
        inj_dict : dict
            The injection parameter dict as produced by
            ``injection_parameters.py`` and enriched by the pipeline
            (``ra``, ``dec``, ``gps_time`` added by sky/time distribution).
        """
        inj = cls()
        inj.name        = str(inj_dict.get("name", ""))
        inj.hrss        = float(inj_dict.get("hrss", 0.0))
        inj.target_snr  = float(inj_dict.get("target_snr",
                                inj_dict.get("targeted_snr", 0.0)))
        inj.ra          = float(inj_dict.get("ra", 0.0))
        inj.dec         = float(inj_dict.get("dec", 0.0))
        inj.gps_time    = float(inj_dict.get("gps_time", 0.0))
        inj.pol         = float(inj_dict.get("pol", 0.0))
        inj.approximant = str(inj_dict.get("approximant", ""))
        try:
            inj.parameters = json.dumps(inj_dict, default=str)
        except (TypeError, ValueError):
            inj.parameters = "{}"
        return inj


# ---------------------------------------------------------------------------
# Parquet schema-inference utilities
# ---------------------------------------------------------------------------

# Fields that the simulation_summary already promotes to dedicated top-level
# columns.  These are NOT added again when we expand injection parameters.
_ALREADY_CAPTURED = frozenset({
    "gps_time", "name", "hrss", "ra", "dec", "pol",
    "approximant", "trial_idx", "target_snr", "targeted_snr",
    # synthetic fields that never appear in the raw injection dict
    "real_start", "real_end",
})


def _infer_arrow_type(values: list[Any]) -> pa.DataType:
    """Return the tightest Arrow scalar type that fits *values*.

    Rules (checked in order):
    - All non-null bools       → ``pa.bool_()``
    - All non-null ints        → ``pa.int64()``
    - All non-null ints/floats → ``pa.float64()``
    - Homogeneous lists        → ``pa.list_(element_type)``
    - Anything else            → ``pa.string()``
    """
    non_null = [v for v in values if v is not None]
    if not non_null:
        # No information — default to string; will be all-null
        return pa.string()

    # Check lists first (before bool/int/float since list is a container)
    if all(isinstance(v, list) for v in non_null):
        # Infer element type from all list elements across all rows
        elements = [e for row in non_null for e in row if e is not None]
        elem_type = _infer_arrow_type(elements) if elements else pa.float64()
        return pa.list_(elem_type)

    if all(isinstance(v, bool) for v in non_null):
        return pa.bool_()

    if all(isinstance(v, int) and not isinstance(v, bool) for v in non_null):
        return pa.int64()

    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null):
        return pa.float64()

    return pa.string()


def infer_injection_fields(
    injection_dicts: list[dict],
    skip_keys: frozenset[str] | None = None,
) -> dict[str, pa.DataType]:
    """Scan *injection_dicts* and return a ``{field_name: arrow_type}`` mapping.

    Fields listed in *skip_keys* (default: :data:`_ALREADY_CAPTURED`) are
    excluded — they are already promoted to dedicated columns elsewhere.

    Parameters
    ----------
    injection_dicts:
        Raw injection parameter dicts, e.g. from
        ``[sim for seg in job_segments for sim in (seg.injections or [])]``.
    skip_keys:
        Keys to exclude from the inferred schema.  Defaults to
        :data:`_ALREADY_CAPTURED`.

    Returns
    -------
    dict[str, pa.DataType]
        Ordered dict (insertion order = first-seen order across all dicts)
        mapping field name to its inferred Arrow type.
    """
    if skip_keys is None:
        skip_keys = _ALREADY_CAPTURED

    # Collect all values per key across every dict
    values_per_key: dict[str, list[Any]] = {}
    for d in injection_dicts:
        for k, v in d.items():
            if k in skip_keys:
                continue
            values_per_key.setdefault(k, []).append(v)

    return {k: _infer_arrow_type(vals) for k, vals in values_per_key.items()}


def rows_to_flat_table(
    rows: list[dict],
    extra_fields: dict[str, pa.DataType],
    fixed_schema: pa.Schema | None = None,
) -> pa.Table:
    """Convert *rows* to a flat PyArrow Table.

    Parameters
    ----------
    rows:
        List of row dicts as produced by :func:`build_simulation_summary`.
        Each row may contain a ``"parameters"`` key (dict of raw injection
        fields) which is exploded into individual columns.
    extra_fields:
        Mapping from field name to Arrow type, as returned by
        :func:`infer_injection_fields`.  These become the extra flat columns.
    fixed_schema:
        Optional Arrow schema for the fixed (non-injection) columns.  When
        provided the table is cast to this schema before adding the extra
        columns, ensuring consistent dtypes.  When ``None`` the dtypes are
        inferred from the Python values.

    Returns
    -------
    pa.Table
        Flat table — no nested structs, no JSON blobs.
    """
    # Separate fixed columns from the injection parameter blob
    fixed_rows: list[dict] = []
    for row in rows:
        fixed = {k: v for k, v in row.items() if k != "parameters"}
        fixed_rows.append(fixed)

    # Build the fixed-column table
    fixed_table = pa.Table.from_pylist(fixed_rows, schema=fixed_schema)

    # Append one column per extra injection field
    for field_name, arrow_type in extra_fields.items():
        values = []
        for row in rows:
            params = row.get("parameters") or {}
            values.append(params.get(field_name))
        # Cast to the inferred type; use a list array so nulls are preserved
        try:
            col = pa.array(values, type=arrow_type)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            # Fall back to string if type inference was wrong for some values
            col = pa.array([str(v) if v is not None else None for v in values],
                           type=pa.string())
        fixed_table = fixed_table.append_column(
            pa.field(field_name, arrow_type), col
        )

    return fixed_table


def build_fixed_schema(ifo_list: list[str] | None = None) -> pa.Schema:
    """Return the Arrow schema for the fixed simulation-summary columns.

    These are the columns that are always present regardless of injection type.

    Parameters
    ----------
    ifo_list:
        Not used currently; reserved for future per-IFO columns.
    """
    return pa.schema([
        pa.field("sim_idx",          pa.int64()),
        pa.field("trial_idx",        pa.int64()),
        pa.field("gps_time",         pa.float64()),
        pa.field("real_start",       pa.float64(),   nullable=True),
        pa.field("real_end",         pa.float64(),   nullable=True),
        pa.field("real_duration",    pa.float64(),   nullable=True),
        pa.field("segment_idx",      pa.int64()),
        pa.field("vetoed_cat0",      pa.bool_(),     nullable=True),
        pa.field("vetoed_cat1",      pa.bool_(),     nullable=True),
        pa.field("vetoed_cat2",      pa.bool_(),     nullable=True),
        pa.field("across_segments",  pa.bool_(),     nullable=True),
        pa.field("error",            pa.string(),    nullable=True),
    ])
