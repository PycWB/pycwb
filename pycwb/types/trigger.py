"""
pycwb.types.trigger
====================

Defines the :class:`Trigger` dataclass – the canonical in-memory representation of a
single reconstructed gravitational-wave transient candidate produced by the cWB pipeline.

Design goals
------------
* **Self-describing names** – every field has a physics-meaningful name; no positional
  arrays (``rho[0]``, ``netcc[2]``, …) survive into user-facing code.
* **PyArrow / Parquet compatible** – :meth:`Trigger.arrow_schema` returns the matching
  ``pa.Schema``; :meth:`Trigger.to_arrow_dict` returns a flat dict ready for
  ``pa.Table.from_pylist`` or ``pa.RecordBatch``.
* **Variable-size network** – per-IFO quantities are stored as ``list[float]`` in the
  same order as :attr:`ifo_list`.  The Arrow schema uses ``pa.list_(pa.float32())``
  columns so the schema is network-size-independent.
* **Injection parameters isolated** – simulation-only fields live in
  :class:`InjectionParams`, which serialises to a nullable Arrow ``struct`` column.
  Background-run triggers carry ``injection=None``.

See ``docs/dev/TRIGGER_CLASS_DESIGN.md`` for the full design rationale and the
old-name → new-name cross-reference table.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, fields as dc_fields
from typing import Optional

# ---------------------------------------------------------------------------
# InjectionParams  (simulation-only sub-structure)
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
    def arrow_struct():
        """Return the PyArrow struct type for :class:`InjectionParams`."""
        import pyarrow as pa
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
        import json as _json
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
            inj.parameters = _json.dumps(inj_dict, default=str)
        except (TypeError, ValueError):
            inj.parameters = "{}"
        return inj


# ---------------------------------------------------------------------------
# Per-IFO field name registries
# Ordered tuples used by arrow_schema() and to_arrow_dict() to generate flat
# ``{field}_{ifo}`` columns when an ifo_list is supplied.
# ---------------------------------------------------------------------------

#: Per-IFO timing fields stored at float64 precision (GPS times).
_PER_IFO_F64: tuple = (
    "time", "segment_start", "event_start", "event_stop",
)

#: Per-IFO fields stored at float32 precision.
_PER_IFO_F32: tuple = (
    "left_edge", "right_edge", "duration", "time_lag", "segment_lag",
    "central_freq", "freq_low", "freq_high", "bandwidth", "sample_rate",
    "hrss", "noise_rms", "data_energy", "signal_energy", "cross_energy",
    "null_energy", "residual_energy", "fp", "fx",
)


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------

@dataclass
class Trigger:
    """A single reconstructed GW burst candidate from the cWB pipeline.

    Fields are grouped into four categories:

    1. **Identity / bookkeeping** – IDs, job info, IFO list.
    2. **Network scalar statistics** – single float quantities describing the
       full-network coherence, energy, and sky position.
    3. **Per-IFO lists** – one value per detector, ordered by :attr:`ifo_list`.
    4. **Injection parameters** – :class:`InjectionParams` (``None`` for background runs).

    See ``docs/dev/TRIGGER_CLASS_DESIGN.md`` for the complete old→new name mapping.
    """

    # ------------------------------------------------------------------
    # 1. Identity / bookkeeping
    # ------------------------------------------------------------------

    id: str = ""
    """Unique string identifier derived from stop time + hash."""

    job_id: int = 0
    """Job segment index  (legacy ``run`` / ``job_id``)."""

    lag_idx: int = 0
    """Index of the time-lag used when this trigger was produced."""

    trial_idx: int = 0
    """Injection trial index (0 for background / no-injection runs)."""

    cluster_id: int = 0
    """Cluster ID within the job segment  (legacy ``eventID[0]``)."""

    event_index: int = 0
    """Sequential trigger number within the job  (legacy ``nevent``)."""

    n_detectors: int = 0
    """Number of detectors in the network  (legacy ``ndim``)."""

    ifo_list: list[str] = field(default_factory=list)
    """Ordered list of detector names, e.g. ``["H1", "L1", "V1"]``."""

    hybrid: bool = False
    """True when the event was produced by the legacy CWB-based (hybrid) pipeline."""

    # ------------------------------------------------------------------
    # 2a. Network coherent SNR and correlation
    # ------------------------------------------------------------------

    rho: float = 0.0
    """Effective correlated SNR: sqrt(Ec/(K-1)) in 2G  (legacy ``rho[0]``)."""

    rho_alt: float = 0.0
    """Raw rho before cc reduction (netrho), or rho·chirp_factor for non-pat0
    (legacy ``rho[1]``)."""

    net_cc: float = 0.0
    """MRA network correlation coefficient Ec/(|Ec|+N)  (legacy ``netcc[0]``)."""

    sky_cc: float = 0.0
    """All-resolution cc statistic  (legacy ``netcc[1]``)."""

    subnet_cc: float = 0.0
    """MRA sub-network statistic  (legacy ``netcc[2]``)."""

    subnet_cc2: float = 0.0
    """All-resolution sub-network statistic  (legacy ``netcc[3]``)."""

    # ------------------------------------------------------------------
    # 2b. Energy statistics
    # ------------------------------------------------------------------

    likelihood: float = 0.0
    """Network waveform likelihood L = Σ|s[i]|²."""

    coherent_energy: float = 0.0
    """Sum of off-diagonal likelihood matrix elements Ec  (legacy ``ecor``)."""

    coherent_energy_norm: float = 0.0
    """Normalised coherent energy ECOR = Σ Lij·rij  (legacy ``ECOR``)."""

    net_energy_disb: float = 0.0
    """Network energy disbalance ED  (legacy ``neted[0]``)."""

    net_null: float = 0.0
    """Total null energy with Gaussian bias correction  (legacy ``neted[1]``)."""

    net_energy: float = 0.0
    """Total event energy  (legacy ``neted[2]``)."""

    like_sky: float = 0.0
    """Total likelihood at all sky/resolutions  (legacy ``neted[3]``)."""

    energy_sky: float = 0.0
    """Total energy at all sky/resolutions  (legacy ``neted[4]``)."""

    # ------------------------------------------------------------------
    # 2c. Network quality factors
    # ------------------------------------------------------------------

    network_sensitivity: float = 0.0
    """Network sensitivity factor  (legacy ``gnet``)."""

    network_alignment_factor: float = 0.0
    """Network alignment factor  (legacy ``anet``)."""

    network_index: float = 0.0
    """Network index  (legacy ``inet``)."""

    packet_norm: float = 0.0
    """Packet norm: ratio of total energy to reconstructed energy across resolution
    levels  (legacy ``norm``)."""

    penalty: float = 0.0
    """Cluster chi²/nDoF penaly factor  (legacy ``penalty``)."""

    cluster_union_size: float = 0.0
    """Cluster union size  (legacy ``usize``)."""

    strain: float = 0.0
    """Network hrss amplitude: sqrt(Σ hrss²) over detectors  (legacy ``strain[0]``)."""

    # ------------------------------------------------------------------
    # 2d. Pixel / cluster counts
    # ------------------------------------------------------------------

    n_pixels_total: int = 0
    """Total TF pixels entering the likelihood stage  (legacy ``volume[0]``)."""

    n_pixels_positive: int = 0
    """Pixels with coherent energy > 0  (legacy ``volume[1]``)."""

    n_pixels_core: int = 0
    """Core pixels with energy > Acore used in reconstruction  (legacy ``size[0]``)."""

    sky_size: int = 0
    """Pixels contributing to sky map  (legacy ``size[1]``)."""

    # ------------------------------------------------------------------
    # 2e. Sky localisation (named scalars — no more positional phi/theta arrays)
    # ------------------------------------------------------------------

    phi: float = 0.0
    """Estimated sky longitude in cWB Earth coordinates (degrees)  (legacy ``phi[0]``)."""

    theta: float = 0.0
    """Estimated sky co-latitude in cWB Earth coordinates (degrees)  (legacy ``theta[0]``)."""

    ra: float = 0.0
    """Estimated Right Ascension (equatorial degrees)  (legacy ``phi[2]``)."""

    dec: float = 0.0
    """Estimated Declination (equatorial degrees)  (legacy ``theta[2]``)."""

    phi_det: float = 0.0
    """Sky longitude derived from detection statistics  (legacy ``phi[3]``)."""

    theta_det: float = 0.0
    """Sky co-latitude derived from detection statistics  (legacy ``theta[3]``)."""

    psi: float = 0.0
    """Estimated polarisation angle (radians)  (legacy ``psi[0]``)."""

    iota: float = 0.0
    """Estimated inclination angle  (legacy ``iota[0]``)."""

    sky_error_regions: list[float] = field(default_factory=list)
    """Sky credible interval areas (sqrt of area in sr²): 11 entries.

    * ``sky_error_regions[0]``: sqrt(area) enclosing injected position (simulation only).
    * ``sky_error_regions[1..9]``: 10 %, 20 %, …, 90 % credible regions.
    * ``sky_error_regions[10]``: probability at injected position (simulation only).

    (legacy ``erA[0..10]``)
    """

    # ------------------------------------------------------------------
    # 2f. Chirp reconstruction
    # ------------------------------------------------------------------

    mchirp: float = 0.0
    """Reconstructed chirp mass (M☉)  (legacy ``chirp[1]``)."""

    mchirp_err: float = 0.0
    """Reconstructed chirp mass uncertainty  (legacy ``chirp[2]``)."""

    chirp_ellip: float = 0.0
    """Chirp ellipticity / amplitude factor  (legacy ``chirp[3]``)."""

    chirp_pfrac: float = 0.0
    """Fraction of pixels consistent with chirp morphology  (legacy ``chirp[4]``)."""

    chirp_efrac: float = 0.0
    """Fraction of energy consistent with chirp morphology / chi² of fit
    (legacy ``chirp[5]``)."""

    ebbh: list[float] = field(default_factory=list)
    """BBH-specific energy statistics (placeholder, CBC mode only)  (legacy ``eBBH``)."""

    # ------------------------------------------------------------------
    # 2g. Post-processing / ranking
    # ------------------------------------------------------------------

    ifar: float = 0.0
    """Inverse false-alarm rate (years)  — filled by background estimation stage."""

    q_veto: float = 0.0
    """Q-veto statistic  (legacy ``Qveto[0]``)."""

    q_factor: float = 0.0
    """Default Q-factor  (legacy ``Qveto[1]``)."""

    gps_time: float = 0.0
    """Network GPS time of the trigger (scalar copy of ``time[0]``, the reference-detector
    arrival time).  Stored as a top-level ``float64`` column so that Parquet row-group
    statistics enable efficient predicate pushdown for time-range queries."""

    # ------------------------------------------------------------------
    # 3. Per-IFO lists  (length = n_detectors, ordered by ifo_list)
    # ------------------------------------------------------------------

    # --- Timing ---
    time: list[float] = field(default_factory=list)
    """GPS time of trigger at each detector (float64 precision)  (legacy ``time[ifo]``)."""

    segment_start: list[float] = field(default_factory=list)
    """GPS start time of the job-segment data for each detector  (legacy ``gps[ifo]``)."""

    event_start: list[float] = field(default_factory=list)
    """GPS time of the earliest cluster pixel for each detector  (legacy ``start[ifo]``)."""

    event_stop: list[float] = field(default_factory=list)
    """GPS time of the latest cluster pixel for each detector  (legacy ``stop[ifo]``)."""

    left_edge: list[float] = field(default_factory=list)
    """Seconds from segment start to event start per detector  (legacy ``left[ifo]``)."""

    right_edge: list[float] = field(default_factory=list)
    """Seconds from event stop to segment end per detector  (legacy ``right[ifo]``)."""

    duration: list[float] = field(default_factory=list)
    """Event duration per detector; ``duration[0]`` is energy-weighted across all
    resolutions  (legacy ``duration[ifo]``)."""

    time_lag: list[float] = field(default_factory=list)
    """Time lag (shift) applied to each detector data stream  (legacy ``lag[ifo]``,
    first half)."""

    segment_lag: list[float] = field(default_factory=list)
    """Super-lag (segment shift) per detector  (legacy ``slag[ifo]``)."""

    # --- Frequency ---
    central_freq: list[float] = field(default_factory=list)
    """Central frequency per detector; ``central_freq[0]`` is waveform-derived
    (legacy ``frequency[ifo]``)."""

    freq_low: list[float] = field(default_factory=list)
    """Minimum TF-pixel frequency per detector  (legacy ``low[ifo]``)."""

    freq_high: list[float] = field(default_factory=list)
    """Maximum TF-pixel frequency per detector  (legacy ``high[ifo]``)."""

    bandwidth: list[float] = field(default_factory=list)
    """Bandwidth; ``bandwidth[0]`` is energy-weighted  (legacy ``bandwidth[ifo]``)."""

    sample_rate: list[float] = field(default_factory=list)
    """Wavelet decomposition rate of the reconstruction level  (legacy ``rate[ifo]``)."""

    # --- Amplitude / SNR ---
    hrss: list[float] = field(default_factory=list)
    """Reconstructed root-sum-squared strain per detector  (legacy ``hrss[ifo]``)."""

    noise_rms: list[float] = field(default_factory=list)
    """Noise floor (strain/√Hz) per detector  (legacy ``noise[ifo]``)."""

    data_energy: list[float] = field(default_factory=list)
    """Data energy Σ(wave²+w_90²) per detector (≈ SNR²)  (legacy ``snr[ifo]`` = ``enrg``)."""

    signal_energy: list[float] = field(default_factory=list)
    """Reconstructed signal energy Σ(asnr²+a_90²) per detector  (legacy ``sSNR[ifo]``)."""

    cross_energy: list[float] = field(default_factory=list)
    """Data–signal cross-energy per detector  (legacy ``xSNR[ifo]``)."""

    null_energy: list[float] = field(default_factory=list)
    """Null energy per detector  (legacy ``null[ifo]``)."""

    residual_energy: list[float] = field(default_factory=list)
    """Residual energy (xSNR − sSNR) per detector  (legacy ``nill[ifo]``)."""

    # --- Antenna patterns ---
    fp: list[float] = field(default_factory=list)
    """Antenna response F₊ at estimated sky position per detector  (legacy ``bp[ifo]``)."""

    fx: list[float] = field(default_factory=list)
    """Antenna response F× at estimated sky position per detector  (legacy ``bx[ifo]``)."""

    # ------------------------------------------------------------------
    # 4. Injection parameters  (None for background triggers)
    # ------------------------------------------------------------------

    injection: Optional[InjectionParams] = None
    """Simulation-only injection parameters.  ``None`` for background runs."""

    # ==================================================================
    # Derived / computed properties
    # ==================================================================

    @property
    def hash_id(self) -> str:
        """8-character hex hash based on stop time, start time, and frequency bounds."""
        if not self.event_stop:
            return "00000000"
        key = f"{self.event_stop[0]}_{self.event_start[0]}_{self.freq_low[0] if self.freq_low else 0}_{self.freq_high[0] if self.freq_high else 0}"
        return hashlib.md5(key.encode()).hexdigest()[-10:]

    @property
    def long_id(self) -> str:
        """Unique string ID: ``<stop_time>_<hash>``."""
        if not self.event_stop:
            return "unknown"
        return f"{self.event_stop[0]:.4f}_{self.hash_id}"

    @property
    def snr(self) -> float:
        """Convenience: primary SNR value (= :attr:`rho`)."""
        return self.rho

    def network_snr(self) -> float:
        """Network SNR from ``signal_energy``: sqrt(Σ signal_energy) over detectors."""
        return math.sqrt(sum(self.signal_energy)) if self.signal_energy else 0.0

    # ==================================================================
    # PyArrow serialisation
    # ==================================================================

    @classmethod
    def arrow_schema(cls, ifo_list: list | None = None):
        """Return the :class:`pyarrow.Schema` for a table of :class:`Trigger` rows.

        Parameters
        ----------
        ifo_list : list[str], optional
            Detector names, e.g. ``["H1", "L1"]``.  When provided, per-IFO list
            columns (``time``, ``central_freq``, …) are **replaced** by flat scalar
            columns named ``time_H1``, ``time_L1``, ``central_freq_H1``, etc.
            This enables Parquet predicate pushdown for per-IFO filters.
            When ``None`` (default) the legacy ``list<float>`` columns are used.

        Usage::

            import pyarrow as pa
            # network-independent (list columns)
            schema = Trigger.arrow_schema()

            # flat columns for H1/L1 network
            schema = Trigger.arrow_schema(["H1", "L1"])
            table  = pa.Table.from_pylist(
                [t.to_arrow_dict(["H1", "L1"]) for t in triggers],
                schema=schema,
            )
        """
        import pyarrow as pa
        f32  = pa.float32()
        f64  = pa.float64()
        i32  = pa.int32()
        lf32 = pa.list_(f32)
        lf64 = pa.list_(f64)
        ls   = pa.list_(pa.string())

        fields = [
            # bookkeeping
            pa.field("id",                       pa.string()),
            pa.field("job_id",                   i32),
            pa.field("lag_idx",                  i32),
            pa.field("trial_idx",                i32),
            pa.field("cluster_id",               i32),
            pa.field("event_index",              i32),
            pa.field("n_detectors",              pa.int8()),
            pa.field("ifo_list",                 ls),
            pa.field("hybrid",                   pa.bool_()),
            # coherent SNR
            pa.field("rho",                      f32),
            pa.field("rho_alt",                  f32),
            pa.field("net_cc",                   f32),
            pa.field("sky_cc",                   f32),
            pa.field("subnet_cc",                f32),
            pa.field("subnet_cc2",               f32),
            # energy
            pa.field("likelihood",               f32),
            pa.field("coherent_energy",          f32),
            pa.field("coherent_energy_norm",     f32),
            pa.field("net_energy_disb",          f32),
            pa.field("net_null",                 f32),
            pa.field("net_energy",               f32),
            pa.field("like_sky",                 f32),
            pa.field("energy_sky",               f32),
            # network quality
            pa.field("network_sensitivity",      f32),
            pa.field("network_alignment_factor", f32),
            pa.field("network_index",            f32),
            pa.field("packet_norm",              f32),
            pa.field("penalty",                  f32),
            pa.field("cluster_union_size",       f32),
            pa.field("strain",                   f32),
            # pixel counts
            pa.field("n_pixels_total",           i32),
            pa.field("n_pixels_positive",        i32),
            pa.field("n_pixels_core",            i32),
            pa.field("sky_size",                 i32),
            # sky scalars
            pa.field("phi",                      f32),
            pa.field("theta",                    f32),
            pa.field("ra",                       f32),
            pa.field("dec",                      f32),
            pa.field("phi_det",                  f32),
            pa.field("theta_det",                f32),
            pa.field("psi",                      f32),
            pa.field("iota",                     f32),
            pa.field("sky_error_regions",        lf32),
            # chirp
            pa.field("mchirp",                   f32),
            pa.field("mchirp_err",               f32),
            pa.field("chirp_ellip",              f32),
            pa.field("chirp_pfrac",              f32),
            pa.field("chirp_efrac",              f32),
            pa.field("ebbh",                     lf32),
            # post-processing
            pa.field("ifar",                     f64),
            pa.field("q_veto",                   f32),
            pa.field("q_factor",                 f32),
            pa.field("gps_time",                 f64),
        ]

        # per-IFO fields: flat named scalars when ifo_list is known, list columns otherwise
        if ifo_list:
            for name in _PER_IFO_F64:
                for ifo in ifo_list:
                    fields.append(pa.field(f"{name}_{ifo}", f64))
            for name in _PER_IFO_F32:
                for ifo in ifo_list:
                    fields.append(pa.field(f"{name}_{ifo}", f32))
        else:
            for name in _PER_IFO_F64:
                fields.append(pa.field(name, lf64))
            for name in _PER_IFO_F32:
                fields.append(pa.field(name, lf32))

        fields.append(pa.field("injection", InjectionParams.arrow_struct(), nullable=True))
        return pa.schema(fields)

    def to_arrow_dict(self, ifo_list: list | None = None) -> dict:
        """Return a flat dict whose keys and value types match :meth:`arrow_schema`.

        Parameters
        ----------
        ifo_list : list[str], optional
            Detector names, e.g. ``["H1", "L1"]``.  When provided, per-IFO lists
            are expanded into ``{field}_{ifo}`` scalar keys to match a flat schema.
            Must match the ``ifo_list`` passed to :meth:`arrow_schema`.

        Use this to build a :class:`pyarrow.Table`::

            import pyarrow as pa
            ifo    = ["H1", "L1"]
            rows   = [t.to_arrow_dict(ifo) for t in triggers]
            table  = pa.Table.from_pylist(rows, schema=Trigger.arrow_schema(ifo))
        """
        d = {
            "id":                       self.id,
            "job_id":                   int(self.job_id),
            "lag_idx":                  int(self.lag_idx),
            "trial_idx":                int(self.trial_idx),
            "cluster_id":               int(self.cluster_id),
            "event_index":              int(self.event_index),
            "n_detectors":              int(self.n_detectors),
            "ifo_list":                 list(self.ifo_list),
            "hybrid":                   bool(self.hybrid),
            "rho":                      float(self.rho),
            "rho_alt":                  float(self.rho_alt),
            "net_cc":                   float(self.net_cc),
            "sky_cc":                   float(self.sky_cc),
            "subnet_cc":                float(self.subnet_cc),
            "subnet_cc2":               float(self.subnet_cc2),
            "likelihood":               float(self.likelihood),
            "coherent_energy":          float(self.coherent_energy),
            "coherent_energy_norm":     float(self.coherent_energy_norm),
            "net_energy_disb":          float(self.net_energy_disb),
            "net_null":                 float(self.net_null),
            "net_energy":               float(self.net_energy),
            "like_sky":                 float(self.like_sky),
            "energy_sky":               float(self.energy_sky),
            "network_sensitivity":      float(self.network_sensitivity),
            "network_alignment_factor": float(self.network_alignment_factor),
            "network_index":            float(self.network_index),
            "packet_norm":              float(self.packet_norm),
            "penalty":                  float(self.penalty),
            "cluster_union_size":       float(self.cluster_union_size),
            "strain":                   float(self.strain),
            "n_pixels_total":           int(self.n_pixels_total),
            "n_pixels_positive":        int(self.n_pixels_positive),
            "n_pixels_core":            int(self.n_pixels_core),
            "sky_size":                 int(self.sky_size),
            "phi":                      float(self.phi),
            "theta":                    float(self.theta),
            "ra":                       float(self.ra),
            "dec":                      float(self.dec),
            "phi_det":                  float(self.phi_det),
            "theta_det":                float(self.theta_det),
            "psi":                      float(self.psi),
            "iota":                     float(self.iota),
            "sky_error_regions":        [float(v) for v in self.sky_error_regions],
            "mchirp":                   float(self.mchirp),
            "mchirp_err":               float(self.mchirp_err),
            "chirp_ellip":              float(self.chirp_ellip),
            "chirp_pfrac":              float(self.chirp_pfrac),
            "chirp_efrac":              float(self.chirp_efrac),
            "ebbh":                     [float(v) for v in self.ebbh],
            "ifar":                     float(self.ifar),
            "q_veto":                   float(self.q_veto),
            "q_factor":                 float(self.q_factor),
            "gps_time":                 float(self.gps_time),
            "injection": self.injection.to_dict() if self.injection is not None else None,
        }

        # Serialize any dynamic attributes (set after construction, not declared as dataclass fields).
        # Values are cast to float/int/str based on their Python type.
        _declared = {f.name for f in dc_fields(self)}
        for k, v in self.__dict__.items():
            if k in _declared or k in d:
                continue
            if isinstance(v, bool):
                d[k] = bool(v)
            elif isinstance(v, int):
                d[k] = int(v)
            elif isinstance(v, float):
                d[k] = float(v)
            elif isinstance(v, str):
                d[k] = str(v)
            # skip non-scalar types (lists, dicts, objects)

        # per-IFO fields: flat scalars or list columns
        if ifo_list:
            for name in _PER_IFO_F64:
                lst = getattr(self, name)
                for i, ifo in enumerate(ifo_list):
                    d[f"{name}_{ifo}"] = float(lst[i]) if i < len(lst) else 0.0
            for name in _PER_IFO_F32:
                lst = getattr(self, name)
                for i, ifo in enumerate(ifo_list):
                    d[f"{name}_{ifo}"] = float(lst[i]) if i < len(lst) else 0.0
        else:
            for name in _PER_IFO_F64:
                d[name] = [float(v) for v in getattr(self, name)]
            for name in _PER_IFO_F32:
                d[name] = [float(v) for v in getattr(self, name)]

        return d

    # ==================================================================
    # Bridge adapter from legacy Event
    # ==================================================================

    @classmethod
    def from_event(cls, ev) -> "Trigger":
        """Convert a legacy :class:`~pycwb.types.network_event.Event` to a :class:`Trigger`.

        This adapter exists for the transition period.  Once all callers are updated to
        produce :class:`Trigger` directly, this method will be removed.
        """
        t = cls()

        # --- identity ---
        t.id           = getattr(ev, "id", "")
        t.job_id       = getattr(ev, "job_id", 0) or getattr(ev, "run", 0)
        t.trial_idx    = getattr(ev, "trial_idx", None) or 0
        t.lag_idx      = getattr(ev, "lag_idx",   None) or 0
        t.cluster_id   = ev.eventID[0] if len(getattr(ev, "eventID", [])) > 0 else 0
        t.event_index  = getattr(ev, "nevent", 0)
        t.n_detectors  = getattr(ev, "ndim", 0)
        t.ifo_list     = list(getattr(ev, "ifo_list", []))

        # --- SNR / correlation ---
        rho = getattr(ev, "rho", [0.0, 0.0])
        t.rho         = float(rho[0]) if len(rho) > 0 else 0.0
        t.rho_alt     = float(rho[1]) if len(rho) > 1 else 0.0

        netcc = getattr(ev, "netcc", [0.0] * 4)
        t.net_cc      = float(netcc[0]) if len(netcc) > 0 else 0.0
        t.sky_cc      = float(netcc[1]) if len(netcc) > 1 else 0.0
        t.subnet_cc   = float(netcc[2]) if len(netcc) > 2 else 0.0
        t.subnet_cc2  = float(netcc[3]) if len(netcc) > 3 else 0.0

        # --- energy ---
        t.likelihood           = float(getattr(ev, "likelihood", 0.0))
        t.coherent_energy      = float(getattr(ev, "ecor", 0.0))
        t.coherent_energy_norm = float(getattr(ev, "ECOR", 0.0))

        neted = getattr(ev, "neted", [0.0] * 5)
        t.net_energy_disb = float(neted[0]) if len(neted) > 0 else 0.0
        t.net_null        = float(neted[1]) if len(neted) > 1 else 0.0
        t.net_energy      = float(neted[2]) if len(neted) > 2 else 0.0
        t.like_sky        = float(neted[3]) if len(neted) > 3 else 0.0
        t.energy_sky      = float(neted[4]) if len(neted) > 4 else 0.0

        # --- network quality ---
        t.network_sensitivity      = float(getattr(ev, "gnet", 0.0))
        t.network_alignment_factor = float(getattr(ev, "anet", 0.0))
        t.network_index            = float(getattr(ev, "inet", 0.0))
        t.packet_norm              = float(getattr(ev, "norm", 0.0))
        t.penalty                  = float(getattr(ev, "penalty", 0.0))
        t.cluster_union_size       = float(getattr(ev, "usize", 0.0))

        strain = getattr(ev, "strain", [0.0])
        t.strain = float(strain[0]) if strain else 0.0

        # --- pixel counts ---
        volume = getattr(ev, "volume", [0, 0])
        size   = getattr(ev, "size",   [0, 0])
        t.n_pixels_total    = int(volume[0]) if len(volume) > 0 else 0
        t.n_pixels_positive = int(volume[1]) if len(volume) > 1 else 0
        t.n_pixels_core     = int(size[0])   if len(size) > 0   else 0
        t.sky_size          = int(size[1])   if len(size) > 1   else 0

        # --- sky ---
        phi   = getattr(ev, "phi",   [0.0] * 4)
        theta = getattr(ev, "theta", [0.0] * 4)
        psi   = getattr(ev, "psi",   [0.0])
        iota  = getattr(ev, "iota",  [0.0])
        t.phi       = float(phi[0])   if len(phi)   > 0 else 0.0
        t.theta     = float(theta[0]) if len(theta) > 0 else 0.0
        t.ra        = float(phi[2])   if len(phi)   > 2 else 0.0
        t.dec       = float(theta[2]) if len(theta) > 2 else 0.0
        t.phi_det   = float(phi[3])   if len(phi)   > 3 else 0.0
        t.theta_det = float(theta[3]) if len(theta) > 3 else 0.0
        t.psi       = float(psi[0])   if psi  else 0.0
        t.iota      = float(iota[0])  if iota else 0.0

        erA_raw = getattr(ev, "erA", [])
        # erA inside Event is appended once per IFO (each entry is a 11-element array);
        # take the first non-empty entry as the network-level sky area.
        import numpy as np
        if erA_raw:
            first = erA_raw[0]
            if hasattr(first, "__len__"):
                t.sky_error_regions = [float(v) for v in first]
            else:
                t.sky_error_regions = [float(v) for v in erA_raw]
        else:
            t.sky_error_regions = []

        # --- chirp ---
        chirp = getattr(ev, "chirp", [0.0] * 6)
        t.mchirp      = float(chirp[1]) if len(chirp) > 1 else 0.0
        t.mchirp_err  = float(chirp[2]) if len(chirp) > 2 else 0.0
        t.chirp_ellip = float(chirp[3]) if len(chirp) > 3 else 0.0
        t.chirp_pfrac = float(chirp[4]) if len(chirp) > 4 else 0.0
        t.chirp_efrac = float(chirp[5]) if len(chirp) > 5 else 0.0
        t.ebbh        = [float(v) for v in getattr(ev, "eBBH", [])]

        # --- per-IFO ---
        def _fl(lst):
            return [float(v) for v in (lst if lst is not None else []) if v is not None]

        t.time             = _fl(getattr(ev, "time",      []))
        t.segment_start    = _fl(getattr(ev, "gps",       []))
        t.event_start      = _fl(getattr(ev, "start",     []))
        t.event_stop       = _fl(getattr(ev, "stop",      []))
        t.left_edge        = _fl(getattr(ev, "left",      []))
        t.right_edge       = _fl(getattr(ev, "right",     []))
        t.duration         = _fl(getattr(ev, "duration",  []))
        t.central_freq     = _fl(getattr(ev, "frequency", []))
        t.freq_low         = _fl(getattr(ev, "low",       []))
        t.freq_high        = _fl(getattr(ev, "high",      []))
        t.bandwidth        = _fl(getattr(ev, "bandwidth", []))
        t.hrss             = _fl(getattr(ev, "hrss",      []))
        t.noise_rms        = _fl(getattr(ev, "noise",     []))
        t.data_energy      = _fl(getattr(ev, "snr",       []))
        t.signal_energy    = _fl(getattr(ev, "sSNR",      []))
        t.cross_energy     = _fl(getattr(ev, "xSNR",      []))
        t.null_energy      = _fl(getattr(ev, "null",      []))
        t.residual_energy  = _fl(getattr(ev, "nill",      []))
        t.fp               = _fl(getattr(ev, "bp",        []))
        t.fx               = _fl(getattr(ev, "bx",        []))
        t.sample_rate      = _fl(getattr(ev, "rate",      []))
        t.segment_lag      = _fl(getattr(ev, "slag",      []))

        # lag: Event stores [shift]*n_ifo + [lagShift]*n_ifo; take first half
        lag_raw = getattr(ev, "lag", [])
        n = t.n_detectors or (len(lag_raw) // 2)
        t.time_lag = _fl(lag_raw[:n])

        # scalar reference time: copy of the first-detector GPS time for fast Parquet filtering
        t.gps_time = t.time[0] if t.time else 0.0

        # --- injection (from dict or InjectionParams) ---
        inj_raw = getattr(ev, "injection", None)
        isnr    = getattr(ev, "iSNR",      [])
        osnr    = getattr(ev, "oSNR",      [])
        iosnr   = getattr(ev, "ioSNR",     [])
        deff    = getattr(ev, "Deff",      [])
        phi1    = float(phi[1])   if len(phi)   > 1 else 0.0
        theta1  = float(theta[1]) if len(theta) > 1 else 0.0
        psi1    = float(getattr(ev, "psi",  [0.0, 0.0])[1]) if len(getattr(ev, "psi", [])) > 1 else 0.0
        iota1   = float(getattr(ev, "iota", [0.0, 0.0])[1]) if len(getattr(ev, "iota", [])) > 1 else 0.0
        chirp0  = float(chirp[0]) if chirp else 0.0
        strain1 = float(strain[1]) if len(strain) > 1 else 0.0
        n_ifo   = t.n_detectors
        bp_raw  = getattr(ev, "bp", [])
        bx_raw  = getattr(ev, "bx", [])
        t_raw   = getattr(ev, "time", [])
        inj_fp  = _fl(bp_raw[n_ifo:n_ifo * 2]) if len(bp_raw) >= n_ifo * 2 else []
        inj_fx  = _fl(bx_raw[n_ifo:n_ifo * 2]) if len(bx_raw) >= n_ifo * 2 else []
        inj_time = _fl(t_raw[n_ifo:n_ifo * 2]) if len(t_raw) >= n_ifo * 2 else []

        has_injection = (
            bool(isnr) or bool(iota1) or bool(phi1) or bool(strain1)
            or (isinstance(inj_raw, dict) and inj_raw)
            or isinstance(inj_raw, InjectionParams)
        )
        if has_injection:
            if isinstance(inj_raw, InjectionParams):
                inj = inj_raw
                _need_ra_fallback  = False  # ra/dec already set in the object
                _need_dec_fallback = False
            elif isinstance(inj_raw, dict) and inj_raw:
                inj = InjectionParams.from_injection_dict(inj_raw)
                # Fall back to cWB equatorial coords only when the user dict
                # did not supply the key — avoids overwriting a valid RA=0.0.
                _need_ra_fallback  = "ra"  not in inj_raw
                _need_dec_fallback = "dec" not in inj_raw
            else:
                # Legacy cWB path: no user dict — reconstruct from Event arrays
                import json as _json
                inj = InjectionParams()
                inj.ra       = float(phi[2])   if len(phi)   > 2 else 0.0
                inj.dec      = float(theta[2]) if len(theta) > 2 else 0.0
                inj.hrss     = strain1
                inj.gps_time = inj_time[0] if inj_time else 0.0
                inj.parameters = _json.dumps({
                    "phi":    phi1,
                    "theta":  theta1,
                    "psi":    psi1,
                    "iota":   iota1,
                    "mchirp": chirp0,
                    "strain": strain1,
                })
                _need_ra_fallback  = False  # ra/dec set directly above
                _need_dec_fallback = False
            # Always fill per-IFO injection-matched fields from cWB Event arrays
            # (intentional clobber): these are computed at the *injected* sky
            # position/time and are distinct from the Trigger's reconstructed
            # per-IFO columns, even when inj_raw was already an InjectionParams.
            inj.fp          = inj_fp
            inj.fx          = inj_fx
            inj.time        = inj_time
            inj.snr_sq      = _fl(isnr)
            inj.rec_snr_sq  = _fl(osnr)
            inj.overlap_snr = _fl(iosnr)
            inj.d_eff       = _fl(deff)
            inj.hrss_det    = _fl(getattr(ev, "hrss0", []))
            # Fall back to cWB equatorial coords when the dict did not supply
            # ra/dec.  Guard uses explicit flags (not truthiness) so that a
            # valid RA=0.0 (Vernal Equinox direction) is never silently overwritten.
            if _need_ra_fallback and len(phi) > 2:
                inj.ra = float(phi[2])
            if _need_dec_fallback and len(theta) > 2:
                inj.dec = float(theta[2])
            t.injection = inj

        # --- q-veto ---
        qveto_arr = getattr(ev, "Qveto", [])
        if len(qveto_arr) >= 2:
            t.q_veto   = float(qveto_arr[0])
            t.q_factor = float(qveto_arr[1])
        else:
            t.q_veto   = float(getattr(ev, "qveto",   0.0))
            t.q_factor = float(getattr(ev, "qfactor", 0.0))

        if not t.id:
            t.id = t.long_id

        # Forward any dynamic (non-declared) attributes set on the legacy Event
        # object onto the Trigger instance so that to_arrow_dict() can serialise
        # them via its own dynamic-attribute fallback.
        _declared = {f.name for f in dc_fields(t)}
        for k, v in ev.__dict__.items():
            if k not in _declared and not k.startswith("_"):
                try:
                    setattr(t, k, v)
                except Exception:
                    pass

        return t
