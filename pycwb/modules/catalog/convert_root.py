"""
Convert cWB ROOT ``waveburst`` trees to the pycWB Arrow/Parquet catalog format.

Requires ``uproot`` (``pip install uproot awkward``).

Typical usage
-------------
::

    from pycwb.modules.catalog.convert_root import convert_root_to_catalog

    # create a brand-new catalog from a ROOT file
    convert_root_to_catalog(
        root_file  = "wave_O4_K17_C00_LH_BurstLF_BKG_run1.M2.root",
        catalog_file = "catalog_bkg.parquet",
        ifo_list   = ["H1", "L1"],
    )

    # convert several files and append into one catalog
    for f in root_files:
        convert_root_to_catalog(f, "catalog_bkg.parquet", ifo_list=["H1","L1"],
                                 append=True)

ROOT → Trigger field mapping
----------------------------
The mapping follows the legacy names documented in each :class:`~pycwb.types.trigger.Trigger`
field docstring and mirrors the logic in :meth:`~pycwb.types.trigger.Trigger.from_event`.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Branch list
# All branches we try to read from the waveburst tree.
# Missing branches produce None and are handled gracefully.
# ---------------------------------------------------------------------------

_SCALAR_BRANCHES = [
    "run", "nevent", "ndim",
    "likelihood", "ecor", "ECOR",
    "gnet", "anet", "inet", "norm", "penalty", "usize",
]

_ARRAY_BRANCHES = [
    # identity
    "eventID",
    # SNR / cc
    "rho", "netcc",
    # energy breakdown
    "neted",
    # strain / size
    "strain", "volume", "size",
    # sky
    "phi", "theta", "psi", "iota", "erA",
    # chirp
    "chirp", "eBBH",
    # q-veto
    "Qveto", "Lveto",
    # per-IFO timing
    "time", "gps", "start", "stop", "left", "right", "duration", "lag", "slag",
    # per-IFO frequency
    "frequency", "low", "high", "bandwidth", "rate",
    # per-IFO amplitude
    "hrss", "noise", "snr", "sSNR", "xSNR", "null", "nill",
    # per-IFO antenna
    "bp", "bx",
    # injection (optional – only present in injection runs)
    "type", "factor", "phi0", "theta0", "ra0", "dec0",
    "psi0", "iota0", "chirp0", "strain0", "range",
    "time0", "hrss0", "bp0", "bx0",
    "iSNR", "oSNR", "ioSNR", "Deff",
    "mass", "spin",
]


def _safe(arr, idx: int, default=0.0):
    """Return element *idx* of *arr* as float, or *default* if out of range."""
    if arr is None:
        return default
    try:
        return float(arr[idx])
    except (IndexError, TypeError):
        return default


def _fl(arr, default=0.0) -> list[float]:
    """Convert an array-like to ``list[float]``, returning ``[]`` for None."""
    if arr is None:
        return []
    try:
        return [float(v) for v in arr]
    except TypeError:
        return []


def _make_id(job_id: int, cluster_id: int, event_index: int) -> str:
    raw = f"{job_id}_{cluster_id}_{event_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]  # noqa: S324


def _row_to_trigger(row: dict, job_id: int) -> "Trigger":
    """Build a :class:`~pycwb.types.trigger.Trigger` from a single *row* dict.

    *row* maps branch name → scalar / 1-D numpy array for that event.
    """
    from pycwb.types.trigger import InjectionParams, Trigger

    t = Trigger()

    # --- identity --------------------------------------------------------
    event_id  = int(_safe(row.get("eventID"), 0))
    nevent    = int(row.get("nevent", 0) or 0)
    t.job_id       = job_id
    t.cluster_id   = event_id
    t.event_index  = nevent
    t.n_detectors  = int(row.get("ndim", 0) or 0)
    t.id           = _make_id(job_id, event_id, nevent)

    # --- SNR / correlation -----------------------------------------------
    rho   = row.get("rho")
    netcc = row.get("netcc")
    t.rho        = _safe(rho,   0)
    t.rho_alt    = _safe(rho,   1)
    t.net_cc     = _safe(netcc, 0)
    t.sky_cc     = _safe(netcc, 1)
    t.subnet_cc  = _safe(netcc, 2)
    t.subnet_cc2 = _safe(netcc, 3)

    # --- energy ----------------------------------------------------------
    neted  = row.get("neted")
    t.likelihood           = float(row.get("likelihood", 0.0) or 0.0)
    t.coherent_energy      = float(row.get("ecor", 0.0) or 0.0)
    t.coherent_energy_norm = float(row.get("ECOR", 0.0) or 0.0)
    t.net_energy_disb = _safe(neted, 0)
    t.net_null        = _safe(neted, 1)
    t.net_energy      = _safe(neted, 2)
    t.like_sky        = _safe(neted, 3)
    t.energy_sky      = _safe(neted, 4)

    # --- network quality -------------------------------------------------
    t.network_sensitivity      = float(row.get("gnet", 0.0) or 0.0)
    t.network_alignment_factor = float(row.get("anet", 0.0) or 0.0)
    t.network_index            = float(row.get("inet", 0.0) or 0.0)
    t.packet_norm              = float(row.get("norm", 0.0) or 0.0)
    t.penalty                  = float(row.get("penalty", 0.0) or 0.0)
    t.cluster_union_size       = float(row.get("usize", 0.0) or 0.0)
    t.strain                   = _safe(row.get("strain"), 0)

    # --- pixel counts ----------------------------------------------------
    volume = row.get("volume")
    size   = row.get("size")
    t.n_pixels_total    = int(_safe(volume, 0))
    t.n_pixels_positive = int(_safe(volume, 1))
    t.n_pixels_core     = int(_safe(size,   0))
    t.sky_size          = int(_safe(size,   1))

    # --- sky position ----------------------------------------------------
    phi   = row.get("phi")
    theta = row.get("theta")
    psi   = row.get("psi")
    iota  = row.get("iota")
    t.phi       = _safe(phi,   0)
    t.theta     = _safe(theta, 0)
    t.ra        = _safe(phi,   2)
    t.dec       = _safe(theta, 2)
    t.phi_det   = _safe(phi,   3)
    t.theta_det = _safe(theta, 3)
    t.psi       = _safe(psi,   0)
    t.iota      = _safe(iota,  0)

    erA = row.get("erA")
    t.sky_error_regions = _fl(erA) if erA is not None else []

    # --- chirp -----------------------------------------------------------
    chirp = row.get("chirp")
    t.mchirp      = _safe(chirp, 1)
    t.mchirp_err  = _safe(chirp, 2)
    t.chirp_ellip = _safe(chirp, 3)
    t.chirp_pfrac = _safe(chirp, 4)
    t.chirp_efrac = _safe(chirp, 5)
    t.ebbh        = _fl(row.get("eBBH"))

    # --- q-veto ----------------------------------------------------------
    qveto = row.get("Qveto")
    t.q_veto   = _safe(qveto, 0)
    t.q_factor = _safe(qveto, 1)

    # --- per-IFO timing --------------------------------------------------
    #   lag: waveburst stores [shift_0, shift_1, ...] + [lag_0, lag_1, ...]
    lag_raw = _fl(row.get("lag"))
    n_ifo   = t.n_detectors or (len(lag_raw) // 2)
    t.time          = _fl(row.get("time"))
    t.segment_start = _fl(row.get("gps"))
    t.event_start   = _fl(row.get("start"))
    t.event_stop    = _fl(row.get("stop"))
    t.left_edge     = _fl(row.get("left"))
    t.right_edge    = _fl(row.get("right"))
    t.duration      = _fl(row.get("duration"))
    t.time_lag      = lag_raw[:n_ifo]
    t.segment_lag   = _fl(row.get("slag"))

    t.gps_time = t.time[0] if t.time else 0.0

    # --- per-IFO frequency -----------------------------------------------
    t.central_freq = _fl(row.get("frequency"))
    t.freq_low     = _fl(row.get("low"))
    t.freq_high    = _fl(row.get("high"))
    t.bandwidth    = _fl(row.get("bandwidth"))
    t.sample_rate  = _fl(row.get("rate"))

    # --- per-IFO amplitude -----------------------------------------------
    t.hrss            = _fl(row.get("hrss"))
    t.noise_rms       = _fl(row.get("noise"))
    t.data_energy     = _fl(row.get("snr"))
    t.signal_energy   = _fl(row.get("sSNR"))
    t.cross_energy    = _fl(row.get("xSNR"))
    t.null_energy     = _fl(row.get("null"))
    t.residual_energy = _fl(row.get("nill"))
    t.fp              = _fl(row.get("bp"))
    t.fx              = _fl(row.get("bx"))

    # --- injection (optional) -------------------------------------------
    phi0   = row.get("phi0")
    theta0 = row.get("theta0")
    if phi0 is not None or theta0 is not None:
        import json as _json
        ra0     = row.get("ra0")
        dec0    = row.get("dec0")
        chirp0  = row.get("chirp0")
        strain0 = row.get("strain0")
        rng     = row.get("range")
        time0   = _fl(row.get("time0"))
        hrss0   = _fl(row.get("hrss0"))
        bp0     = _fl(row.get("bp0"))
        bx0     = _fl(row.get("bx0"))
        isnr    = _fl(row.get("iSNR"))
        osnr    = _fl(row.get("oSNR"))
        iosnr   = _fl(row.get("ioSNR"))
        deff    = _fl(row.get("Deff"))
        # Pack all available ROOT injection fields into JSON parameters.
        params_dict = {
            "phi":              _safe(phi0,   0),
            "theta":            _safe(theta0, 0),
            "psi":              _safe(row.get("psi0"),  0),
            "iota":             _safe(row.get("iota0"), 0),
            "mchirp":           _safe(chirp0, 0),
            "strain":           _safe(strain0, 0),
            "distance":         _safe(rng,    1),
            "waveform_type":    int(_safe(row.get("type"), 1)),
            "amplitude_factor": float(row.get("factor", 0.0) or 0.0),
            "mass":             _fl(row.get("mass")),
            "spin":             _fl(row.get("spin")),
        }
        t.injection = InjectionParams(
            ra          = _safe(ra0,  0) if ra0  is not None else _safe(phi0,   2),
            dec         = _safe(dec0, 0) if dec0 is not None else _safe(theta0, 2),
            gps_time    = time0[0] if time0 else 0.0,
            hrss        = _safe(strain0, 0),
            pol         = _safe(row.get("psi0"), 0),
            parameters  = _json.dumps(params_dict),
            time        = time0,
            hrss_det    = hrss0,
            fp          = bp0,
            fx          = bx0,
            snr_sq      = isnr,
            rec_snr_sq  = osnr,
            overlap_snr = iosnr,
            d_eff       = deff,
        )

    return t


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_root_triggers(
    root_file: str,
    ifo_list: list[str],
    job_id: int = 0,
    tree_name: str = "waveburst",
    batch_size: int = 10_000,
) -> list["Trigger"]:
    """Read all events from a cWB ROOT file and return a list of :class:`~pycwb.types.trigger.Trigger`.

    Parameters
    ----------
    root_file : str
        Path to the ``.root`` file.
    ifo_list : list[str]
        Detector names in network order, e.g. ``["H1", "L1"]``.
    job_id : int
        Job ID to assign (use the run number from the filename or a counter).
    tree_name : str
        Name of the ROOT tree.  Default: ``"waveburst"``.
    batch_size : int
        Number of events to read per uproot batch.

    Returns
    -------
    list[Trigger]
    """
    try:
        import uproot
    except ImportError as exc:
        raise ImportError("uproot is required: pip install uproot awkward") from exc

    all_branches = _SCALAR_BRANCHES + _ARRAY_BRANCHES

    triggers: list = []
    with uproot.open(f"{root_file}:{tree_name}") as tree:
        available = set(tree.keys())
        branches  = [b for b in all_branches if b in available]
        missing   = [b for b in all_branches if b not in available]
        if missing:
            logger.debug("Branches not found in %s (skipped): %s", root_file, missing)
        logger.info("Reading %d events from %s", tree.num_entries, root_file)

        for batch in tree.iterate(branches, library="np", step_size=batch_size):
            n = len(next(iter(batch.values())))
            for i in range(n):
                row = {k: (v[i] if v is not None else None) for k, v in batch.items()}
                row["ndim"] = row.get("ndim", len(ifo_list))
                t = _row_to_trigger(row, job_id)
                t.ifo_list = list(ifo_list)
                triggers.append(t)

    logger.info("Converted %d triggers from %s", len(triggers), root_file)
    return triggers


def convert_root_to_catalog(
    root_file: str,
    catalog_file: str,
    ifo_list: list[str],
    config=None,
    jobs: Optional[list] = None,
    job_id: int = 0,
    tree_name: str = "waveburst",
    append: bool = False,
    batch_size: int = 10_000,
) -> None:
    """Convert a cWB ROOT ``waveburst`` tree to an Arrow/Parquet catalog.

    Parameters
    ----------
    root_file : str
        Path to the input ``.root`` file.
    catalog_file : str
        Destination ``.parquet`` path.
    ifo_list : list[str]
        Detector names in network order, e.g. ``["H1", "L1"]``.
    config : Config or None
        pycWB config to embed in catalog metadata.  A minimal stub is created
        if ``None``.
    jobs : list or None
        Job segment list to embed in metadata.  Empty list if ``None``.
    job_id : int
        Job ID to stamp on every trigger.
    tree_name : str
        ROOT tree name.  Default: ``"waveburst"``.
    append : bool
        If ``True`` and *catalog_file* already exists, append to it instead
        of creating a new file.
    batch_size : int
        Number of events read from ROOT per iteration.
    """
    from pycwb.modules.catalog import Catalog

    # --- ensure catalog exists ---
    if not append or not os.path.exists(catalog_file):
        if config is None:
            # minimal stub so the file is self-describing
            class _StubConfig:
                ifo = list(ifo_list)
                def __init__(self):
                    self.__dict__ = {"ifo": list(ifo_list)}
            config = _StubConfig()
        Catalog.create(catalog_file, config, jobs or [])
        logger.info("Created catalog: %s", catalog_file)

    # --- read and append ---
    triggers = read_root_triggers(
        root_file, ifo_list, job_id=job_id,
        tree_name=tree_name, batch_size=batch_size,
    )
    if triggers:
        Catalog.open(catalog_file).add_triggers(triggers)
        logger.info("Appended %d triggers to %s", len(triggers), catalog_file)
    else:
        logger.warning("No triggers found in %s", root_file)
