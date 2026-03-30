"""
Coherence analysis module for gravitational wave burst detection.

This module implements the core coherence pipeline: WDM wavelet decomposition,
time-frequency map construction, pixel selection, and clustering. It provides
a streaming-friendly API (setup once, iterate over lags) for efficient
multi-lag analysis.

The main entry points are :func:`coherence` (all-in-one) and :func:`setup_coherence`
combined with :func:`coherence_single_lag` (streaming mode).
"""
import time
import logging
import numpy as np
from scipy.special import gammainccinv
from wdm_wavelet.wdm import WDM as WDMWavelet
from pycwb.config import Config
from pycwb.types.job import WaveSegment
from pycwb.types.time_series import TimeSeries
from pycwb.types.time_frequency_map import TimeFrequencyMap
from pycwb.types.network_cluster import FragmentCluster, Cluster, ClusterMeta
from pycwb.types.network_pixel import Pixel, PixelData
from pycwb.modules.cwb_coherence.tf_batch_generation import batch_t2w_detectors
from pycwb.modules.cwb_coherence.time_delay_max_energy import time_delay_max_energy
from numba import njit
from typing import Any
from scipy.special import gammaincc as _scipy_gammaincc
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as _cc

logger = logging.getLogger(__name__)


def coherence(config: Config, strains: list[TimeSeries], return_rejected: bool = False, job_seg: WaveSegment | None = None) -> list[list[FragmentCluster]]:
    """
    Select the significant pixels for all resolution levels and all lags.

    This is the interactive convenience wrapper.  Internally it calls
    :func:`setup_coherence` once (expensive: WDM decomposition + TF maps)
    and then :func:`coherence_single_lag` for every lag (cheap: pixel
    selection + clustering only).

    Parameters
    ----------
    config : Config
        Configuration object.
    strains : list[TimeSeries]
        List of whitened strain time series.
    return_rejected : bool
        If True, keep rejected clusters in the output.
    job_seg : WaveSegment, optional
        Job segment supplying lag count and per-lag time shifts.
        When *None* a single zero-lag pass is performed.

    Returns
    -------
    list[list[FragmentCluster]]
        ``result[res][lag]`` — one FragmentCluster per resolution per lag.
    """
    logger.info("Starting coherence")

    if job_seg is None:
        # Minimal single-lag fallback for interactive / testing use.
        n_ifo = len(strains)
        import types as _types
        job_seg = _types.SimpleNamespace(
            n_lag=1,
            lag_shifts=[np.zeros(n_ifo)],
        )

    setups = setup_coherence(config, strains, job_seg=job_seg)
    n_lag = job_seg.n_lag
    n_res = len(setups)

    # Run per-lag coherence using the pre-built setup
    per_lag = [coherence_single_lag(setups, lag, return_rejected) for lag in range(n_lag)]

    # Transpose from [lag][res] → [res][lag] (legacy output format)
    result = [[per_lag[lag][res] for lag in range(n_lag)] for res in range(n_res)]

    return result


# ---------------------------------------------------------------------------
# Streaming-friendly API: setup once, iterate over lags
# ---------------------------------------------------------------------------

def setup_coherence(config: Config, strains: list[TimeSeries], job_seg: WaveSegment | None = None) -> list[dict]:
    """
    Compute all lag-independent coherence data (TF maps after max_energy,
    threshold, lag plan) for every resolution level.

    Call this once per job segment, then pass the returned list to
    :func:`coherence_single_lag` for each lag.

    Parameters
    ----------
    config : Config
        Configuration object.
    strains : list[TimeSeries]
        Whitened strain time series.
    job_seg : WaveSegment or None, optional
        Job segment (provides lag count via ``job_seg.n_lag``).

    Returns
    -------
    list[dict]
        One setup dict per resolution, keyed by ``tf_maps``, ``Eo``,
        ``job_seg``, ``pattern``, ``level``, ``layers``, ``rate``,
        ``select_subrho``, ``select_subnet``, ``segEdge``.
    """
    # Compute upsample factor for max_energy (minimum 1)
    up_n = int(config.rateANA / 1024)
    if up_n < 1:
        up_n = 1

    # Normalize input strains to pycwb TimeSeries objects
    normalized_strains = [TimeSeries.from_input(strain) for strain in strains]

    # Build setups for each resolution level independently
    # (expensive WDM transforms, TF maps, and thresholds are computed once here,
    #  then reused across all lags in coherence_single_lag)
    setups = [
        _setup_coherence_single_res(i, config, normalized_strains, up_n,
                                    job_seg=job_seg)
        for i in range(config.nRES)
    ]

    return setups


def _setup_coherence_single_res(i: int, config: Config, strains: list[TimeSeries], up_n: int, job_seg: WaveSegment | None = None) -> dict:
    """
    Lag-independent coherence setup for one resolution level.

    Builds the WDM wavelet, TF maps, applies max_energy, computes the
    energy threshold, and builds the lag plan.  Nothing here depends on
    which lag is being processed.

    Returns
    -------
    dict
        Keys: ``tf_maps``, ``Eo``, ``job_seg``, ``pattern``, ``level``,
        ``layers``, ``rate``, ``select_subrho``, ``select_subnet``,
        ``segEdge``.
    """
    timer_start = time.perf_counter()
    level = config.l_high - i
    layers = 2 ** level if level > 0 else 0
    rate = config.rateANA // 2 ** level

    # Ensure at least one WDM layer for zero-lag case
    wdm_layers = max(1, layers)
    wdm_wavelet = WDMWavelet(
        M=wdm_layers,
        K=wdm_layers,
        beta_order=config.WDM_beta_order,
        precision=config.WDM_precision,
    )

    # Build time-frequency maps via batch WDM transform (preferring fast path)
    try:
        batch_data_list, (dt, df) = batch_t2w_detectors(strains, wdm_wavelet)
        tf_maps = [
            TimeFrequencyMap(
                data=batch_data_list[n],
                is_whitened=True,
                dt=dt,
                df=df,
                start=float(strains[n].t0),
                stop=float(strains[n].end_time),
                f_low=getattr(config, "fLow", None),
                f_high=getattr(config, "fHigh", None),
                edge=getattr(config, "segEdge", None),
                wavelet=wdm_wavelet,
                len_timeseries=len(strains[n].data),
            )
            for n in range(len(strains))
        ]
    except Exception as exc:  # broad catch intentional: batch_t2w_detectors may raise any of
        # TypeError / ValueError / AttributeError / RuntimeError / numpy internals depending on
        # the WDM implementation version; we always want the serial fallback to succeed.
        logger.warning("Batch t2w failed (%s); falling back to serial from_timeseries", exc)
        tf_maps = [
            TimeFrequencyMap.from_timeseries(
                ts=strain,
                wavelet=wdm_wavelet,
                is_whitened=True,
                f_low=getattr(config, "fLow", None),
                f_high=getattr(config, "fHigh", None),
                edge=getattr(config, "segEdge", None),
            )
            for strain in strains
        ]

    logger.info(
        "level : %d\t rate(hz) : %d\t layers : %d\t df(hz) : %f\t dt(ms) : %f",
        level, rate, layers,
        config.rateANA / 2. / (2 ** level),
        1000. / rate,
    )

    # Apply max_energy skymap projection to decorrelate across lags
    # (computes the optimal coherent energy over sky positions)
    max_delay = config.max_delay
    pattern = config.pattern
    alp = 0.0
    for n, tf_map in enumerate(tf_maps):
        tf_maps[n], alp_n = max_energy(
            tf_map=tf_map,
            max_delay=max_delay,
            up_n=up_n,
            pattern=pattern,
            f_low=config.fLow,
            f_high=config.fHigh,
        )
        alp += alp_n
    # Average the Gamma-to-Gauss scaling factor across detectors
    alp = alp / config.nIFO

    # Compute pixel energy threshold based on black-pixel probability
    # (independent of lag since TF maps are lag-independent)
    Eo = compute_threshold(
        tf_maps,
        config.bpp,
        alp=alp if pattern != 0 else None,
        edge=config.segEdge,
    )

    # Extract lag count from job segment for setup dictionary
    n_lag = job_seg.n_lag

    logger.info("level %d setup done: Eo=%.4g, n_lag=%d  (%.2f s)", level, Eo, n_lag, time.perf_counter() - timer_start)

    return {
        "tf_maps": tf_maps,
        "Eo": Eo,
        "job_seg": job_seg,
        "pattern": pattern,
        "level": level,
        "layers": layers,
        "rate": rate,
        "select_subrho": config.select_subrho,
        "select_subnet": config.select_subnet,
        "segEdge": config.segEdge,
    }


def coherence_single_lag(
    coherence_setups: list[dict],
    lag_idx: int,
    return_rejected: bool = False,
    veto_windows: list[tuple[float, float]] | None = None,
) -> list[FragmentCluster]:
    """
    Compute coherence for one lag index, using pre-built per-resolution setups from :func:`setup_coherence`.

    This is the per-lag cheap counterpart: only pixel selection and clustering
    are performed here; all expensive TF/WDM work is already done.

    Parameters
    ----------
    coherence_setups : list[dict]
        Returned by :func:`setup_coherence`.
    lag_idx : int
        Zero-based lag index.
    return_rejected : bool
        If True keep rejected clusters in the output.
    veto_windows : list[tuple[float, float]] or None
        GPS intervals ``(start, end)`` to keep.  When not ``None``, a binary
        mask is built via :func:`build_veto_mask` and passed to pixel
        selection so that only pixels inside these windows survive.

    Returns
    -------
    list[FragmentCluster]
        One FragmentCluster per resolution for this lag.
    """
    fragment_clusters = []
    for setup in coherence_setups:
        # Unpack lag-independent setup data for this resolution
        tf_maps = setup["tf_maps"]
        Eo = setup["Eo"]
        job_seg = setup["job_seg"]
        pattern = setup["pattern"]

        # Validate lag index is within range
        if lag_idx >= job_seg.n_lag:
            raise IndexError(
                f"lag_idx={lag_idx} is out of range n_lag={job_seg.n_lag}"
            )

        # Select significant pixels above threshold for this lag
        # (applies time shifts and optional veto masks)
        veto = None
        if veto_windows is not None:
            veto = build_veto_mask(tf_maps[0], veto_windows, edge=setup["segEdge"])
        candidates = select_network_pixels(
            tf_maps=tf_maps,
            lag_index=lag_idx,
            energy_threshold=Eo,
            lag_shifts=job_seg.lag_shifts[lag_idx],
            veto=veto,
            edge=setup["segEdge"],
        )
        n_candidates = int(len(candidates["frequency"])) if isinstance(candidates, dict) else -1

        # Cluster selected pixels and apply statistical selection criteria
        # (min/max cluster sizes depend on wave pattern)
        if pattern != 0:
            # Multi-pixel clusters for network patterns (kt=2 time bins, kf=3 freq bins)
            c = cluster_pixels(candidates, kt=2, kf=3)
            c.select("subrho", setup["select_subrho"])
            c.select("subnet", setup["select_subnet"])
        else:
            # Single-pixel clusters for non-network patterns
            c = cluster_pixels(candidates, kt=1, kf=1)

        # Remove clusters rejected by statistical selection unless explicitly requested
        if not return_rejected:
            c.remove_rejected()

        logger.info(
            "lag=%3d level=%d | events=%d pixels=%d candidates=%d",
            lag_idx, setup["level"], c.event_count(), c.pixel_count(), n_candidates,
        )
        fragment_clusters.append(c)

    return fragment_clusters


def max_energy(tf_map: TimeFrequencyMap, max_delay: float, up_n: int, pattern: int,
               f_low: float | None = None, f_high: float | None = None, hist: list | None = None) -> tuple[TimeFrequencyMap, float]:
    """
    Compute max-energy skymap projection for a detector TF map.

    Calls :func:`time_delay_max_energy` from the module-level pure-function
    implementation and returns a new TF map together with the Gamma-to-Gauss
    scaling parameter.

    Parameters
    ----------
    tf_map : TimeFrequencyMap
        Detector time-frequency map object.
    max_delay : float
        Maximum delay for the time series.
    up_n : int
        Upsample factor for decorrelation.
    pattern : int
        Wave packet pattern identifier.
    f_low : float | None, optional
        Low-frequency cutoff in Hz.
    f_high : float | None, optional
        High-frequency cutoff in Hz.
    hist : list | None, optional
        Optional histogram container for statistics.

    Returns
    -------
    tuple[TimeFrequencyMap, float]
        Updated TF map after max-energy projection and Gamma-to-Gauss scaling
        factor.
    """
    if hasattr(tf_map, "bandpass"):
        tf_map.bandpass(f_low=f_low, f_high=f_high)

    new_tf_map, result = time_delay_max_energy(tf_map, max_delay, downsample=up_n, pattern=pattern, hist=hist)
    return new_tf_map, result


def build_veto_mask(tf_map, veto_windows: list[tuple[float, float]], edge: float | None = None) -> np.ndarray:
    """
    Build a binary keep-mask from GPS time windows for a TF map timeline.

    Bins inside any of *veto_windows* are marked 1 (keep); everything else
    is 0 (reject).  The resulting array can be passed as the ``veto``
    argument to :func:`select_network_pixels`.

    Parameters
    ----------
    tf_map : object
        Reference TF map that defines the time axis (must expose ``data``,
        ``dt``, and ``start`` attributes).
    veto_windows : list[tuple[float, float]]
        GPS intervals ``(start, end)`` to keep.  Overlapping windows are
        handled correctly (union).
    edge : float or None
        Ignored for mask construction but accepted for API consistency.

    Returns
    -------
    np.ndarray
        1-D int16 array of length ``n_time`` (mask: 1=keep, 0=reject).
    """
    data = np.asarray(getattr(tf_map, "data", []))
    n_samples = int(data.shape[1]) if data.ndim == 2 else int(data.size)
    if n_samples <= 0:
        return np.zeros(0, dtype=np.int16)

    dt = float(getattr(tf_map, "dt", 0.0))
    if dt <= 0:
        raise ValueError("tf_map.dt must be positive for veto mask construction")

    rate = 1.0 / dt
    start = float(getattr(tf_map, "start", 0.0))
    stop = float(getattr(tf_map, "stop", start + n_samples * dt))

    mask = np.zeros(n_samples, dtype=np.int16)
    for seg_start, seg_end in veto_windows:
        s = min(max(float(seg_start), start), stop)
        e = min(max(float(seg_end), start), stop)
        jb = max(0, int((s - start) * rate))
        je = min(n_samples, int((e - start) * rate))
        if je > jb:
            mask[jb:je] = 1
    return mask


def _igamma_inv_upper(shape: float, p: float) -> float:
    """Inverse of the upper regularized incomplete gamma function.

    Returns x such that the upper regularized incomplete gamma
    Q(shape, x) = p, using scipy's ``gammainccinv``.
    """
    p = float(np.clip(p, 1.0e-12, 1.0 - 1.0e-12))
    s = float(max(shape, 1.0e-12))
    return float(gammainccinv(s, p))


def _get_tf_energy_array(tf_map: Any, edge: float | None = None) -> np.ndarray:
    """
    Return a real-valued TF energy array from a map-like object.

    :param tf_map: time-frequency map exposing `data` and optional `dt`
    :type tf_map: object
    :param edge: optional edge (seconds) cropped on both time sides
    :type edge: float | None
    :return: 2D or 1D float64 energy array
    :rtype: np.ndarray
    """
    arr = np.asarray(tf_map.data)
    if np.iscomplexobj(arr):
        arr = arr.real
    arr = np.asarray(arr, dtype=np.float64)

    if arr.ndim == 2 and edge is not None and hasattr(tf_map, "dt") and tf_map.dt > 0:
        e = int(max(0, round(float(edge) / float(tf_map.dt))))
        if e > 0 and arr.shape[1] > 2 * e:
            arr = arr[:, e:-e]
    return arr


def compute_threshold(tf_maps: list[TimeFrequencyMap], bpp: float, alp: float | None = None, edge: float | None = None) -> float:
    """
    Compute pixel energy threshold from time-frequency map statistics.

    Uses a Python-native implementation inspired by cWB ``network::THRESHOLD``
    logic based on the black-pixel probability (false-alarm rate).

    Parameters
    ----------
    tf_maps : list[TimeFrequencyMap]
        Per-detector time-frequency maps.
    bpp : float
        Black-pixel probability (target false-alarm rate).
    alp : float | None, optional
        Packet-shape scaling parameter (``None`` for the no-shape path).
    edge : float | None, optional
        Edge margin in seconds excluded from threshold statistics.

    Returns
    -------
    float
        Computed pixel energy threshold.
    """
    if not tf_maps or not hasattr(tf_maps[0], "data"):
        raise ValueError("compute_threshold requires TF maps")
    n_ifo = len(tf_maps)

    if alp is not None:
        # C++ THRESHOLD(p, shape) works on flat 1-D time-major data with
        # nL / nR indices, NOT on 2-D edge-cropped arrays.  Replicate that.
        pw0 = tf_maps[0]
        arr0 = np.asarray(pw0.data, dtype=np.float64)
        if np.iscomplexobj(arr0):
            arr0 = arr0.real
        M = int(arr0.shape[0]) if arr0.ndim == 2 else 1
        # C++ stores time-major: flat[t*M + m].  Python (M, T) -> transpose then ravel.
        if arr0.ndim == 2:
            flat0 = arr0.T.ravel()
        else:
            flat0 = arr0.ravel()
        w = flat0.copy()
        for tfm in tf_maps[1:]:
            a = np.asarray(tfm.data, dtype=np.float64)
            if np.iscomplexobj(a):
                a = a.real
            if a.ndim == 2:
                w += a.T.ravel()
            else:
                w += a.ravel()

        nL = int(float(edge or 0.0) * pw0.wavelet_rate * M)
        nR = int(w.size) - nL - 1          # C++: nR = pw->size() - nL - 1

        region = w[nL:nR]                   # C++ loop: for(i=nL; i<nR; i++)
        region = np.clip(region, 0.0, n_ifo * 100.0)
        positive = region[region > 1.0e-3]
        if positive.size == 0:
            return 0.0

        avr = float(np.mean(positive))
        bbb = float(np.mean(np.log(positive)))
        alp_fit = np.log(avr) - bbb
        alp_fit = (3 - alp_fit + np.sqrt((alp_fit - 3) * (alp_fit - 3) + 24 * alp_fit)) / (12 * alp_fit)
        bpp_corr = float(bpp) * alp_fit / float(alp)
        result = avr * _igamma_inv_upper(alp_fit, bpp_corr) / alp_fit / 2.0
        return result

    energies = [_get_tf_energy_array(tfm, edge=edge) for tfm in tf_maps]
    combined = np.sum(energies, axis=0)
    work = combined.ravel()
    if work.size == 0:
        return 0.0

    work = np.clip(work, 0.0, n_ifo * 100.0)
    positive = work[work > 1.0e-3]
    if positive.size == 0:
        return 0.0

    # CWB THRESHOLD(p) exact algorithm: iterative search for Gamma shape m
    # fff = fill fraction (fraction of pixels with energy > 0.0001, matching CWB wavecount)
    fff = float(np.sum(work > 1.0e-4) / work.size)
    if fff <= 0.0:
        return 0.0
    n_total = work.size
    sorted_work = np.sort(work)

    # CWB waveSplit(nL, nR, nR-k): returns (k+1)-th largest in range, i.e., sorted_work[n_total-k-1]
    k_val = int(float(bpp) * fff * n_total)
    k_med = int(0.2 * fff * n_total)
    val = float(sorted_work[max(0, n_total - k_val - 1)]) if k_val > 0 else float(sorted_work[-1])
    med = float(sorted_work[max(0, n_total - k_med - 1)]) if k_med > 0 else float(sorted_work[-1])

    # Find smallest m >= 1.0 (in 0.01 steps) where P(Gamma(N*m) >= med) >= 0.2
    # Matches CWB: while(p00<0.2) {p00 = 1-Gamma(N*m,med); m+=0.01;} if(m>1) m-=0.01;
    m = 1.0
    p00 = 0.0
    while p00 < 0.2:
        p00 = float(_scipy_gammaincc(n_ifo * m, med))
        m += 0.01
    if m > 1.01:
        m -= 0.01

    result = 0.3 * (_igamma_inv_upper(n_ifo * m, float(bpp)) + val) + n_ifo * np.log(m)
    return result


def apply_veto(tf_map: TimeFrequencyMap, tw: float, segment_list: list[tuple[float, float]] | None = None, injection_times: list[float] | None = None, edge: float | None = None, return_mask: bool = False) -> float | tuple[float, np.ndarray]:
    """
    Compute live time and optional veto mask from segments and injections.

    Parameters
    ----------
    tf_map : TimeFrequencyMap
        Reference TF map for timeline definition.
    tw : float
        Injection exclusion window in seconds.
    segment_list : list[tuple[float, float]] | None, optional
        Accepted live segments ``(start, stop)`` in GPS seconds.
    injection_times : list[float] | None, optional
        Injection GPS times to exclude within ±tw/2 seconds.
    edge : float | None, optional
        Edge margin in seconds excluded from live-time integration.
    return_mask : bool, optional
        If True, return ``(live_time, veto_mask)`` instead of just live time.

    Returns
    -------
    float or tuple[float, np.ndarray]
        Live time in seconds. If ``return_mask=True``, returns
        ``(live_time, veto_mask)`` where veto_mask is a 1-D int16 array.
    """
    data = np.asarray(getattr(tf_map, "data", []))
    if data.ndim == 2:
        n_samples = int(data.shape[1])
    else:
        n_samples = int(data.size)
    if n_samples <= 0:
        return (0.0, np.zeros(0, dtype=np.int16)) if return_mask else 0.0

    dt = float(getattr(tf_map, "dt", 0.0))
    if dt <= 0:
        raise ValueError("tf_map.dt must be positive for veto construction")

    rate = 1.0 / dt
    start = float(getattr(tf_map, "start", 0.0))
    stop = float(getattr(tf_map, "stop", start + n_samples * dt))

    veto = np.zeros(n_samples, dtype=np.int16)

    if not segment_list:
        veto[:] = 1
    else:
        for seg_start, seg_stop in segment_list:
            s = min(max(float(seg_start), start), stop)
            e = min(max(float(seg_stop), start), stop)
            jb = max(0, int((s - start) * rate))
            je = min(n_samples, int((e - start) * rate))
            if je > jb:
                veto[jb:je] = 1

    if injection_times:
        w = np.zeros_like(veto)
        tw = max(2.0, float(tw))
        half_window = int(tw * rate / 2.0 + 0.5)
        for gps in injection_times:
            j = int((float(gps) - start) * rate)
            jb = max(0, j - half_window)
            je = min(n_samples, j + half_window)
            if je - jb >= int(rate):  # skip injection windows shorter than 1 s (likely GPS edge artefacts)
                w[jb:je] = 1
        veto = (veto * w).astype(np.int16)

    if edge is None:
        edge = 0.0
    n_edge = int(max(0, edge * rate + 0.5))
    if 2 * n_edge >= n_samples:
        live = 0.0
    else:
        live = float(np.sum(veto[n_edge:n_samples - n_edge])) / rate

    return (live, veto) if return_mask else live


def select_network_pixels(tf_maps: list[TimeFrequencyMap], lag_index: int, energy_threshold: float, lag_shifts: np.ndarray | list | None = None, veto: np.ndarray | None = None, edge: float = 0.0) -> dict:
    """
    Select significant pixels above energy threshold for one lag.

    Parameters
    ----------
    tf_maps : list[TimeFrequencyMap]
        Per-detector time-frequency maps.
    lag_index : int
        Zero-based lag index for time-delay selection.
    energy_threshold : float
        Pixel energy threshold for significance.
    lag_shifts : np.ndarray | list[float] | None, optional
        Per-detector time shifts in seconds for this lag.
    veto : np.ndarray | None, optional
        Binary veto array in time bins (1 keep, 0 reject).
    edge : float, optional
        Edge margin in seconds. Default is 0.0.

    Returns
    -------
    dict
        Candidate payload with keys:

        - ``mask``           : bool ndarray, shape (n_freq, n_time) — pixel selection mask
        - ``time``           : int64 ndarray, shape (n_pix,) — time-bin indices of selected pixels
        - ``frequency``      : int64 ndarray, shape (n_pix,) — frequency-bin indices
        - ``energy``         : float64 ndarray, shape (n_pix,) — summed energy across detectors
        - ``pix_det_energy`` : float64 ndarray, shape (n_pix, n_ifo) — per-detector pixel energy
        - ``pix_det_index``  : int64 ndarray, shape (n_pix, n_ifo) — flat TF index per detector
        - ``rate``           : float — sample rate (Hz)
        - ``layers``         : int — number of WDM frequency layers
        - ``start``          : float — segment GPS start
        - ``stop``           : float — segment GPS stop
        - ``f_low``          : float | None — low-frequency edge
        - ``f_high``         : float | None — high-frequency edge
    """
    if not tf_maps or not hasattr(tf_maps[0], "data"):
        raise ValueError("select_network_pixels requires TF maps")
    arrays = [_get_tf_energy_array(tfm, edge=None) for tfm in tf_maps]
    if not all(arr.ndim == 2 for arr in arrays):
        raise ValueError("python get_network_pixels expects 2D TF arrays")

    n_ifo = len(arrays)
    n_freq, n_time = arrays[0].shape
    dt = float(tf_maps[0].dt)

    if lag_shifts is None:
        shifts_sec = np.zeros(n_ifo, dtype=float)
    else:
        shifts_sec = np.asarray(lag_shifts, dtype=float)
        if shifts_sec.size != n_ifo:
            raise ValueError("lag_shifts size mismatch with number of detectors")
    ref = min(shifts_sec)
    rate = 1.0 / dt
    shift_bins = [int((s - ref) * rate + 0.001) for s in shifts_sec]

    edge_bins = int(max(0, float(edge) * rate + 0.001))
    valid_start = edge_bins
    valid_stop = n_time - edge_bins
    nn_valid = valid_stop - valid_start

    # Pre-compute frequency band limits (passed to the Numba kernel)
    f_low = float(getattr(tf_maps[0], "f_low", 0.0) or 0.0)
    f_high_attr = getattr(tf_maps[0], "f_high", None)
    f_high = float(f_high_attr) if f_high_attr is not None else float((n_freq - 1) * tf_maps[0].df)
    df = float(getattr(tf_maps[0], "df", 0.0) or 0.0)

    ib = 1
    ie = n_freq
    if df > 0:
        freqs = np.arange(n_freq, dtype=np.float64) * df
        for idx_f, freq in enumerate(freqs):
            if freq <= f_high:
                ie = idx_f
            if freq <= f_low:
                ib = idx_f + 1
    ie = min(ie, n_freq - 1)
    ib = max(ib, 1)

    eo = float(energy_threshold)
    em = 2.0 * eo
    eh = em * em

    # Build detector stack once; reused below by _compute_pixel_data_arrays
    arrays_stack   = np.stack(arrays, axis=0).astype(np.float64)  # (n_ifo, n_freq, n_time)
    shift_bins_arr = np.asarray(shift_bins, dtype=np.int64)
    veto_arr = (veto.astype(np.int16) if (veto is not None and len(veto) == n_time)
                else np.zeros(0, dtype=np.int16))

    # Merged align + threshold + pixel selection in one Numba pass
    combined_raw, selected = _align_threshold_select_numba(
        arrays_stack, shift_bins_arr,
        int(valid_start), int(nn_valid),
        veto_arr, (veto is not None and len(veto) == n_time),
        int(edge_bins), int(ib), int(ie),
        eo, em, eh,
    )
    freq_idx, time_idx = np.where(selected)
    values = combined_raw[freq_idx, time_idx]

    # Precompute per-(pixel, detector) numeric values with Numba (avoids inner Python loop)
    # arrays_stack and shift_bins_arr already built above the kernel call
    if len(freq_idx) > 0:
        pix_det_energy, pix_det_index = _compute_pixel_data_arrays(
            freq_idx.astype(np.int64), time_idx.astype(np.int64),
            arrays_stack.astype(np.float64),
            shift_bins_arr,
            int(valid_start), int(valid_stop), int(nn_valid), int(n_freq),
        )
    else:
        pix_det_energy = np.empty((0, n_ifo), dtype=np.float64)
        pix_det_index  = np.empty((0, n_ifo), dtype=np.int64)

    tf0 = tf_maps[0]
    return {
        "mask": selected,
        "time": time_idx,
        "frequency": freq_idx,
        "energy": values,
        "pix_det_energy": pix_det_energy,  # (n_pix, n_ifo) float64: energy per pixel per detector
        "pix_det_index": pix_det_index,    # (n_pix, n_ifo) int64: TF index per pixel per detector
        "rate": float(1.0 / dt),
        "layers": n_freq,
        "start": float(getattr(tf0, "start", 0.0)),
        "stop": float(getattr(tf0, "stop", 0.0)),
        "f_low": float(getattr(tf0, "f_low", 0.0) or 0.0),
        "f_high": float(getattr(tf0, "f_high", (n_freq - 1) * float(getattr(tf0, "df", 0.0))) or 0.0),
    }


def cluster_pixels(pixel_candidates: dict, kt: int = 1, kf: int = 1) -> FragmentCluster:
    """
    Cluster selected pixels using connected-component analysis.

    Parameters
    ----------
    pixel_candidates : dict
        Candidate payload from :func:`select_network_pixels`.
    kt : int
        Time-connectivity radius in bins (adjacency tolerance |Δtime| ≤ kt).
    kf : int
        Frequency-connectivity radius in bins (adjacency tolerance |Δfreq| ≤ kf).

    Returns
    -------
    FragmentCluster
        Clustered pixels with selected and rejected flags based on size.
    """
    if pixel_candidates is None:
        raise ValueError("cluster_pixels requires pixel_candidates")
    mask = np.asarray(pixel_candidates["mask"], dtype=bool)

    kt = int(max(1, kt))
    kf = int(max(1, kf))

    f_idx_arr      = np.asarray(pixel_candidates.get("frequency", []), dtype=np.int64)
    t_idx_arr      = np.asarray(pixel_candidates.get("time", []), dtype=np.int64)
    pix_det_energy = pixel_candidates.get("pix_det_energy", np.empty((0, 0), dtype=np.float64))
    pix_det_index  = pixel_candidates.get("pix_det_index",  np.empty((0, 0), dtype=np.int64))
    energy_arr     = np.asarray(pixel_candidates.get("energy", []), dtype=np.float64)
    layers = int(pixel_candidates.get("layers", 1))
    rate   = float(pixel_candidates.get("rate", 0.0))
    dt     = 1.0 / rate if rate > 0.0 else 1.0
    n_ifo  = int(pix_det_energy.shape[1]) if pix_det_energy.ndim == 2 and pix_det_energy.shape[1] > 0 else 0
    n_pix  = len(f_idx_arr)

    if n_pix == 0:
        clusters = []
    else:
        # --- Connected-components labeling with rectangular (kf, kt) connectivity ---
        # scipy.ndimage.label requires structure shape (3, 3) for 2D inputs in
        # newer scipy versions, so we build a sparse adjacency graph instead.
        # Two pixels are directly connected if |Δfreq| ≤ kf AND |Δtime| ≤ kt.

        # Build sparse adjacency via Numba COO builder (avoids O(n²) dense arrays)
        if n_pix > 0:
            rows_coo, cols_coo = _build_adj_coo(f_idx_arr, t_idx_arr, kf, kt)
            n_edges  = len(rows_coo)
            all_rows = np.concatenate([rows_coo, cols_coo])
            all_cols = np.concatenate([cols_coo, rows_coo])
            adj = csr_matrix(
                (np.ones(2 * n_edges, dtype=np.bool_), (all_rows, all_cols)),
                shape=(n_pix, n_pix),
            )
            _, raw_labels = _cc(adj, directed=False, connection='weak')
            # raw_labels is 0-indexed; shift to 1-indexed to match ndimage convention
            raw_labels = raw_labels + 1
        else:
            raw_labels = np.array([], dtype=np.int64)

        # Build a labeled 2D array (same shape as mask) for downstream code
        labeled = np.zeros(mask.shape, dtype=np.int64)
        labeled[f_idx_arr, t_idx_arr] = raw_labels

        # Group pixel indices by their 2D label (O(n))
        grouped: dict = {}
        for pix_idx in range(n_pix):
            lbl = int(labeled[f_idx_arr[pix_idx], t_idx_arr[pix_idx]])
            if lbl > 0:
                grouped.setdefault(lbl, []).append(pix_idx)

        # Batch subnet/subrho: one Numba call across all clusters instead of ~n_clusters calls.
        group_list = [g for g in grouped.values() if g]
        n_groups = len(group_list)
        if n_groups > 0 and n_ifo > 1:
            n_sub_c = 2.0 * _igamma_inv_upper(float(n_ifo - 1), 0.314)
            all_pix_idx = np.array([pid for g in group_list for pid in g], dtype=np.int64)
            asnr_all = np.sqrt(pix_det_energy[all_pix_idx])   # (n_flat, n_ifo)
            noise_rms_all = np.ones_like(asnr_all)            # noise_rms=1.0 at this stage
            sizes = np.array([len(g) for g in group_list], dtype=np.int64)
            offsets_arr = np.zeros(n_groups + 1, dtype=np.int64)
            offsets_arr[1:] = np.cumsum(sizes)
            subnet_arr, subrho_arr = _subnet_subrho_batch_numba(
                asnr_all, noise_rms_all, offsets_arr, n_sub_c
            )
        else:
            subnet_arr = np.zeros(n_groups, dtype=np.float64)
            subrho_arr = np.zeros(n_groups, dtype=np.float64)

        # Construct Cluster objects; build Pixel objects here (deferred from select_network_pixels)
        clusters = []
        for c_idx, group_indices in enumerate(group_list):
            idx_arr = np.array(group_indices, dtype=np.int64)
            group_pixels = []
            for pid in group_indices:
                f_i = int(f_idx_arr[pid])
                t_i = int(t_idx_arr[pid])
                pixel_data_list = [
                    PixelData(
                        noise_rms=1.0,
                        wave=0.0, w_90=0.0,
                        asnr=float(np.sqrt(pix_det_energy[pid, d])),
                        a_90=0.0, rank=0.0,
                        index=int(pix_det_index[pid, d]),
                    )
                    for d in range(n_ifo)
                ]
                group_pixels.append(Pixel(
                    time=int(t_i * layers + f_i),
                    frequency=f_i,
                    layers=layers,
                    rate=rate,
                    likelihood=float(energy_arr[pid]),
                    null=0.0, theta=0.0, phi=0.0,
                    ellipticity=0.0, polarisation=0.0,
                    core=1, data=pixel_data_list, td_amp=[], neighbors=[],
                ))
            energy = float(energy_arr[idx_arr].sum())
            c_time = float(np.mean(t_idx_arr[idx_arr].astype(float) * dt - dt / 2))
            c_freq = float(np.mean(f_idx_arr[idx_arr].astype(float) * (rate / 2)))
            cluster_meta = ClusterMeta(
                energy=energy,
                like_net=energy,
                sub_net=float(subnet_arr[c_idx]),
                net_rho=float(subrho_arr[c_idx]),
                c_time=c_time,
                c_freq=c_freq,
            )
            clusters.append(Cluster(pixels=group_pixels, cluster_meta=cluster_meta))

    n_pix_final = int(sum(len(c.pixels) for c in clusters))
    logger.info("cluster_pixels: n_clusters=%d n_pix=%d", len(clusters), n_pix_final)
    return FragmentCluster(
        rate=float(pixel_candidates.get("rate", 0.0)),
        start=float(pixel_candidates.get("start", 0.0)),
        stop=float(pixel_candidates.get("stop", 0.0)),
        bpp=1.0,
        shift=0.0,
        f_low=float(pixel_candidates.get("f_low", 0.0)),
        f_high=float(pixel_candidates.get("f_high", 0.0)),
        n_pix=n_pix_final,
        run=0,
        pair=False,
        subnet_threshold=0.0,
        clusters=clusters,
    )


# ---------------------------------------------------------------------------
# Numba JIT helpers (compiled once on first call; cached to disk)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _build_adj_coo(f_arr: np.ndarray, t_arr: np.ndarray, kf: int, kt: int) -> tuple[np.ndarray, np.ndarray]:
    """Build COO edge list for the pixel neighbourhood graph.

    Two pixels are adjacent when |Δfreq| ≤ kf AND |Δtime| ≤ kt.
    Sorts pixels by (f, t) so the inner loop can break early once
    Δf > kf, reducing O(n²) to O(n × avg_neighbors_in_freq_band).
    """
    n = len(f_arr)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    # Sort by f (primary), t (secondary) via a combined key
    max_t_val = np.max(t_arr) + 1
    sort_key = f_arr * max_t_val + t_arr
    order = np.argsort(sort_key)
    sf = f_arr[order]
    st = t_arr[order]

    # Two-pass: count then fill.
    # After sorting, sf[j] >= sf[i] for j > i, so we break as soon as sf[j] - sf[i] > kf.
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if sf[j] - sf[i] > kf:
                break
            if abs(st[j] - st[i]) <= kt:
                count += 1
    rows = np.empty(count, dtype=np.int64)
    cols = np.empty(count, dtype=np.int64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            if sf[j] - sf[i] > kf:
                break
            if abs(st[j] - st[i]) <= kt:
                rows[k] = order[i]
                cols[k] = order[j]
                k += 1
    return rows, cols


@njit(cache=True)
def _compute_pixel_data_arrays(freq_idx: np.ndarray, time_idx: np.ndarray, arrays_stack: np.ndarray,
                                shift_bins: np.ndarray, valid_start: int, valid_stop: int,
                                nn_valid: int, n_freq: int) -> tuple[np.ndarray, np.ndarray]:
    """Precompute per-(pixel, detector) energy and flat index values.

    Parameters
    ----------
    freq_idx, time_idx : int64[n_pix]
    arrays_stack       : float64[n_ifo, n_freq, n_time]
    shift_bins         : int64[n_ifo]
    valid_start, valid_stop, nn_valid, n_freq : int

    Returns
    -------
    det_energy : float64[n_pix, n_ifo]
    det_index  : int64[n_pix, n_ifo]
    """
    n_pix = len(freq_idx)
    n_ifo = arrays_stack.shape[0]
    det_energy = np.empty((n_pix, n_ifo), dtype=np.float64)
    det_index  = np.empty((n_pix, n_ifo), dtype=np.int64)
    for idx in range(n_pix):
        f_i = freq_idx[idx]
        t_i = time_idx[idx]
        for d in range(n_ifo):
            if nn_valid > 0 and valid_start <= t_i < valid_stop:
                u     = t_i - valid_start
                det_t = valid_start + (u + shift_bins[d]) % nn_valid
            else:
                det_t = t_i
            e = arrays_stack[d, f_i, det_t]
            det_energy[idx, d] = e if e > 0.0 else 0.0
            det_index[idx, d]  = det_t * n_freq + f_i
    return det_energy, det_index


@njit(cache=True)
def _subnet_subrho_numba(asnr_arr: np.ndarray, noise_rms_arr: np.ndarray, n_sub: float) -> tuple[float, float]:
    """Compute subnet and subrho statistics for one cluster.

    Parameters
    ----------
    asnr_arr      : float64[n_pix, n_ifo]
    noise_rms_arr : float64[n_pix, n_ifo]
    n_sub         : float  — threshold constant 2 * iGamma^{-1}(n_ifo-1, 0.314)

    Returns
    -------
    subnet : float
    subrho : float
    """
    n_pix = asnr_arr.shape[0]
    n_ifo = asnr_arr.shape[1]
    rho         = 0.0
    e_sub_total = 0.0
    e_max_sum   = 0.0
    subnet_acc  = 0.0
    for p in range(n_pix):
        amp_sum = 0.0
        e_max   = 0.0
        e_tot   = 0.0
        nsd     = 0.0
        msd     = 0.0
        for d in range(n_ifo):
            amp = asnr_arr[p, d]
            x   = amp * amp
            v   = noise_rms_arr[p, d] * noise_rms_arr[p, d]
            amp_sum += abs(amp)
            if x > e_max:
                e_max = x
                msd   = v
            e_tot += x
            if v > 0.0:
                nsd += 1.0 / v
        a_fac = (e_tot / (amp_sum * amp_sum)) if amp_sum > 0.0 else 1.0
        rho   += (1.0 - a_fac) * (e_tot - n_sub * 2.0)
        y_val  = e_tot - e_max
        x_corr = y_val * (1.0 + y_val / (e_max + 1.0e-5))
        if msd > 0.0:
            nsd -= 1.0 / msd
        v_corr = (2.0 * e_max - e_tot) * msd * nsd / 10.0
        e_sub_total += e_tot - e_max
        e_max_sum   += e_max
        a_cut  = x_corr / (x_corr + n_sub) if (x_corr + n_sub) != 0.0 else 0.0
        denom  = x_corr + (v_corr if v_corr > 0.0 else 1.0e-5)
        subnet_acc += (e_tot * x_corr / denom) * (a_cut if a_cut > 0.5 else 0.0)
    subnet = subnet_acc / (e_max_sum + e_sub_total + 0.01)
    subrho = np.sqrt(rho) if rho >= 0.0 else np.nan
    return subnet, subrho


@njit(cache=True)
def _subnet_subrho_batch_numba(
    asnr_all: np.ndarray,
    noise_rms_all: np.ndarray,
    offsets: np.ndarray,
    n_sub: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute subnet and subrho for all clusters in one Numba pass.

    Parameters
    ----------
    asnr_all      : float64[n_pix_total, n_ifo]  — flattened across all clusters
    noise_rms_all : float64[n_pix_total, n_ifo]
    offsets       : int64[n_clusters + 1]  — CSR row-pointer style start/end per cluster
    n_sub         : float  — 2 * iGamma^{-1}(n_ifo-1, 0.314), same for all clusters

    Returns
    -------
    subnet_arr : float64[n_clusters]
    subrho_arr : float64[n_clusters]
    """
    n_clusters = len(offsets) - 1
    subnet_arr = np.empty(n_clusters, dtype=np.float64)
    subrho_arr = np.empty(n_clusters, dtype=np.float64)
    for c in range(n_clusters):
        s = offsets[c]
        e = offsets[c + 1]
        subnet_arr[c], subrho_arr[c] = _subnet_subrho_numba(
            asnr_all[s:e], noise_rms_all[s:e], n_sub
        )
    return subnet_arr, subrho_arr


@njit(cache=True)
def _align_threshold_select_numba(
    arrays_stack: np.ndarray,
    shift_bins: np.ndarray,
    valid_start: int,
    nn_valid: int,
    veto: np.ndarray,
    has_veto: bool,
    edge_bins: int,
    ib: int,
    ie: int,
    eo: float,
    em: float,
    eh: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Align detectors, apply thresholds, and select pixels in one Numba pass.

    Replaces the Python ``align`` and ``thresh`` phases of
    :func:`select_network_pixels`, eliminating all intermediate array
    allocations (no temporary ``aligned`` list, no broadcast copies).

    Parameters
    ----------
    arrays_stack : float64[n_ifo, n_freq, n_time]
    shift_bins   : int64[n_ifo]
    valid_start  : int
    nn_valid     : int
    veto         : int16[n_time] or empty  — 0 = reject, 1 = keep
    has_veto     : bool
    edge_bins    : int
    ib, ie       : int  — inclusive first / exclusive last valid freq indices
    eo, em, eh   : float  — energy threshold, hard cap (2*eo), cap² (em²)

    Returns
    -------
    combined_raw : float64[n_freq, n_time]  — raw shifted sum (unclipped)
    selected     : bool[n_freq, n_time]     — pixel selection mask
    """
    n_ifo  = arrays_stack.shape[0]
    n_freq = arrays_stack.shape[1]
    n_time = arrays_stack.shape[2]

    combined_raw = np.zeros((n_freq, n_time), dtype=np.float64)

    # Phase 1: shift-align each detector and accumulate into combined_raw
    for fi in range(n_freq):
        for t in range(valid_start, valid_start + nn_valid):
            u = t - valid_start
            v = 0.0
            for d in range(n_ifo):
                src_t = valid_start + (u + shift_bins[d]) % nn_valid
                v += arrays_stack[d, fi, src_t]
            combined_raw[fi, t] = v

    # Phase 2: copy → apply veto / edge-zeroing / freq-band / threshold clipping
    combined = combined_raw.copy()

    if has_veto:
        for t in range(n_time):
            if veto[t] == 0:
                for fi in range(n_freq):
                    combined[fi, t] = 0.0

    for t in range(edge_bins):
        for fi in range(n_freq):
            combined[fi, t] = 0.0
    for t in range(n_time - edge_bins, n_time):
        for fi in range(n_freq):
            combined[fi, t] = 0.0

    for fi in range(ib):
        for t in range(n_time):
            combined[fi, t] = 0.0

    for fi in range(n_freq):
        for t in range(n_time):
            v = combined[fi, t]
            if v < eo:
                combined[fi, t] = 0.0
            elif v > em:
                combined[fi, t] = em + 0.1

    # Phase 3: neighbourhood support test (pixel selection)
    ii     = n_freq - 2
    margin = max(edge_bins, 2)
    t_s    = margin
    t_e    = n_time - margin
    f_s    = ib
    f_e    = min(max(ie, f_s), n_freq - 1)

    selected = np.zeros((n_freq, n_time), dtype=np.bool_)

    for fi in range(f_s, f_e):
        for t in range(t_s, t_e):
            e_val = combined[fi, t]
            if e_val < eo:
                continue

            ct = combined[fi + 1, t] + combined[fi, t + 1] + combined[fi + 1, t + 1]
            cb = combined[fi - 1, t] + combined[fi, t - 1] + combined[fi - 1, t - 1]

            ht = combined[fi + 1, t + 2]
            if fi < ii:
                ht += combined[fi + 2, t + 2] + combined[fi + 2, t + 1]

            hb = combined[fi - 1, t - 2]
            if fi >= 2:
                hb += combined[fi - 2, t - 2] + combined[fi - 2, t - 1]

            if not ((ct + cb) * e_val < eh
                    and (ct + ht) * e_val < eh
                    and (cb + hb) * e_val < eh
                    and e_val < em):
                selected[fi, t] = True

    return combined_raw, selected
