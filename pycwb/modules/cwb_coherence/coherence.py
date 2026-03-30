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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Numba JIT helpers (compiled once on first call; cached to disk)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _build_adj_coo(f_arr: np.ndarray, t_arr: np.ndarray, kf: int, kt: int) -> tuple[np.ndarray, np.ndarray]:
    """Build COO edge list for the pixel neighbourhood graph.

    Two pixels are adjacent when |Δfreq| ≤ kf AND |Δtime| ≤ kt.
    Returns upper-triangle (rows, cols) int64 index arrays.  Two-pass
    approach avoids Python lists and allocates exact-sized output.
    """
    n = len(f_arr)
    # First pass: count matching pairs
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(f_arr[i] - f_arr[j]) <= kf and abs(t_arr[i] - t_arr[j]) <= kt:
                count += 1
    # Second pass: fill
    rows = np.empty(count, dtype=np.int64)
    cols = np.empty(count, dtype=np.int64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(f_arr[i] - f_arr[j]) <= kf and abs(t_arr[i] - t_arr[j]) <= kt:
                rows[k] = i
                cols[k] = j
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
    timer_start = time.perf_counter()
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

    logger.info("----------------------------------------")
    logger.info("Coherence time totally: %.2f s", time.perf_counter() - timer_start)
    logger.info("----------------------------------------")

    return result


# ---------------------------------------------------------------------------
# Streaming-friendly API: setup once, iterate over lags
# ---------------------------------------------------------------------------

def setup_coherence(config: Config, strains: list[TimeSeries], job_seg: WaveSegment | None = None) -> list[dict]:
    """
    Compute all lag-independent coherence data.

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
    timer_start = time.perf_counter()

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

    logger.info("Coherence setup time: %.2f s", time.perf_counter() - timer_start)
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
    except Exception as exc:
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
        config.bpp,
        alp if pattern != 0 else None,
        tf_maps=tf_maps,
        edge=config.segEdge,
    )

    # Extract lag count from job segment for setup dictionary
    n_lag = job_seg.n_lag

    logger.info(
        "level %d setup done: Eo=%.4g, n_lag=%d  (%.2f s)",
        level, Eo, n_lag, time.perf_counter() - timer_start,
    )

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
        t_sel = time.perf_counter()
        veto = None
        if veto_windows is not None:
            veto = build_veto_mask(tf_maps[0], veto_windows, edge=setup["segEdge"])
        candidates = select_network_pixels(
            lag_index=lag_idx,
            energy_threshold=Eo,
            tf_maps=tf_maps,
            lag_shifts=job_seg.lag_shifts[lag_idx],
            veto=veto,
            edge=setup["segEdge"],
        )
        t_sel_elapsed = time.perf_counter() - t_sel
        n_candidates = (
            len(candidates["pixels"])
            if isinstance(candidates, dict) and "pixels" in candidates
            else -1
        )

        # Cluster selected pixels and apply statistical selection criteria
        # (min/max cluster sizes depend on wave pattern)
        t_cl = time.perf_counter()
        if pattern != 0:
            # Multi-pixel clusters for network patterns (size 2-3 pixels)
            c = cluster_pixels(min_size=2, max_size=3, pixel_candidates=candidates)
            c.select("subrho", setup["select_subrho"])
            c.select("subnet", setup["select_subnet"])
        else:
            # Single-pixel clusters for non-network patterns
            c = cluster_pixels(min_size=1, max_size=1, pixel_candidates=candidates)
        t_cl_elapsed = time.perf_counter() - t_cl

        # Remove clusters rejected by statistical selection unless explicitly requested
        if not return_rejected:
            c.remove_rejected()

        logger.info(
            "lag=%3d level=%d |%9d |%7d | cand=%d sel=%.4fs clust=%.4fs",
            lag_idx,
            setup["level"],
            c.event_count(),
            c.pixel_count(),
            n_candidates,
            t_sel_elapsed,
            t_cl_elapsed,
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


def compute_threshold(bpp: float, alp: float | None = None, tf_maps: list | None = None, edge: float | None = None) -> float:
    """
    Compute pixel energy threshold from time-frequency map statistics.

    Uses a Python-native implementation inspired by cWB `network::THRESHOLD`
    logic based on the black-pixel probability (false-alarm rate).

    Parameters
    ----------
    bpp : float
        Black-pixel probability (target false-alarm rate).
    alp : float | None, optional
        Optional packet-shape reference value for scaling.
    tf_maps : list[TimeFrequencyMap] | None, optional
        List of python TF maps for computing threshold statistics.
    edge : float | None, optional
        Edge margin in seconds for excluding boundary data.

    Returns
    -------
    float
        Computed pixel energy threshold.
    """
    if tf_maps is None or len(tf_maps) == 0 or not hasattr(tf_maps[0], "data"):
        raise ValueError("compute_threshold requires python TF maps")
    return _threshold_python(tf_maps, bpp=bpp, shape=alp, edge=edge)


def select_network_pixels(lag_index: int, energy_threshold: float, tf_maps: list | None = None, lag_shifts: np.ndarray | list | None = None, veto: np.ndarray | None = None, edge: float = 0.0) -> dict:
    """
    Select significant pixels above energy threshold.

    Parameters
    ----------
    lag_index : int
        Zero-based lag index for time-delay selection.
    energy_threshold : float
        Pixel energy threshold for significance.
    tf_maps : list[TimeFrequencyMap] | None, optional
        List of python TF maps (required for pixel selection).
    lag_shifts : np.ndarray | list[float] | None, optional
        Per-detector time shifts in seconds for this lag.
    veto : np.ndarray | None, optional
        Binary veto array in time bins (1 keep, 0 reject).
    edge : float, optional
        Edge margin in seconds. Default is 0.0.

    Returns
    -------
    dict
        Candidate payload with selected mask, coordinates, energies, and pixels.
    """
    if tf_maps is None or len(tf_maps) == 0 or not hasattr(tf_maps[0], "data"):
        raise ValueError("select_network_pixels requires python TF maps")
    return _get_network_pixels_python(
        tf_maps=tf_maps,
        lag_index=lag_index,
        energy_threshold=energy_threshold,
        lag_shifts=lag_shifts,
        veto=veto,
        edge=edge,
    )


def cluster_pixels(min_size: int, max_size: int, pixel_candidates: dict | None = None) -> FragmentCluster:
    """
    Cluster selected pixels using connected-component analysis.

    Parameters
    ----------
    min_size : int
        Minimum number of pixels in a cluster (typically 1-2).
    max_size : int
        Maximum number of pixels in a cluster (typically 2-3).
    pixel_candidates : dict | None, optional
        Candidate payload from :func:`select_network_pixels` (required).

    Returns
    -------
    FragmentCluster
        Clustered pixels with selected and rejected flags based on size.
    """
    if pixel_candidates is None:
        raise ValueError("cluster_pixels requires python pixel_candidates")
    return _cluster_pixels_python(pixel_candidates, kt=min_size, kf=max_size)
    

def apply_veto(iwindow: int, tf_map: Any, segment_list: list[tuple[float, float]] | None = None, injection_times: list[float] | None = None, edge: float | None = None, return_mask: bool = False) -> float | tuple[float, np.ndarray]:
    """
    Compute live time and optional veto mask from segments and injections.

    Parameters
    ----------
    iwindow : int
        Veto window size in time bins.
    tf_map : object
        Reference TF map for timeline definition.
    segment_list : list[tuple[float, float]] | None, optional
        List of (start, stop) GPS segments to keep.
    injection_times : list[float] | None, optional
        List of injection GPS times to exclude.
    edge : float | None, optional
        Edge margin in seconds for live-time integration.
    return_mask : bool, optional
        If True, return both live time and veto mask. Default is False.

    Returns
    -------
    float or tuple[float, np.ndarray]
        Live time in seconds (zero lag). If ``return_mask=True``, returns
        ``(live_time, veto_mask)`` where veto_mask is a binary array.
    """
    live, veto_mask = _set_veto_python(
        tf_map=tf_map,
        tw=float(iwindow),
        segment_list=segment_list,
        injection_times=injection_times,
        edge=edge,
    )
    return (live, veto_mask) if return_mask else live


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
    """Upper-tail inverse incomplete gamma matching C++ iGamma step search.

    C++ scans x = 0, 1e-5, 2e-5, ... using TMath::Gamma until the upper
    regularized incomplete gamma drops to *p*.  We start near the scipy result
    and search locally using ROOT's TMath::Gamma for exact agreement.
    Falls back to scipy-only if ROOT is unavailable.
    """
    p = float(np.clip(p, 1.0e-12, 1.0 - 1.0e-12))
    s = float(max(shape, 1.0e-12))

    try:
        import ROOT as _ROOT
        _tmath_gamma = _ROOT.TMath.Gamma
    except Exception:
        return float(gammainccinv(s, p))

    x_approx = float(gammainccinv(s, p))
    step = 1.0e-5
    k = max(0, int(x_approx / step) - 2)
    x = k * step
    while 1.0 - _tmath_gamma(s, x) > p:
        x += step
    return x


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


def _threshold_python(tf_maps: list, bpp: float, shape: float | None = None, edge: float | None = None) -> float:
    """
    Python-native approximation of cWB threshold estimation.

    :param tf_maps: detector TF maps
    :type tf_maps: list
    :param bpp: black pixel probability
    :type bpp: float
    :param shape: optional packet-shape parameter
    :type shape: float | None
    :param edge: optional edge (seconds) for array extraction
    :type edge: float | None
    :return: detection threshold in energy units
    :rtype: float
    """
    n_ifo = len(tf_maps)

    if shape is not None:
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
        alp = np.log(avr) - bbb
        alp = (3 - alp + np.sqrt((alp - 3) * (alp - 3) + 24 * alp)) / (12 * alp)
        bpp_corr = float(bpp) * alp / float(shape)
        result = avr * _igamma_inv_upper(alp, bpp_corr) / alp / 2.0
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
    from scipy.special import gammaincc as _scipy_gammaincc
    m = 1.0
    p00 = 0.0
    while p00 < 0.2:
        p00 = float(_scipy_gammaincc(n_ifo * m, med))
        m += 0.01
    if m > 1.01:
        m -= 0.01

    result = 0.3 * (_igamma_inv_upper(n_ifo * m, float(bpp)) + val) + n_ifo * np.log(m)
    return result


def _set_veto_python(tf_map: Any, tw: float, segment_list: list[tuple[float, float]] | None = None, injection_times: list[float] | None = None, edge: float | None = None) -> tuple[float, np.ndarray]:
    """
    Build a veto mask and live-time estimate for a TF map timeline.

    :param tf_map: reference TF map
    :type tf_map: object
    :param tw: injection veto window (seconds)
    :type tw: float
    :param segment_list: accepted live segments `(start, stop)` in GPS seconds
    :type segment_list: list[tuple[float, float]] | None
    :param injection_times: injection GPS times
    :type injection_times: list[float] | None
    :param edge: edge excluded from live-time computation (seconds)
    :type edge: float | None
    :return: `(live_time, veto_mask)`
    :rtype: tuple[float, np.ndarray]
    """
    data = np.asarray(getattr(tf_map, "data", []))
    if data.ndim == 2:
        n_samples = int(data.shape[1])
    else:
        n_samples = int(data.size)
    if n_samples <= 0:
        return 0.0, np.zeros(0, dtype=np.int16)

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
            if je - jb >= int(rate):
                w[jb:je] = 1
        veto = (veto * w).astype(np.int16)

    if edge is None:
        edge = 0.0
    n_edge = int(max(0, edge * rate + 0.5))
    if 2 * n_edge >= n_samples:
        live = 0.0
    else:
        live = float(np.sum(veto[n_edge:n_samples - n_edge])) / rate

    return live, veto


def _get_network_pixels_python(tf_maps: list, lag_index: int, energy_threshold: float, lag_shifts: np.ndarray | list | None = None, veto: np.ndarray | None = None, edge: float = 0.0) -> dict:
    """
    Select coherent network pixels for one lag in pure Python.

    :param tf_maps: per-detector TF maps
    :type tf_maps: list
    :param lag_index: lag index (stored for compatibility)
    :type lag_index: int
    :param energy_threshold: base energy threshold
    :type energy_threshold: float
    :param lag_shifts: per-detector lag shifts in seconds
    :type lag_shifts: np.ndarray | list[float] | None
    :param veto: optional time veto mask
    :type veto: np.ndarray | None
    :param edge: optional edge cut in seconds
    :type edge: float
    :return: candidate payload for clustering
    :rtype: dict
    """
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

    aligned = [np.zeros_like(arrays[i]) for i in range(n_ifo)]
    if nn_valid > 0:
        out_idx = np.arange(nn_valid, dtype=np.int64)
        for det_idx in range(n_ifo):
            src_idx = valid_start + ((out_idx + shift_bins[det_idx]) % nn_valid)
            aligned[det_idx][:, valid_start:valid_stop] = arrays[det_idx][:, src_idx]

    combined_raw = np.sum(aligned, axis=0)
    combined = np.array(combined_raw, copy=True)

    if veto is not None and len(veto) == n_time:
        combined = combined * veto.reshape(1, -1)

    if edge_bins > 0 and n_time > 2 * edge_bins:
        combined[:, :edge_bins] = 0.0
        combined[:, -edge_bins:] = 0.0

    eo = float(energy_threshold)
    em = 2.0 * eo
    eh = em * em

    # cWB-like frequency band handling
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

    # cWB-like thresholding and loud-pixel degradation before support test
    combined[:ib, :] = 0.0
    combined[combined < eo] = 0.0
    combined[combined > em] = em + 0.1

    selected = np.zeros_like(combined, dtype=bool)
    ii = n_freq - 2

    # Need +/-2 time halo and +/-2 freq accesses in the same pattern as cWB
    t_start = max(edge_bins, 2)
    t_end = n_time - max(edge_bins, 2)
    f_start = ib
    f_end = min(max(ie, f_start), n_freq - 1)

    # --- Vectorized pixel selection (replaces the double Python for-loop) ---
    # All slices operate on the subregion [f_start:f_end, t_start:t_end]
    if f_end > f_start and t_end > t_start:
        # Core energy in the valid region
        e_val = combined[f_start:f_end, t_start:t_end]          # (nf, nt)

        # Only examine pixels above the base threshold
        above_thresh = e_val >= eo                                # (nf, nt)

        # Neighbourhood sums (ct = top halo, cb = bottom halo)
        ct = (combined[f_start + 1:f_end + 1, t_start:t_end]
              + combined[f_start:f_end,     t_start + 1:t_end + 1]
              + combined[f_start + 1:f_end + 1, t_start + 1:t_end + 1])  # (nf, nt)

        cb = (combined[f_start - 1:f_end - 1, t_start:t_end]
              + combined[f_start:f_end,     t_start - 1:t_end - 1]
              + combined[f_start - 1:f_end - 1, t_start - 1:t_end - 1])  # (nf, nt)

        # ht: base term always present, extra terms only when f < ii (= n_freq-2)
        ht = combined[f_start + 1:f_end + 1, t_start + 2:t_end + 2].copy()  # (nf, nt)
        f_end_ht = min(f_end, ii)        # rows [f_start, f_end_ht) have extra ht
        nf_ht = f_end_ht - f_start
        if nf_ht > 0:
            ht[:nf_ht] += (combined[f_start + 2:f_end_ht + 2, t_start + 2:t_end + 2]
                           + combined[f_start + 2:f_end_ht + 2, t_start + 1:t_end + 1])

        # hb: base term always present, extra terms only when f > 1
        hb = combined[f_start - 1:f_end - 1, t_start - 2:t_end - 2].copy()  # (nf, nt)
        f_start_hb = max(f_start, 2)    # rows [f_start_hb, f_end) have extra hb
        nf_hb_skip = f_start_hb - f_start
        if f_start_hb < f_end:
            hb[nf_hb_skip:] += (combined[f_start_hb - 2:f_end - 2, t_start - 2:t_end - 2]
                                 + combined[f_start_hb - 2:f_end - 2, t_start - 1:t_end - 1])

        # A pixel is selected when at least one neighbourhood condition is satisfied
        # (negation of "all conditions fail → skip")
        not_selected = (
            ((ct + cb) * e_val < eh)
            & ((ct + ht) * e_val < eh)
            & ((cb + hb) * e_val < eh)
            & (e_val < em)
        )
        selected[f_start:f_end, t_start:t_end] = above_thresh & ~not_selected

    freq_idx, time_idx = np.where(selected)
    values = combined_raw[freq_idx, time_idx]

    # Precompute per-(pixel, detector) numeric values with Numba (avoids inner Python loop)
    arrays_stack   = np.stack(arrays, axis=0)                    # (n_ifo, n_freq, n_time)
    shift_bins_arr = np.asarray(shift_bins, dtype=np.int64)
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

    pixels = []
    coord_to_index = {}
    for idx, (f_idx_i, t_idx_i, energy) in enumerate(zip(freq_idx, time_idx, values)):
        pixel_time_index = int(t_idx_i * n_freq + f_idx_i)
        pixel_data = [
            PixelData(
                noise_rms=1.0,
                wave=0.0,
                w_90=0.0,
                asnr=float(np.sqrt(pix_det_energy[idx, d])),
                a_90=0.0,
                rank=0.0,
                index=int(pix_det_index[idx, d]),
            )
            for d in range(n_ifo)
        ]

        pixels.append(
            Pixel(
                time=pixel_time_index,
                frequency=int(f_idx_i),
                layers=int(n_freq),
                rate=float(1.0 / dt),
                likelihood=float(energy),
                null=0.0,
                theta=0.0,
                phi=0.0,
                ellipticity=0.0,
                polarisation=0.0,
                core=1,
                data=pixel_data,
                td_amp=[],
                neighbors=[],
            )
        )
        coord_to_index[(int(f_idx_i), int(t_idx_i))] = idx

    tf0 = tf_maps[0]
    return {
        "mask": selected,
        "time": time_idx,
        "frequency": freq_idx,
        "energy": values,
        "pixels": pixels,
        "coord_to_index": coord_to_index,
        "rate": float(1.0 / dt),
        "layers": n_freq,
        "start": float(getattr(tf0, "start", 0.0)),
        "stop": float(getattr(tf0, "stop", 0.0)),
        "f_low": float(getattr(tf0, "f_low", 0.0) or 0.0),
        "f_high": float(getattr(tf0, "f_high", (n_freq - 1) * float(getattr(tf0, "df", 0.0))) or 0.0),
    }


def _cluster_pixels_python(pixel_candidates: dict, kt: int = 1, kf: int = 1) -> FragmentCluster:
    """
    Cluster selected pixels with connected-components labeling.

    :param pixel_candidates: payload from `_get_network_pixels_python`
    :type pixel_candidates: dict
    :param kt: time-connectivity radius (bins)
    :type kt: int
    :param kf: frequency-connectivity radius (bins)
    :type kf: int
    :return: clustered fragment payload
    :rtype: FragmentCluster
    """
    mask = np.asarray(pixel_candidates["mask"], dtype=bool)

    kt = int(max(1, kt))
    kf = int(max(1, kf))

    pixels = pixel_candidates.get("pixels", [])
    freq_arr = pixel_candidates.get("frequency", np.array([], dtype=np.int64))
    time_arr = pixel_candidates.get("time", np.array([], dtype=np.int64))

    def _compute_subnet_subrho(cluster_pixels):
        if not cluster_pixels:
            return 0.0, 0.0
        n_ifo_local = len(cluster_pixels[0].data) if cluster_pixels[0].data else 0
        if n_ifo_local <= 1:
            return 0.0, 0.0
        n_sub = 2.0 * _igamma_inv_upper(float(n_ifo_local - 1), 0.314)
        asnr_arr      = np.array([[d.asnr      for d in p.data] for p in cluster_pixels],
                                  dtype=np.float64)
        noise_rms_arr = np.array([[d.noise_rms for d in p.data] for p in cluster_pixels],
                                  dtype=np.float64)
        return _subnet_subrho_numba(asnr_arr, noise_rms_arr, n_sub)

    if not pixels:
        clusters = []
    else:
        # --- Connected-components labeling with rectangular (kf, kt) connectivity ---
        # scipy.ndimage.label requires structure shape (3, 3) for 2D inputs in
        # newer scipy versions, so we build a sparse adjacency graph instead.
        # Two pixels are directly connected if |Δfreq| ≤ kf AND |Δtime| ≤ kt.
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components as _cc

        f_idx_arr = freq_arr[:len(pixels)].astype(np.int64)
        t_idx_arr = time_arr[:len(pixels)].astype(np.int64)
        n_pix = len(f_idx_arr)

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

        # Group pixel objects by their 2D label (O(n) instead of O(n²))
        grouped: dict = {}
        for pix_idx, px in enumerate(pixels):
            f_idx = int(freq_arr[pix_idx]) if pix_idx < len(freq_arr) else int(px.frequency)
            t_idx = int(time_arr[pix_idx]) if pix_idx < len(time_arr) else (
                (int(px.time) - int(px.frequency)) // max(1, int(px.layers))
            )
            lbl = int(labeled[f_idx, t_idx])
            if lbl > 0:
                grouped.setdefault(lbl, []).append(px)

        clusters = []
        for group_idx, cluster_pixels in enumerate(grouped.values()):
            if not cluster_pixels:
                continue

            energy = float(np.sum([p.likelihood for p in cluster_pixels]))
            subnet, subrho = _compute_subnet_subrho(cluster_pixels)

            cluster_meta = ClusterMeta(
                energy=energy,
                like_net=energy,
                sub_net=subnet,
                net_rho=subrho,
                c_time=float(np.mean([p.time_in_seconds for p in cluster_pixels])),
                c_freq=float(np.mean([p.frequency_in_hz for p in cluster_pixels])),
            )

            clusters.append(Cluster(pixels=cluster_pixels, cluster_meta=cluster_meta))

    n_pix_final = int(sum(len(c.pixels) for c in clusters))
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


