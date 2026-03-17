import time
import logging
import numpy as np
from scipy.special import gammainccinv
from wdm_wavelet.wdm import WDM as WDMWavelet
from pycwb.types.detector import get_max_delay as detector_get_max_delay
from pycwb.types.time_series import TimeSeries
from pycwb.types.time_frequency_map import TimeFrequencyMap
from pycwb.types.network_cluster import FragmentCluster, Cluster, ClusterMeta
from pycwb.types.network_pixel import Pixel, PixelData
from pycwb.modules.cwb_coherence.tf_batch_generation import batch_t2w_detectors
from pycwb.modules.cwb_coherence.time_delay_max_energy import time_delay_max_energy, time_delay_max_energy_numba

logger = logging.getLogger(__name__)


def coherence(config, strains, return_rejected: bool = False, job_seg=None):
    """
    Select the significant pixels

    Loop over resolution levels (nRES)

    * Loop over detectors (cwb::nIFO)

      * Compute the maximum energy of TF pixels (WSeries<double>::maxEnergy)
      * Set pixel energy selection threshold (network::THRESHOLD)
      * Loop over time lags (network::nLag)

      * Select the significant pixels (network::getNetworkPixels)
      * Single resolution clustering (network::cluster)

    Parameters
    ----------
    config : pycwb.config.Config
        Configuration object
    strains : list
        List of whitened strain time series (pycwb TimeSeries, gwpy, or pycbc)
    nRMS_list : list of pycwb.types.time_frequency_series.TimeFrequencySeries
        List of noise RMS
    net : pycwb.types.network.Network, optional
        Network object, by default None

    Returns
    -------
    fragment_clusters: list[list[pycwb.types.network_cluster.FragmentCluster]]
        List of fragment clusters
    """
    # calculate upsample factor
    timer_start = time.perf_counter()
    logger.info("Start coherence" + " in parallel" if config.nproc > 1 else "")

    # upper sample factor
    up_n = int(config.rateANA / 1024)
    if up_n < 1:
        up_n = 1

  
    normalized_strains = [TimeSeries.from_input(strain) for strain in strains]

    fragment_clusters_multi_res = [coherence_single_res(i, config, normalized_strains, up_n, return_rejected=return_rejected, job_seg=job_seg) for i in
                                    range(config.nRES)]

    logger.info("----------------------------------------")
    logger.info("Coherence time totally: %f s", time.perf_counter() - timer_start)
    logger.info("----------------------------------------")

    return fragment_clusters_multi_res


def coherence_single_res(i, config, strains, up_n=None, return_rejected: bool = False, job_seg=None):
    """
    Calculate the coherence for a single resolution

    :param i: index of resolution
    :type i: int
    :param config: configuration object
    :type config: Config
    :param net: network
    :type net: ROOT.network
    :param strains: list of whitened strain time series
    :type strains: list[TimeSeries]
    :param wdm: wdm used for current resolution
    :type wdm: WDM
    :param up_n: upsample factor
    :type up_n: int
    :param return_rejected: if True, keep rejected clusters in output; if False, remove them
    :type return_rejected: bool
    :return: (sparse_table, fragment_clusters)
    :rtype: (ROOT.SSeries, list[ROOT.netcluster])
    """
    # timer
    timer_start = time.perf_counter()

    # print level infos
    level = config.l_high - i
    layers = 2 ** level if level > 0 else 0
    rate = config.rateANA // 2 ** level

    # use python-native wdm_wavelet for TF map generation and max-energy calculation
    wdm_layers = max(1, layers)
    wdm_wavelet = WDMWavelet(
        M=wdm_layers,
        K=wdm_layers,
        beta_order=config.WDM_beta_order,
        precision=config.WDM_precision,
    )

    # Batch t2w over all detectors in one JAX vmap call instead of a serial loop.
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
        level,
        rate,
        layers,
        config.rateANA / 2. / (2 ** level),
        1000. / rate,
    )

    fragment_clusters = []
    ###############################
    # cWB2G coherence calculation #
    ###############################

    # produce TF maps with max over the sky energy
    alp = 0.0

    max_delay = config.max_delay
    pattern = config.pattern
    n_lag = job_seg.n_lag

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
    alp = alp / config.nIFO

    # set threshold
    # threshold is calculated based on the data layers and rate of the default ifo data
    Eo = compute_threshold(
        config.bpp,
        alp if pattern != 0 else None,
        tf_maps=tf_maps,
        edge=config.segEdge,
    )
    logger.info("thresholds in units of noise variance: Eo=%g Emax=%g", Eo, Eo * 2)
    logger.info("lag_plan: n_lag=%d", n_lag)

    # set veto array
    # TODO: the veto is applied to veto the non-injected periods. Will implement later
    # TL, veto_mask = apply_veto(
    #     config.iwindow,
    #     tf_maps[0],
    #     edge=config.segEdge,
    #     segment_list=segment_list,
    #     injection_times=injection_times,
    #     return_mask=True,
    # )
    # logger_info += "live time in zero lag: %g \n" % TL

    # if TL <= 0.:
    #     raise ValueError("live time is zero")

    logger.info("lag | clusters | pixels | select_t(s) | cluster_t(s)")

    # loop over time lags
    for j in range(n_lag):
        # select pixels above Eo
        t_sel = time.perf_counter()
        candidates = select_network_pixels(
            lag_index=j,
            energy_threshold=Eo,
            tf_maps=tf_maps,
            lag_shifts=job_seg.lag_shifts[j],
            veto=None,
            edge=config.segEdge,
        )
        t_sel_elapsed = time.perf_counter() - t_sel
        n_candidates = len(candidates["pixels"]) if isinstance(candidates, dict) and "pixels" in candidates else -1

        # get pixel list
        t_cl = time.perf_counter()
        if pattern != 0:
            c = cluster_pixels(min_size=2, max_size=3, pixel_candidates=candidates)
            # remove pixels below subrho
            c.select("subrho", config.select_subrho)
            # remove pixels below subnet
            c.select("subnet", config.select_subnet)
        else:
            c = cluster_pixels(min_size=1, max_size=1, pixel_candidates=candidates)
        t_cl_elapsed = time.perf_counter() - t_cl

        if not return_rejected:
            c.remove_rejected()

        fragment_cluster = c
        fragment_clusters.append(fragment_cluster)

        logger.info(
            "%3d |%9d |%7d | cand=%d sel=%.4fs clust=%.4fs",
            j,
            fragment_cluster.event_count(),
            fragment_cluster.pixel_count(),
            n_candidates,
            t_sel_elapsed,
            t_cl_elapsed,
        )

    logger.info("Coherence time for single level: %f s", time.perf_counter() - timer_start)
    return fragment_clusters


# ---------------------------------------------------------------------------
# Streaming-friendly API: setup once, iterate over lags
# ---------------------------------------------------------------------------

def setup_coherence(config, strains, job_seg=None):
    """
    Compute all lag-independent coherence data (TF maps after max_energy,
    threshold, lag plan) for every resolution level.

    Call this once per job segment, then pass the returned list to
    :func:`coherence_single_lag` for each lag.

    Parameters
    ----------
    config : Config
    strains : list
        Whitened strain time series (pycwb TimeSeries, gwpy, or pycbc).
    job_seg : WaveSegment
        Job segment (provides lag count via ``job_seg.n_lag``).

    Returns
    -------
    list[dict]
        One setup dict per resolution, keyed by ``tf_maps``, ``Eo``,
        ``job_seg``, ``pattern``, ``level``, ``layers``, ``rate``,
        ``select_subrho``, ``select_subnet``, ``segEdge``.
    """
    timer_start = time.perf_counter()

    up_n = int(config.rateANA / 1024)
    if up_n < 1:
        up_n = 1

    normalized_strains = [TimeSeries.from_input(strain) for strain in strains]

    setups = [
        _setup_coherence_single_res(i, config, normalized_strains, up_n,
                                    job_seg=job_seg)
        for i in range(config.nRES)
    ]

    logger.info("Coherence setup time: %.2f s", time.perf_counter() - timer_start)
    return setups


def _setup_coherence_single_res(i, config, strains, up_n, job_seg=None):
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

    wdm_layers = max(1, layers)
    wdm_wavelet = WDMWavelet(
        M=wdm_layers,
        K=wdm_layers,
        beta_order=config.WDM_beta_order,
        precision=config.WDM_precision,
    )

    # Build TF maps
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

    # Apply max_energy over the sky (modifies tf-map data in place)
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
    alp = alp / config.nIFO

    # Compute energy threshold (lag-independent)
    Eo = compute_threshold(
        config.bpp,
        alp if pattern != 0 else None,
        tf_maps=tf_maps,
        edge=config.segEdge,
    )

    # Build lag plan
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


def coherence_single_lag(coherence_setups, lag_idx, return_rejected=False):
    """
    Compute coherence for one lag index, using pre-built per-resolution setups
    from :func:`setup_coherence`.

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

    Returns
    -------
    list[FragmentCluster]
        One FragmentCluster per resolution for this lag.
    """
    fragment_clusters = []
    for setup in coherence_setups:
        tf_maps = setup["tf_maps"]
        Eo = setup["Eo"]
        job_seg = setup["job_seg"]
        pattern = setup["pattern"]

        if lag_idx >= job_seg.n_lag:
            raise IndexError(
                f"lag_idx={lag_idx} is out of range n_lag={job_seg.n_lag}"
            )

        t_sel = time.perf_counter()
        candidates = select_network_pixels(
            lag_index=lag_idx,
            energy_threshold=Eo,
            tf_maps=tf_maps,
            lag_shifts=job_seg.lag_shifts[lag_idx],
            veto=None,
            edge=setup["segEdge"],
        )
        t_sel_elapsed = time.perf_counter() - t_sel
        n_candidates = (
            len(candidates["pixels"])
            if isinstance(candidates, dict) and "pixels" in candidates
            else -1
        )

        t_cl = time.perf_counter()
        if pattern != 0:
            c = cluster_pixels(min_size=2, max_size=3, pixel_candidates=candidates)
            c.select("subrho", setup["select_subrho"])
            c.select("subnet", setup["select_subnet"])
        else:
            c = cluster_pixels(min_size=1, max_size=1, pixel_candidates=candidates)
        t_cl_elapsed = time.perf_counter() - t_cl

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


def max_energy(tf_map: TimeFrequencyMap, max_delay, up_n, pattern,
               f_low=None, f_high=None, hist=None):
    """
    Decoupled max-energy computation for a detector TF map.

    Calls :func:`time_delay_max_energy` from the module-level pure-function
    implementation and returns a new TF map together with the Gamma-to-Gauss
    scaling parameter.

    :param tf_map: detector TF map object
    :type tf_map: TimeFrequencyMap
    :param max_delay: maximum delay for the time series
    :type max_delay: float
    :param up_n: upsample factor
    :type up_n: int
    :param pattern: wave packet pattern
    :type pattern: int
    :param f_low: low cut frequency
    :type f_low: float | None
    :param f_high: high cut frequency
    :type f_high: float | None
    :param hist: optional histogram container
    :type hist: list | None
    :return: ``(new_tf_map, alp)`` — updated TF map and Gamma-to-Gauss scaling
    :rtype: tuple[TimeFrequencyMap, float]
    """
    if hasattr(tf_map, "bandpass"):
        tf_map.bandpass(f_low=f_low, f_high=f_high)

    new_tf_map, result = time_delay_max_energy(tf_map, max_delay, downsample=up_n, pattern=pattern, hist=hist)
    return new_tf_map, result


def compute_threshold(bpp, alp=None, tf_maps=None, edge=None):
    """
    Decoupled threshold calculation.

    If `tf_maps` are provided, uses a Python-native implementation inspired by
    cWB `network::THRESHOLD` logic.

    :param bpp: black pixel probability
    :type bpp: float
    :param alp: optional packet-shape reference value
    :type alp: float | None
    :param tf_maps: optional list of python TF maps (for python-native threshold)
    :type tf_maps: list | None
    :param edge: optional edge in seconds for data trimming
    :type edge: float | None
    :return: calculated threshold
    :rtype: float
    """
    if tf_maps is None or len(tf_maps) == 0 or not hasattr(tf_maps[0], "data"):
        raise ValueError("compute_threshold requires python TF maps")
    return _threshold_python(tf_maps, bpp=bpp, shape=alp, edge=edge)


def select_network_pixels(lag_index, energy_threshold, tf_maps=None, lag_shifts=None, veto=None, edge=0.0):
    """
    Decoupled significant-pixel selection.

    Returns Python pixel candidates compatible with
    `_cluster_pixels_python`.

    :param lag_index: lag index
    :type lag_index: int
    :param energy_threshold: pixel threshold
    :type energy_threshold: float
    :param tf_maps: optional python TF maps
    :type tf_maps: list | None
    :param lag_shifts: per-detector lag shifts for this lag (seconds)
    :type lag_shifts: np.ndarray | list[float] | None
    :param veto: optional veto array in time bins (1 keep, 0 reject)
    :type veto: np.ndarray | None
    :param edge: optional edge in seconds
    :type edge: float
    :return: candidate payload with selected mask, coordinates, energies and pixels
    :rtype: dict
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


def cluster_pixels(min_size, max_size, pixel_candidates=None):
    """
    Decoupled clustering function.

    If `pixel_candidates` is provided (output of `_get_network_pixels_python`),
    performs python-native connected-component clustering and returns labels.
    This function is intentionally python-only for the cwb_coherence path.

    :param min_size: minimum size of clusters
    :type min_size: int
    :param max_size: maximum size of clusters
    :type max_size: int
    :param pixel_candidates: optional python pixel candidate payload
    :type pixel_candidates: dict | None
    :return: clustered pixels
    :rtype: object
    """
    if pixel_candidates is None:
        raise ValueError("cluster_pixels requires python pixel_candidates")
    return _cluster_pixels_python(pixel_candidates, kt=min_size, kf=max_size)
    

def apply_veto(iwindow, tf_map, segment_list=None, injection_times=None, edge=None, return_mask=False):
    """
    Decoupled veto application.

    Applies Python-native veto construction based on cWB `network::setVeto` logic.

    :param iwindow: window size for veto
    :type iwindow: int
    :param tf_map: reference TF map for timeline definition
    :type tf_map: object
    :param segment_list: list of (start, stop) GPS segments
    :type segment_list: list[tuple[float, float]] | None
    :param injection_times: list of injection times (GPS)
    :type injection_times: list[float] | None
    :param edge: optional edge in seconds for live-time integration
    :type edge: float | None
    :param return_mask: if True, returns `(live_time, veto_mask)`
    :type return_mask: bool
    :return: live time in zero lag
    :rtype: float | tuple[float, np.ndarray]
    """
    live, veto_mask = _set_veto_python(
        tf_map=tf_map,
        tw=float(iwindow),
        segment_list=segment_list,
        injection_times=injection_times,
        edge=edge,
    )
    return (live, veto_mask) if return_mask else live


def get_max_delay(net):
    """
    Decoupled accessor for network max-delay.

    Resolution order is intentionally Python-first:
    1) detector-level `tau` values (pure Python utility),
    2) ROOT network backend (`net.net.getDelay("MAX")`),
    3) legacy wrapper method (`net.get_max_delay()`).

    :param net: network object
    :type net: Network
    :return: maximum delay in seconds
    :rtype: float
    """
    # Preferred path: derive delay directly from detector tau maps.
    if hasattr(net, "n_ifo") and hasattr(net, "get_ifo"):
        detectors = [net.get_ifo(i) for i in range(net.n_ifo)]
        delay = detector_get_max_delay(detectors)
        if delay > 0:
            return float(delay)

    # Compatibility fallback for ROOT-backed network instances.
    if hasattr(net, "net") and hasattr(net.net, "getDelay"):
        return float(net.net.getDelay("MAX"))

    # Final fallback for legacy wrappers.
    if hasattr(net, "get_max_delay"):
        return float(net.get_max_delay())

    return 0.0


def _igamma_inv_upper(shape, p):
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


def _get_tf_energy_array(tf_map, edge=None):
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


def _threshold_python(tf_maps, bpp, shape=None, edge=None):
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


def _set_veto_python(tf_map, tw, segment_list=None, injection_times=None, edge=None):
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


def _get_network_pixels_python(tf_maps, lag_index, energy_threshold, lag_shifts=None, veto=None, edge=0.0):
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

    pixels = []
    coord_to_index = {}
    for idx, (f_idx, t_idx, energy) in enumerate(zip(freq_idx, time_idx, values)):
        pixel_time_index = int(t_idx * n_freq + f_idx)
        pixel_data = []
        for det_idx in range(n_ifo):
            if nn_valid > 0 and valid_start <= int(t_idx) < valid_stop:
                u = int(t_idx) - valid_start
                det_t = int(valid_start + ((u + shift_bins[det_idx]) % nn_valid))
            else:
                det_t = int(t_idx)
            det_energy = float(max(arrays[det_idx][int(f_idx), det_t], 0.0))
            det_index = int(det_t * n_freq + f_idx)
            pixel_data.append(
                PixelData(
                    noise_rms=1.0,
                    wave=0.0,
                    w_90=0.0,
                    asnr=float(np.sqrt(det_energy)),
                    a_90=0.0,
                    rank=0.0,
                    index=det_index,
                )
            )

        pixels.append(
            Pixel(
                time=pixel_time_index,
                frequency=int(f_idx),
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
        coord_to_index[(int(f_idx), int(t_idx))] = idx

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


def _cluster_pixels_python(pixel_candidates, kt=1, kf=1):
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

        rho = 0.0
        e_sub = 0.0
        e_max_sum = 0.0
        subnet_acc = 0.0

        for p in cluster_pixels:
            amp_sum = 0.0
            e_max = 0.0
            e_tot = 0.0
            nsd = 0.0
            msd = 0.0

            for det in p.data:
                amp = float(det.asnr)
                x = amp * amp
                v = float(det.noise_rms) * float(det.noise_rms)

                amp_sum += abs(amp)
                if x > e_max:
                    e_max = x
                    msd = v
                e_tot += x
                if v > 0:
                    nsd += 1.0 / v

            a_fac = (e_tot / (amp_sum * amp_sum)) if amp_sum > 0 else 1.0
            rho += (1.0 - a_fac) * (e_tot - n_sub * 2.0)

            y_val = e_tot - e_max
            x_corr = y_val * (1.0 + y_val / (e_max + 1.0e-5))

            if msd > 0:
                nsd -= 1.0 / msd
            v_corr = (2.0 * e_max - e_tot) * msd * nsd / 10.0

            e_sub += (e_tot - e_max)
            e_max_sum += e_max

            a_cut = x_corr / (x_corr + n_sub) if (x_corr + n_sub) != 0 else 0.0
            denom = x_corr + (v_corr if v_corr > 0 else 1.0e-5)
            subnet_acc += (e_tot * x_corr / denom) * (a_cut if a_cut > 0.5 else 0.0)

        subnet = subnet_acc / (e_max_sum + e_sub + 0.01)
        subrho = float(np.sqrt(rho)) if rho >= 0 else float("nan")
        return float(subnet), subrho

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

        # Build sparse adjacency (upper triangle only; undirected)
        if n_pix > 0:
            df = np.abs(f_idx_arr[:, None] - f_idx_arr[None, :])
            dt = np.abs(t_idx_arr[:, None] - t_idx_arr[None, :])
            adj = csr_matrix((df <= kf) & (dt <= kt))
            _, raw_labels = _cc(adj, directed=False, connection='weak')
            # raw_labels is 0-indexed; shift to 1-indexed to match ndimage convention
            raw_labels = raw_labels + 1
        else:
            raw_labels = np.array([], dtype=np.int64)

        # Build a labeled 2D array (same shape as mask) for downstream code
        labeled = np.zeros(mask.shape, dtype=np.int64)
        for pix_idx in range(n_pix):
            labeled[int(f_idx_arr[pix_idx]), int(t_idx_arr[pix_idx])] = int(raw_labels[pix_idx])

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


def _extract_net_values(net, n_ifo, fallback_max_delay):
    """
    Extract network runtime values needed by Python coherence flow.

    :param net: network wrapper
    :type net: Network
    :param n_ifo: number of detectors
    :type n_ifo: int
    :param fallback_max_delay: config fallback max delay
    :type fallback_max_delay: float
    :return: dictionary with pattern, lags, shifts, veto-related metadata
    :rtype: dict
    """
    pattern = int(getattr(net, "pattern", 0))
    n_lag = int(getattr(net, "nLag", 1))

    max_delay = float(fallback_max_delay)
    if max_delay <= 0:
        max_delay = get_max_delay(net)

    lag_shifts = np.zeros((n_lag, n_ifo), dtype=float)
    for det_idx in range(n_ifo):
        ifo = net.get_ifo(det_idx)
        shifts = np.asarray(ifo.lagShift.data, dtype=float)
        if shifts.size < n_lag:
            lag_shifts[:shifts.size, det_idx] = shifts
        else:
            lag_shifts[:, det_idx] = shifts[:n_lag]

    segment_list = None
    if hasattr(net, "net") and hasattr(net.net, "segList"):
        try:
            segment_list = [(float(seg.start), float(seg.stop)) for seg in net.net.segList]
        except Exception:
            segment_list = None

    injection_times = None
    if hasattr(net, "net") and hasattr(net.net, "mdcTime"):
        try:
            injection_times = [float(x) for x in net.net.mdcTime if float(x) != 0.0]
        except Exception:
            injection_times = None

    return {
        "pattern": pattern,
        "n_lag": n_lag,
        "max_delay": max_delay,
        "lag_shifts": lag_shifts,
        "segment_list": segment_list,
        "injection_times": injection_times,
    }