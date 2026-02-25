import time
import logging
import numpy as np
from scipy.special import gammainccinv
from pycwb.types.network import Network
from wdm_wavelet.wdm import WDM as WDMWavelet
from pycwb.types.detector import get_max_delay as detector_get_max_delay
from pycwb.types.time_series import TimeSeries
from pycwb.types.time_frequency_series import TimeFrequencyMap
from pycwb.types.network_cluster import FragmentCluster, Cluster, ClusterMeta
from pycwb.types.network_pixel import Pixel, PixelData
from pycwb.modules.cwb_coherence.lag_plan import build_lag_plan_from_config

logger = logging.getLogger(__name__)


def coherence(config, strains, return_rejected: bool = False):
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

    fragment_clusters_multi_res = [coherence_single_res(i, config, normalized_strains, up_n, return_rejected=return_rejected) for i in
                                    range(config.nRES)]

    logger.info("----------------------------------------")
    logger.info("Coherence time totally: %f s", time.perf_counter() - timer_start)
    logger.info("----------------------------------------")

    return fragment_clusters_multi_res


def coherence_single_res(i, config, strains, up_n=None, return_rejected: bool = False):
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

    # use string instead of directly logging to avoid messy output in parallel
    logger_info = "level : %d\t rate(hz) : %d\t layers : %d\t df(hz) : %f\t dt(ms) : %f \n" % (
        level, rate, layers, config.rateANA / 2. / (2 ** level), 1000. / rate)

    fragment_clusters = []
    ###############################
    # cWB2G coherence calculation #
    ###############################

    # produce TF maps with max over the sky energy
    alp = 0.0

    max_delay = config.max_delay
    pattern = config.pattern
    lag_plan = build_lag_plan_from_config(config, tf_maps)
    n_lag = lag_plan.n_lag


    for n, tf_map in enumerate(tf_maps):
        alp += max_energy(
            tf_map=tf_map,
            max_delay=max_delay,
            up_n=up_n,
            pattern=pattern,
            f_low=config.fLow,
            f_high=config.fHigh,
        )

    logger_info += "max energy in units of noise variance: %g \n" % alp

    alp = alp / config.nIFO

    # set threshold
    # threshold is calculated based on the data layers and rate of the default ifo data
    Eo = compute_threshold(
        config.bpp,
        alp if pattern != 0 else None,
        tf_maps=tf_maps,
        edge=config.segEdge,
    )

    logger_info += "thresholds in units of noise variance: Eo=%g Emax=%g \n" % (Eo, Eo * 2)

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

    logger_info += "lag | clusters | pixels \n"

    # loop over time lags
    for j in range(n_lag):
        # select pixels above Eo
        candidates = select_network_pixels(
            lag_index=j,
            energy_threshold=Eo,
            tf_maps=tf_maps,
            lag_shifts=lag_plan.lag_shifts[j],
            veto=None,
            edge=config.segEdge,
        )
        # get pixel list
        if pattern != 0:
            c = cluster_pixels(min_size=2, max_size=3, pixel_candidates=candidates)
            # remove pixels below subrho
            c.select("subrho", config.select_subrho)
            # remove pixels below subnet
            c.select("subnet", config.select_subnet)
        else:
            c = cluster_pixels(min_size=1, max_size=1, pixel_candidates=candidates)

        if not return_rejected:
            c.remove_rejected()

        fragment_cluster = c
        fragment_clusters.append(fragment_cluster)

        logger_info += "%3d |%9d |%7d \n" % (j, fragment_cluster.event_count(), fragment_cluster.pixel_count())

    logger_info += "Coherence time for single level: %f s" % (time.perf_counter() - timer_start)

    logger.info(logger_info)
    return fragment_clusters


def max_energy(tf_map: TimeFrequencyMap, max_delay, up_n, pattern,
               f_low=None, f_high=None, hist=None):
    """
    Decoupled max-energy computation for a detector TF map.

    Python-native path is used when `tf_map` exposes `time_delay_max_energy`.
    This function is intentionally python-only for the cwb_coherence path.

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
    :return: maximum energy in units of noise variance
    :rtype: float
    """
    if hasattr(tf_map, "bandpass"):
        tf_map.bandpass(f_low=f_low, f_high=f_high)

    if hasattr(tf_map, "time_delay_max_energy"):
        return float(tf_map.time_delay_max_energy(max_delay, downsample=up_n, pattern=pattern, hist=hist))

    raise ValueError("max_energy requires python TimeFrequencyMap with time_delay_max_energy")


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
    """Upper-tail inverse incomplete gamma helper used by threshold estimators."""
    p = float(np.clip(p, 1.0e-12, 1.0 - 1.0e-12))
    s = float(max(shape, 1.0e-12))
    return float(gammainccinv(s, p))


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
    energies = [_get_tf_energy_array(tfm, edge=edge) for tfm in tf_maps]
    combined = np.sum(energies, axis=0)
    work = combined.ravel()

    n_ifo = len(tf_maps)
    if work.size == 0:
        return 0.0

    work = np.clip(work, 0.0, n_ifo * 100.0)
    positive = work[work > 1.0e-3]
    if positive.size == 0:
        return 0.0

    if shape is not None:
        avr = float(np.mean(positive))
        bbb = float(np.mean(np.log(positive)))
        alp = np.log(avr) - bbb
        alp = (3 - alp + np.sqrt((alp - 3) * (alp - 3) + 24 * alp)) / (12 * alp)
        bpp_corr = float(bpp) * alp / float(shape)
        return avr * _igamma_inv_upper(alp, bpp_corr) / alp / 2.0

    med = float(np.quantile(positive, 0.8))
    m = max(1.0, med / max(_igamma_inv_upper(n_ifo, 0.2), 1.0e-12))
    p_eff = float(np.clip(bpp, 1.0e-8, 0.2))
    val = float(np.quantile(positive, 1.0 - p_eff))
    return (0.3 * (_igamma_inv_upper(n_ifo * m, p_eff) + val)) + n_ifo * np.log(m)


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

    for t in range(t_start, t_end):
        for f_idx in range(f_start, f_end):
            e_val = combined[f_idx, t]
            if e_val < eo:
                continue

            ct = combined[f_idx + 1, t] + combined[f_idx, t + 1] + combined[f_idx + 1, t + 1]
            cb = combined[f_idx - 1, t] + combined[f_idx, t - 1] + combined[f_idx - 1, t - 1]

            ht = combined[f_idx + 1, t + 2]
            if f_idx < ii:
                ht += combined[f_idx + 2, t + 2] + combined[f_idx + 2, t + 1]

            hb = combined[f_idx - 1, t - 2]
            if f_idx > 1:
                hb += combined[f_idx - 2, t - 2] + combined[f_idx - 2, t - 1]

            if (
                (ct + cb) * e_val < eh
                and (ct + ht) * e_val < eh
                and (cb + hb) * e_val < eh
                and e_val < em
            ):
                continue

            selected[f_idx, t] = True

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
    coord_to_index = pixel_candidates.get("coord_to_index", {})

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
        n_points = len(pixels)
        parent = np.arange(n_points, dtype=np.int32)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        # cWB-like sort and neighbor linking
        order = list(range(n_points))
        order.sort(key=lambda idx: float(pixels[idx].time) / pixels[idx].rate / pixels[idx].layers)

        m_layers = int(pixels[0].layers)
        r_rate = float(pixels[0].rate)
        cluster_rate = float(pixel_candidates.get("rate", r_rate))
        is_wavelet = (int(cluster_rate / r_rate + 0.1) == m_layers)

        for i_ord in range(n_points):
            p_idx = order[i_ord]
            p = pixels[p_idx]
            for j_ord in range(i_ord + 1, n_points):
                q_idx = order[j_ord]
                q = pixels[q_idx]

                if is_wavelet:
                    dt_index = int(q.time) - int(p.time)
                else:
                    dt_index = int(q.time / m_layers) - int(p.time / m_layers)

                if dt_index < 0:
                    continue

                if is_wavelet:
                    if dt_index / m_layers > kt:
                        break
                else:
                    if dt_index > kt:
                        break

                if abs(int(q.frequency) - int(p.frequency)) <= kf:
                    union(p_idx, q_idx)

        grouped = {}
        for idx in range(n_points):
            root = find(idx)
            grouped.setdefault(root, []).append(pixels[idx])

        clusters = []
        for cluster_pixels in grouped.values():
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

    return FragmentCluster(
        rate=float(pixel_candidates.get("rate", 0.0)),
        start=float(pixel_candidates.get("start", 0.0)),
        stop=float(pixel_candidates.get("stop", 0.0)),
        bpp=1.0,
        shift=0.0,
        f_low=float(pixel_candidates.get("f_low", 0.0)),
        f_high=float(pixel_candidates.get("f_high", 0.0)),
        n_pix=int(sum(len(c.pixels) for c in clusters)),
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