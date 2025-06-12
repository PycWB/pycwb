import logging
import numpy as np
from pycbc.types import TimeSeries
from pycwb.modules.multi_resolution_wdm import create_wdm_set
from multiprocessing import Pool
from numba import njit

logger = logging.getLogger(__name__)


def get_MRA_wave(cluster, wdmList, rate, ifo, a_type, mode, nproc, whiten=False) -> TimeSeries:
    """
    get MRA waveforms of type atype in time domain given lag nomber and cluster ID

    Parameters
    ----------
    cluster : pycwb.types.network_cluster.Cluster
        cluster object
    wdmList : list of pycwb.types.wdm.WDM
        list of WDM objects
    rate : float
        sampling rate
    ifo : int
        IFO id
    a_type : str
        type of waveforms, the value can be 'signal' or 'strain'
    mode : int
        -1/0/1 - return 90/mra/0 phase

    Returns
    -------
    waveform : pycbc.types.timeseries.TimeSeries
        reconstructed waveform
    """
    if not cluster.pixels:
        return None

    max_f_len = max([wdm.m_H / rate for wdm in wdmList])

    # find event time interval, fill in amplitudes
    tmin = 1e20
    tmax = 0
    for pix in cluster.pixels:
        T = int(pix.time / pix.layers)  # get time index
        T = T / pix.rate  # time in seconds from the start
        tmin = min(tmin, T)
        tmax = max(tmax, T)

    tmin = int(tmin - max_f_len) - 1  # start event time in sec
    tmax = int(tmax + max_f_len) + 1  # end event time in sec

    # create a time series with np.zeros(int(rate*(tmax-tmin)+0.1))
    z = TimeSeries(np.zeros(int(rate * (tmax - tmin) + 0.1)), delta_t=1 / rate, epoch=tmin)

    io = int(tmin / z.delta_t + 0.01)  # index offset of z-array

    s00 = 0
    s90 = 0

    z_len = len(z.data)

    results = [_process_pixels(pix, ifo, a_type, mode, wdmList, io, z_len, whiten) for pix in cluster.pixels]
    # if min(nproc, len(cluster.pixels)) == 1:
    #     results = [_process_pixels(pix, ifo, a_type, mode, wdmList, io, z_len) for pix in cluster.pixels]
    # else:
    #     with Pool(processes=min(nproc, len(cluster.pixels))) as pool:
    #         results = pool.starmap(_process_pixels,
    #                                [(pix, ifo, a_type, mode, wdmList, io, z_len) for pix in cluster.pixels])

    for result in results:
        if result is None:
            continue
        valid_indices_values_00, val00, valid_indices_values_90, val90, s00_val, s90_val = result
        z.data[valid_indices_values_00] += val00
        z.data[valid_indices_values_90] += val90
        s00 += s00_val
        s90 += s90_val

    return z


def get_network_MRA_wave(config, cluster, rate, nIFO, rTDF, a_type, mode, tof, whiten=False, in_rate=None, start_time=None):
    """
    get MRA waveforms of type atype in time domain given lag nomber and cluster ID

    Parameters
    ----------
    config : pycwb.config.Config
        configuration object
    cluster : pycwb.types.network_cluster.Cluster
        cluster object
    rate : float
        sampling rate
    nIFO : int
        number of IFOs
    rTDF : float
        effective time-delay
    a_type : str
        type of waveforms, the value can be 'signal' or 'strain'
    mode : int
        -1/0/1 - return 90/mra/0 phase
    tof : bool
        if tof = true, apply time-of-flight corrections
    whiten : bool
        if whiten = true, reconstruct whitened waveforms
    in_rate : float, optional
        input rate of the original data, used for rescaling the waveforms.

    Returns
    -------
    waveforms : list of pycbc.types.timeseries.TimeSeries
        reconstructed waveform
    """
    wdm_list = create_wdm_set(config)

    v = cluster.sky_time_delay  # backward time delay configuration

    # time-of-flight backward correction for reconstructed waveforms
    waveforms = []
    for i in range(nIFO):
        x = get_MRA_wave(cluster, wdm_list, rate, i, a_type, mode, nproc=config.nproc, whiten=whiten)
        if len(x.data) == 0:
            logger.warning("zero length")
            return False

        # apply time delay
        if tof:
            R = rTDF  # effective time-delay rate
            t_shift = -v[i] / R
            xf = x.to_frequencyseries()
            xf.data *= np.exp(-2j * np.pi * xf.sample_frequencies * t_shift)
            x = xf.to_timeseries()

        waveforms.append(x)
        
        if in_rate is not None:
            rescale = 1. / np.sqrt(2) ** (np.log2(in_rate / x.sample_rate))
            x.data *= rescale

        if start_time is not None:
            x.start_time += start_time

    return waveforms


def _process_pixels(pix, ifo, a_type, mode, wdmList, io, z_len, whiten=False):
    if not pix.core:
        return

    rms = pix.data[ifo].noise_rms
    a00 = pix.data[ifo].asnr if a_type == 'signal' else pix.data[ifo].wave
    a90 = pix.data[ifo].a_90 if a_type == 'signal' else pix.data[ifo].w_90
    a00 *= 1 if whiten else rms
    a90 *= 1 if whiten else rms

    # find the object which pix.layers == wdm.max_layers + 1
    wdm = [w for w in wdmList if w.max_layer + 1 == pix.layers][0]
    j00, x00 = get_base_wave(wdm.max_layer, np.array(wdm.wavelet.wdmFilter), pix.time, False) # wdm.get_base_wave(pix.time, False)
    j90, x90 = get_base_wave(wdm.max_layer, np.array(wdm.wavelet.wdmFilter), pix.time, True) #wdm.get_base_wave(pix.time, True)
    j00 -= io
    j90 -= io
    if mode < 0:
        a00 = 0
    if mode > 0:
        a90 = 0

    s00 = a00 * a00
    s90 = a90 * a90

    # Calculate the valid range of indices
    indices_00 = np.arange(len(x00)) + j00
    indices_90 = np.arange(len(x90)) + j90

    valid_indices_00 = np.logical_and(indices_00 >= 0, indices_00 < z_len)
    valid_indices_90 = np.logical_and(indices_90 >= 0, indices_90 < z_len)

    # Filter the valid indices and corresponding values in x00 and x90
    valid_indices_values_00 = indices_00[valid_indices_00]
    valid_indices_values_90 = indices_90[valid_indices_90]
    valid_x00 = x00[valid_indices_00]
    valid_x90 = x90[valid_indices_90]

    # Perform the addition operation
    return valid_indices_values_00, valid_x00 * a00, valid_indices_values_90, valid_x90 * a90, s00, s90


def _get_base_wave(wdm_filter, max_layer, m, n):
    N = len(wdm_filter)
    M = max_layer

    if m == 0:
        if n % 2:
            w = np.zeros(2 * (N - 1) + 1)
        else:
            w = np.concatenate((np.flipud(wdm_filter), wdm_filter[1:]))
    elif m == M:
        if (n - M) % 2:
            w = np.zeros(2 * (N - 1) + 1)
        else:
            w_pos = np.zeros(N - 1)  # symmetric array
            s = -1 if M % 2 else 1
            w0 = s * wdm_filter[0]
            for i in range(1, N):
                s = -s
                w_pos[i - 1] = s * wdm_filter[i]
            w = np.concatenate((np.flipud(w_pos), np.array([w0]), w_pos))
    else:
        ratio = m * np.pi / M
        sign = np.sqrt(2)
        if (m + n) % 2:
            w_pos = -sign * np.sin(np.arange(1, N) * ratio) * wdm_filter[1:]
            w_neg = - np.flipud(w_pos)
            w = np.concatenate((w_neg, np.array([0]), w_pos))
        else:
            w_pos = sign * np.cos(np.arange(1, N) * ratio) * wdm_filter[1:]
            w_neg = np.flipud(w_pos)
            w = np.concatenate((w_neg, np.array([sign * wdm_filter[0]]), w_pos))

    return n * M - (N-1), w


def _get_base_wave_quad(wdm_filter, max_layer, m, n):
    N = len(wdm_filter)
    M = max_layer

    if m == 0:
        if n % 2:
            w = np.concatenate((np.flipud(wdm_filter), wdm_filter[1:]))
        else:
            w = np.zeros(2*(N-1) + 1)
    elif m == M:
        if (n-M) % 2:
            w_pos = np.zeros(N-1) # symmetric array
            s = 1
            w0 = wdm_filter[0]
            for i in range(1, N):
                s = -s
                w_pos[i-1] = s * wdm_filter[i]
            w = np.concatenate((np.flipud(w_pos), np.array([w0]), w_pos))
        else:
            w = np.zeros(2*(N-1) + 1)
    else:
        ratio = m * np.pi / M
        sign = np.sqrt(2)
        if (m+n) % 2:
            w0 = sign * wdm_filter[0]
            w_pos = sign * np.cos(np.arange(1, N) * ratio) * wdm_filter[1:]
            w = np.concatenate((np.flipud(w_pos), np.array([w0]), w_pos))
        else:
            w_pos = sign * np.sin(np.arange(1, N) * ratio) * wdm_filter[1:]
            w = np.concatenate((- np.flipud(w_pos), np.array([0]), w_pos))

    return n * M - (N-1), w


def get_base_wave(max_layer: int, wdm_filter: np.array, tf_index: int, quad: bool) -> np.array:
    M1 = max_layer + 1
    m = tf_index % M1
    n = tf_index // M1
    if quad:
        return _get_base_wave_quad(wdm_filter, max_layer, m, n)
    else:
        return _get_base_wave(wdm_filter, max_layer, m, n)
