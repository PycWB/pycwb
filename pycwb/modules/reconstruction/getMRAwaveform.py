import logging
import numpy as np
from pycbc.types import TimeSeries
from pycwb.modules.multi_resolution_wdm import create_wdm_set

logger = logging.getLogger(__name__)


def get_MRA_wave(cluster, wdmList, rate, ifo, a_type, mode):
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

    max_f_len = max([wdm.m_H/rate for wdm in wdmList])

    # find event time interval, fill in amplitudes
    tmin = 1e20
    tmax = 0
    for pix in cluster.pixels:
        T = int(pix.time/pix.layers) # get time index
        T = T/pix.rate # time in seconds from the start
        tmin = min(tmin, T)
        tmax = max(tmax, T)

    tmin = int(tmin - max_f_len) - 1 # start event time in sec
    tmax = int(tmax + max_f_len) + 1 # end event time in sec

    # create a time series with np.zeros(int(rate*(tmax-tmin)+0.1))
    z = TimeSeries(np.zeros(int(rate*(tmax-tmin)+0.1)), delta_t=1/rate, epoch=tmin)

    io = int(tmin/z.delta_t+0.01) # index offset of z-array

    s00 = 0
    s90 = 0

    for pix in cluster.pixels:
        if not pix.core:
            continue

        rms = pix.data[ifo].noise_rms
        a00 = pix.data[ifo].asnr if a_type == 'signal' else pix.data[ifo].wave
        a90 = pix.data[ifo].a_90 if a_type == 'signal' else pix.data[ifo].w_90
        a00 *= rms if a_type == 'strain' else 1
        a90 *= rms if a_type == 'strain' else 1

        # find the object which pix.layers == wdm.max_layers + 1
        wdm = [w for w in wdmList if w.max_layer + 1 == pix.layers][0]
        j00, x00 = wdm.get_base_wave(pix.time, False)
        j90, x90 = wdm.get_base_wave(pix.time, True)
        j00 -= io
        j90 -= io
        if mode < 0:
            a00 = 0
        if mode > 0:
            a90 = 0

        s00 += a00*a00
        s90 += a90*a90

        # TODO: optimize with numpy
        for i in range(len(x00)):
            if j00+i < 0 or j00+i >= len(z.data):
                continue
            z.data[j00+i] += x00[i]*a00
            z.data[j90+i] += x90[i]*a90
    return z


def get_network_MRA_wave(config, cluster, rate, nIFO, rTDF, a_type, mode, tof):
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
        x = get_MRA_wave(cluster, wdm_list, rate, i, a_type, mode)
        if len(x.data) == 0:
            logger.warning("zero length")
            return False

        # apply time delay
        if tof:
            R = rTDF  # effective time-delay rate
            t_shift = v[i] / R
            xf = x.to_frequencyseries()
            xf = xf.cyclic_time_shift(t_shift)
            x = xf.to_timeseries()

        waveforms.append(x)

    return waveforms