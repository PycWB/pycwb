import ROOT
import pyburst, os
import numpy as np
from gwpy.timeseries import TimeSeries
from pycbc.types.timeseries import TimeSeries as pycbcTimeSeries
import logging
import ctypes

from pyburst.types import TimeFrequencySeries, WDM

c_double_p = ctypes.POINTER(ctypes.c_double)

logger = logging.getLogger(__name__)

if not hasattr(ROOT, "WDM"):
    logger.info("Loading wavelet library")
    try:
        pyburst_path = os.path.dirname(pyburst.__file__)
        ROOT.gSystem.Load(f"{pyburst_path}/vendor/lib/wavelet")
    except:
        logger.error("Cannot find wavelet library in pycwb, trying to load from system")
        try:
            ROOT.gSystem.Load("wavelet")
        except:
            logger.error("Cannot find wavelet library")
            raise Exception("Cannot find wavelet library")


def declare_function():
    """
    This is to declare a c++ function to copy numpy array to wavearray (copying with python loop is too slow) and
    return wavearray data to float array
    """
    ROOT.gInterpreter.Declare("""
    void _copy_to_wavearray(double *value, wavearray<double> *wave, int size) {
        for (int i = 0; i < size; i++) {
            wave->data[i] = value[i];
        }
    };
    
    std::vector<double> _get_wavearray_data(wavearray<double> *wave) {
        std::vector<double> data;
        for (int i = 0; i < wave->size(); i++) {
            data.push_back(wave->data[i]);
        }
        
        return data;
    };
    
    std::vector<double> _get_wseries_data(WSeries<double> *wave) {
        std::vector<double> data;
        for (int i = 0; i < wave->size(); i++) {
            data.push_back(wave->data[i]);
        }
        
        return data;
    };
    """)


def convert_to_wseries(data):
    """
    Convert all known types to wseries
    :param data: input data
    :return: wseries
    """
    if isinstance(data, TimeFrequencySeries):
        logger.info("Converting TimeFrequencySeries to ROOT.WSeries")
        output = convert_time_frequency_series_to_wseries(data)
    elif isinstance(data, pycbcTimeSeries):
        logger.info("Converting pycbc TimeSeries to ROOT.WSeries")
        output = convert_pycbc_timeseries_to_wavearray(data)
    else:
        output = data

    return output


def convert_timeseries_to_wavearray(data: TimeSeries):
    """
    Convert gwpy timeseries to wavearray with c++ function

    :param data: gwpy timeseries
    :return: Converted ROOT.wavearray
    :rtype: ROOT.wavearray
    """
    h = ROOT.wavearray(np.double)(len(data.value))

    data_val = np.round(data.value, 25)

    if not hasattr(ROOT, "_copy_to_wavearray"):
        declare_function()

    ROOT._copy_to_wavearray(data_val.ctypes.data_as(c_double_p), h, len(data.value))

    h.start(np.asarray(data.t0, dtype=np.double))
    h.rate(int(1. / np.asarray(data.dt, dtype=np.double)))

    return h


def convert_pycbc_timeseries_to_wavearray(data: pycbcTimeSeries):
    """
    Convert pycbc timeseries to wavearray with c++ function

    :param data: pycbc timeseries
    :return: Converted ROOT.wavearray
    :rtype: ROOT.wavearray
    """

    h = ROOT.wavearray(np.double)(len(data.data))

    data_val = np.round(data.data, 25)

    if not hasattr(ROOT, "_copy_to_wavearray"):
        declare_function()

    ROOT._copy_to_wavearray(data_val.ctypes.data_as(c_double_p), h, len(data.data))

    h.start(np.asarray(data.start_time, dtype=np.double))
    h.rate(int(1. / np.asarray(data.delta_t, dtype=np.double)))

    return h


def WSeries_to_matrix(w):
    """
    Convert WSeries to numpy matrix

    :param w: ROOT.WSeries
    :type w: ROOT.WSeries
    :return: Converted matrix
    :rtype: np.array
    """
    matrix = list()
    for n in range(w.maxLayer()):
        a = ROOT.wavearray(np.double)()
        w.getLayer(a, n)
        matrix.append(np.asarray(a))
    matrix = np.asarray(matrix, dtype=float)

    return matrix


def convert_wavearray_to_timeseries(h):
    """
    Convert wavearray to gwpy timeseries (get 3 times faster with c++ function)

    :param h: ROOT.wavearray
    :type h: ROOT.wavearray
    :return: Converted gwpy timeseries
    :rtype: gwpy.timeseries.TimeSeries
    """

    if not hasattr(ROOT, "_copy_to_wavearray"):
        declare_function()

    ar = np.array(ROOT._get_wavearray_data(h))

    ar = TimeSeries(ar, dt=1. / h.rate(), t0=h.start())

    return ar


def convert_wseries_to_timeseries(h):
    """
    Convert wavearray to gwpy timeseries (get 3 times faster with c++ function)

    :param h: ROOT.WSeries
    :type h: ROOT.WSeries
    :return: Converted gwpy timeseries
    :rtype: gwpy.timeseries.TimeSeries
    """

    if not hasattr(ROOT, "_copy_to_wavearray"):
        declare_function()

    ar = np.array(ROOT._get_wseries_data(h))

    ar = TimeSeries(ar, dt=1. / h.rate(), t0=h.start())

    return ar


def convert_wseries_to_pycbc_timeseries(h):
    """
    Convert wavearray to pycbc timeseries (get 3 times faster with c++ function)

    :param h: ROOT.WSeries
    :type h: ROOT.WSeries
    :return: Converted gwpy timeseries
    :rtype: gwpy.timeseries.TimeSeries
    """

    if not hasattr(ROOT, "_copy_to_wavearray"):
        declare_function()

    ar = np.array(ROOT._get_wseries_data(h))

    ar = pycbcTimeSeries(ar, delta_t=1. / h.rate(), epoch=h.start())

    return ar


def convert_wseries_to_time_frequency_series(h):
    """
    Convert wavearray to time frequency series

    :param h: ROOT.WSeries
    :type h: ROOT.WSeries
    :return: Time frequency series
    :rtype: TimeFrequencySeries
    """

    data = convert_wseries_to_pycbc_timeseries(h)

    return TimeFrequencySeries(data=data, wavelet=WDM(h.pWavelet), whiten_mode=h.w_mode)


def convert_time_frequency_series_to_wseries(h):
    """
    Convert time frequency series to wseries

    :param h: Time frequency series
    :type h: TimeFrequencySeries
    :return: Converted wseries
    :rtype: WSeries
    """

    data = convert_pycbc_timeseries_to_wavearray(h.data)
    w = ROOT.WSeries(np.double)(data, h.wavelet.wavelet)

    w.w_mode = h.whiten_mode

    return w

def _convert_wseries_to_wavearray(w):
    """
    Python fuction to convert wseries to wavearray. This is a slow function, should be replaced by c++ function

    :param w: ROOT.WSeries
    :return: Converted ROOT.wavearray
    :rtype: ROOT.wavearray
    """
    h = ROOT.wavearray(np.double)()

    for i in range(w.size()):
        h.append(w.data[i])

    h.start(w.start())
    h.rate(w.rate())

    return h


def _convert_wavearray_to_wseries(data):
    """
    This is to convert wavearray to wseries, it substituted the wseries.Forward(wavearray) that is not working.
    Maybe we can understand why

    :param data: ROOT.wavearray
    :return: Converted ROOT.WSeries
    :rtype: ROOT.WSeries
    """
    w = ROOT.WSeries(np.double)()

    for d in data:
        w.append(d)

    w.start(data.start())
    w.rate(data.rate())
    w.wrate(0.)
    w.f_high = data.rate() / 2.
    w.pWavelet.allocate(w.size(),
                        w.data)
    return w


def _convert_numpy_to_wavearray(data: np.array, start: np.double, stop: np.double, rate: int):
    """
    Convert numpy array to wavearray with python loop

    :param data: numpy array
    :type data: np.array
    :param start: start time
    :type start: np.double
    :param stop: stop time
    :type stop: np.double
    :param rate: sample rate
    :type rate: int
    :return: Converted ROOT.wavearray
    :rtype: ROOT.wavearray
    """
    data = np.asarray(data, dtype=float)
    h = ROOT.wavearray(np.double)()
    h.start(np.asarray(start, dtype=np.double))
    h.stop(np.asarray(stop, dtype=np.double))
    h.rate(int(rate))

    for d in data:
        h.append(d)

    return h


def _data_to_TFmap(h):
    # TODO: write something
    lev = int(h.rate() / 2)  # TFmap and wavearray should have the same rate?
    wdtf = ROOT.WDM(np.double)(lev, 2 * lev, 6, 10)  # what is this? there is a problem with this function
    w = ROOT.WSeries(np.double)(h, wdtf)
    w.Forward()  # Perform n steps of forward wavelet transform

    return w



def _transform(h, time_layer, freq_layer):
    # Edge here is not defined
    # fLow and fHigh not defined
    """

    Input
    -----
    h : (wavearray) input data
    time_layer: (int) time bins
    freq_layer: (freq) frequency bins
    """

    WDMt = ROOT.WDM(np.double)(time_layer,
                               freq_layer,
                               6, 10)
    w = ROOT.WSeries(np.double)()
    w.Forward(h, WDMt)

    return w


def _crop_wavearray(data, rate, totalscratch):
    """
    Crop wavearray according to desired totalscratch

    Input
    -----
    data: (wavearray) data to crop
    rate: (int) sampling rate
    totalscratch: (float) seconds to crop at both sides

    Output
    -----
    data_crop: (wavearray) cropped data

    """
    data_crop = ROOT.wavearray(np.double)()

    begin = int(rate * totalscratch)
    end = int(len(data) - rate * totalscratch)

    for i in range(begin, end):
        data_crop.append(data.data[i])

    data_crop.start(data.start())
    data_crop.rate(data.rate())

    return data_crop


def _get_histogram_as_matrix(histogram, t0, duration, time_bins, F1, F2, freq_bins):
    """
    Returns Z-histogram as numpy matrix

    Input
    -----
    histogram: (WSeries) Z-values of heatmap from cWB

    Output
    ------
    time_arr: (np.array) time dimension as numpy array
    freq_arr: (np.array) frequency dimension as numpy array
    z_arr: (np.array) z-values of heatmap as numpy array

    """

    z_arr = list()

    for i in range(histogram.GetNbinsX()):
        cols = list()
        for j in range(histogram.GetNbinsY()):
            cols.append(histogram.GetBinContent(i, j))
        z_arr.append(cols)
    z_arr = np.asarray(z_arr, dtype=float)

    time_arr = np.linspace(t0, duration, time_bins)
    freq_arr = np.linspace(F1, F2, freq_bins)

    return time_arr, freq_arr, z_arr


def _scaling(x):
    """
    This function returns Marco's scaling. Used in previous papers.
    """
    y = 1 - np.exp(- x)
    return y
