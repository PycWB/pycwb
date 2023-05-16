from gwpy.timeseries import TimeSeries
from pycbc.types.timeseries import TimeSeries as pycbcTimeSeries

from pycwb.types.time_frequency_series import TimeFrequencySeries
from pycwb.types.wdm import WDM
from pycwb.modules.cwb_conversions import WSeries_to_matrix, convert_wavearray_to_timeseries
from gwpy.spectrogram import Spectrogram
import numpy as np
import ROOT


def plot_spectrogram(wavearray, xmin=None, xmax=None, figsize=(24, 6), gwpy_plot=True):
    """
    Plots a spectrogram of the given waveform array

    :param wavearray: given waveform array
    :type wavearray: ROOT.WSeries or gwpy.timeseries.TimeSeries
    :param xmin: x-axis minimum for matplotlib plot
    :type xmin: float, optional
    :param xmax: x-axis maximum for matplotlib plot
    :type xmax: float, optional
    :param figsize: figure size for matplotlib plot, default (24, 6)
    :type figsize: tuple, optional
    :param gwpy_plot: whether to use gwpy or ROOT to transform the waveform array to a spectrogram, default True
    :type gwpy_plot: bool, optional
    :return: the matplotlib plot object for user to further modify or save
    :rtype: matplotlib.pyplot
    """

    if gwpy_plot:
        if isinstance(wavearray, TimeFrequencySeries):
            wavearray = TimeSeries.from_pycbc(wavearray.data)
        elif isinstance(wavearray, pycbcTimeSeries):
            wavearray = TimeSeries.from_pycbc(wavearray)
        elif isinstance(wavearray, TimeSeries):
            pass
        else:
            wavearray = convert_wavearray_to_timeseries(wavearray)

    if isinstance(wavearray, TimeSeries):
        wavearray = wavearray.crop(xmin, xmax)
        specgram = wavearray.spectrogram(0.5, fftlength=0.5, overlap=0.49) ** (1 / 2.)
        plot = specgram.plot(norm='log', figsize=figsize)  # vmin=1e-23, vmax=1e-19
        ax = plot.gca()
        ax.set_ylim(15, 1000)
    else:
        wdm = WDM(32, 64, 4, 8)
        tf_map = ROOT.WSeries(np.double)(wavearray, wdm.wavelet)
        tf_map.Forward()

        plot = Spectrogram(WSeries_to_matrix(tf_map).T,
                           t0=tf_map.start(),
                           dt=1 / tf_map.wRate,
                           f0=tf_map.f_low,
                           df=tf_map.resolution()
                           ).plot()
        ax = plot.gca()
        ax.set_ylim(tf_map.f_low or 15, 1000)

    ax.colorbar(label='GW strain ASD [strain/$\sqrt{\mathrm{Hz}}$]')
    ax.set_yscale('log')
    if xmin or xmax:
        ax.set_xlim(xmin, xmax)

    return plot
