from gwpy.timeseries import TimeSeries
from pycwb.utils import WSeries_to_matrix, convert_wavearray_to_timeseries
from gwpy.spectrogram import Spectrogram
import numpy as np
import ROOT


def plot_spectrogram(wavearray, xmin=None, xmax=None, gwpy_plot=False):
    if gwpy_plot:
        wavearray = convert_wavearray_to_timeseries(wavearray)

    if isinstance(wavearray, TimeSeries):
        specgram = wavearray.spectrogram(0.5, fftlength=0.5, overlap=0.49) ** (1 / 2.)
        plot = specgram.plot(norm='log')  # vmin=1e-23, vmax=1e-19
        ax = plot.gca()
        ax.set_ylim(15, 1000)
    else:
        wdm = ROOT.WDM(np.double)(32, 64, 4, 8)
        tf_map = ROOT.WSeries(np.double)(wavearray, wdm)
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
