from pycwb.utils import WSeries_to_matrix
from gwpy.spectrogram import Spectrogram
import numpy as np
import ROOT


def plot_spectrogram(wavearray, xmin=None, xmax=None):
    wdm = ROOT.WDM(np.double)(32, 64, 4, 8)
    tf_map = ROOT.WSeries(np.double)(wavearray, wdm)
    tf_map.Forward()

    plot = Spectrogram(WSeries_to_matrix(tf_map).T,
                       t0=tf_map.start(),
                       dt=1/tf_map.wRate,
                       f0=tf_map.f_low,
                       df=tf_map.resolution()
                       ).plot()
    ax = plot.gca()
    if xmin or xmax:
        ax.set_xlim(xmin, xmax)
