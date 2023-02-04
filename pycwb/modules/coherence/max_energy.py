from gwpy.timeseries import TimeSeries
from pycwb import utils as ut
import numpy as np
import copy
import ROOT

from pycwb.config import Config


def max_energy(config: Config, net: ROOT.network, h: ROOT.wavearray(np.double),
               wdm_list: list):
    """
    produce TF maps with max over the sky energy
    Input
    -----
    config: Config
    net: ROOT.network
    h: ROOT.wavearray(np.double)
    wdm_list: list
    :return:
    alp: int
    """
    # maximum time delay
    m_tau = net.getDelay('MAX')

    # calculate upsample factor
    up_n = config.rateANA // 1024
    if up_n < 1:
        up_n = 1

    alp_list = []
    for i in range(config.nRES):
        alp = 0
        for n in range(len(config.ifo)):
            alp += net.getifo(n).getTFmap().maxEnergy(h, wdm_list[i],
                                                      m_tau, up_n,
                                                      net.pattern)
            net.getifo(n).getTFmap().setlow(config.fLow)
            net.getifo(n).getTFmap().sethigh(config.fHigh)
        alp_list.append(alp)

    return alp_list