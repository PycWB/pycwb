import numpy as np
import ROOT
import logging
from pycwb.config import Config

logger = logging.getLogger(__name__)


def threshold(config: Config, net: ROOT.network,
              strain_list: list[ROOT.wavearray(np.double)],
              wdm_list: list[ROOT.WDM(np.double)]):
    """
    threshold on pixel energy
    :param net:
    :param config:
    :param alp_list:
    :return:
    """
    m_tau = net.getDelay('MAX')

    # calculate upsample factor
    up_n = config.rateANA // 1024
    if up_n < 1:
        up_n = 1
    threshold_list = []
    for i in range(config.nRES):
        alp = 0.0
        for n in range(len(config.ifo)):
            t = net.getifo(n).getTFmap().maxEnergy(strain_list[n], wdm_list[i],
                                                   m_tau, up_n,
                                                   net.pattern)
            alp += t
            net.getifo(n).getTFmap().setlow(config.fLow)
            net.getifo(n).getTFmap().sethigh(config.fHigh)


        alp = alp / len(config.ifo)
        logger.info("average max energy: %g", alp)

        if net.pattern != 0:
            Eo = net.THRESHOLD(config.bpp, alp)
        else:
            Eo = net.THRESHOLD(config.bpp)
        logger.info("thresholds in units of noise variance: Eo=%g Emax=%g", Eo, Eo * 2)
        logger.info(f"cwb2G::Coherence -RES:{i}-THR:{Eo}")
        # set veto array
        # TODO: test if we need to re-export net again from here, or set veto in init
        TL = net.setVeto(config.iwindow)
        logger.info("live time in zero lag: %g", TL)
        if TL <= 0.:
            raise ValueError("live time is zero")
        threshold_list.append(Eo)

    return np.array(threshold_list)
