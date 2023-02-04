import numpy as np
import ROOT
import logging
from pycwb.config import Config

logger = logging.getLogger(__name__)


def threshold(config: Config, net: ROOT.network, alp_list: np.array):
    """
    threshold on pixel energy
    :param net:
    :param config:
    :param alp_list:
    :return:
    """
    threshold_list = []
    for i, alp in enumerate(alp_list / len(config.ifo)):
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
