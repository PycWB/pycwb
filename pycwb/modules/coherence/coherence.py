import copy

import numpy as np
import ROOT
import logging
from pycwb.config import Config

logger = logging.getLogger(__name__)


def select_pixels(config: Config, net: ROOT.network,
                  strain_list: list[ROOT.wavearray(np.double)],
                  wdm_list: list[ROOT.WDM(np.double)]):
    """
    select pixels
    :param config: config
    :param net: network
    :param strain_list: list of strain
    :param wdm_list: list of wdm
    :param threshold_list: list of threshold
    :return:
    """
    # init sparse table (used in supercluster stage : set the TD filter size)
    m_tau = net.getDelay('MAX')

    # calculate upsample factor
    up_n = config.rateANA // 1024
    if up_n < 1:
        up_n = 1

    sparse_table_list = []
    pwc_list = []
    m_tau = net.getDelay('MAX')
    wc = ROOT.netcluster()

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

        sparse_table = []
        wdm_list[i].setTDFilter(config.TDSize, 1)
        for n in range(config.nIFO):
            ws = ROOT.WSeries(np.double)(strain_list[n], wdm_list[i])
            ws.Forward()
            ss = ROOT.SSeries(np.double)()
            ss.SetMap(ws)
            ss.SetHalo(m_tau)
            sparse_table.append(ss)

        logger.info("lag|clusters|pixels ")

        csize_tot = 0
        psize_tot = 0

        for j in range(int(net.nLag)):
            # select pixels above Eo
            net.getNetworkPixels(j, Eo)
            # get pixel list
            pwc = net.getwc(j)
            if net.pattern != 0:
                # cluster pixels
                net.cluster(2, 3)
                wc.cpf(pwc, False)
                # remove pixels below subrho
                wc.select("subrho", config.select_subrho)
                # remove pixels below subnet
                wc.select("subnet", config.select_subnet)
                # copy selected pixels back to pwc
                pwc.cpf(wc, False)
            else:
                net.cluster(1, 1)

            pwc_list.append(copy.deepcopy(pwc))
            # store cluster into temporary job file
            csize_tot += pwc.csize()
            psize_tot += pwc.size()
            logger.info("%3d|%9d|%7d ", j, csize_tot, psize_tot)

            # add core pixels to sparse table
            for n in range(config.nIFO):
                sparse_table[n].AddCore(n, pwc)

            pwc.clear()

        for n in range(config.nIFO):
            sparse_table[n].UpdateSparseTable()
            sparse_table[n].Clean()
        sparse_table_list.append(sparse_table)

    return sparse_table_list, pwc_list






