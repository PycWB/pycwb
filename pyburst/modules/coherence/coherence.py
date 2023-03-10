import copy
import time
from multiprocessing import Pool
import numpy as np
import ROOT
import logging
from pycbc.types.timeseries import TimeSeries as pycbcTimeSeries
from pyburst.config import Config
from pyburst.types import TimeFrequencySeries
from pyburst.utils import convert_wseries_to_time_frequency_series, convert_to_wseries

logger = logging.getLogger(__name__)


def coherence_parallel(config, net, tf_maps, wdm_list):
    """
    Calculate the coherence parallelly (Not working for now)

    :param config: config
    :type config: Config
    :param net: network
    :type net: ROOT.network
    :param tf_maps: list of strain
    :type tf_maps: list[ROOT.wavearray(np.double)]
    :param wdm_list: list of wdm
    :type wdm_list: list[ROOT.WDM(np.double)]
    :return:
    """
    timer_start = time.perf_counter()
    up_n = config.rateANA // 1024
    if up_n < 1:
        up_n = 1

    sparse_table_list = []
    pwc_list = []
    m_tau = net.getDelay('MAX')

    with Pool(processes=min(config.nproc, config.nRES)) as pool:
        tasks = []
        for i in range(config.nRES):
            tasks.append((i, config, copy.deepcopy(net),tf_maps, wdm_list[i], m_tau, up_n))
        for sparse_table, pwc in pool.starmap(_coherence_single_res, tasks):
            sparse_table_list.append(sparse_table)
            pwc_list += pwc

    logger.info("Coherence time: %f s", time.perf_counter() - timer_start)
    return sparse_table_list, pwc_list


def coherence(config, net, tf_maps, wdm_list):
    """
    Select the significant pixels

    Loop over resolution levels (nRES)

    * Loop over detectors (cwb::nIFO)

      * Compute the maximum energy of TF pixels (WSeries<double>::maxEnergy)
    * Set pixel energy selection threshold (network::THRESHOLD)
    * Loop over time lags (network::nLag)

      * Select the significant pixels (network::getNetworkPixels)
      * Single resolution clustering (network::cluster)

    :param config: config
    :type config: Config
    :param net: network
    :type net: ROOT.network
    :param tf_maps: list of strain
    :type tf_maps: list[TimeFrequencySeries]
    :param wdm_list: list of wdm
    :type wdm_list: list[TimeFrequencySeries]
    :return: (sparse_table_list, pwc_list)
    :rtype: (list[ROOT.SparseTable], list[ROOT.PWC])
    """
    tf_maps = [convert_to_wseries(tf) for tf in tf_maps]

    # calculate upsample factor
    timer_start = time.perf_counter()
    up_n = config.rateANA // 1024
    if up_n < 1:
        up_n = 1

    sparse_table_list = []
    pwc_list = []
    m_tau = net.getDelay('MAX')

    for i in range(config.nRES):
        sparse_table, pwc_list_res = _coherence_single_res(i, config, net, tf_maps, wdm_list[i], m_tau, up_n)
        sparse_table_list.append(sparse_table)
        pwc_list += pwc_list_res

    logger.info("----------------------------------------")
    logger.info("Coherence time: %f s", time.perf_counter() - timer_start)
    logger.info("----------------------------------------")

    return sparse_table_list, pwc_list


def _coherence_single_res(i, config, net, tf_maps, wdm, m_tau, up_n):
    """
    Calculate the coherence for a single resolution

    :param i: index of resolution
    :type i: int
    :param config: configuration object
    :type config: Config
    :param net: network
    :type net: ROOT.network
    :param tf_maps: list of strain
    :type tf_maps: list[ROOT.wavearray(np.double)]
    :param wdm: wdm used for current resolution
    :type wdm: ROOT.WDM(np.double)
    :param m_tau: maximum delay
    :type m_tau: float
    :param up_n: upsample factor
    :type up_n: int
    :return: (sparse_table, pwc_list)
    :rtype: (ROOT.SparseTable, list[ROOT.PWC])
    """

    wc = ROOT.netcluster()

    # print level infos
    level = config.l_high - i
    layers = 2 ** level if level > 0 else 0
    rate = config.rateANA // 2 ** level
    logger.info("level : %d\t rate(hz) : %d\t layers : %d\t df(hz) : %f\t dt(ms) : %f",
                level, rate, layers, config.rateANA / 2. / (2 ** level),
                1000. / rate)

    # produce TF maps with max over the sky energy
    alp = 0.0
    for n in range(len(config.ifo)):
        # tf_map = ROOT.WSeries(np.double)(tf_maps[n])
        ts = ROOT.wavearray(np.double)(tf_maps[n])
        # alp += tf_map.maxEnergy(ts, wdm, m_tau, up_n, net.pattern)
        # tf_map.setlow(config.fLow)
        # tf_map.sethigh(config.fHigh)
        alp += net.getifo(n).getTFmap().maxEnergy(ts, wdm,
                                                  m_tau, up_n,
                                                  net.pattern)
        net.getifo(n).getTFmap().setlow(config.fLow)
        net.getifo(n).getTFmap().sethigh(config.fHigh)
    logger.info("max energy in units of noise variance: %g", alp)
    alp = alp / config.nIFO

    if net.pattern != 0:
        Eo = net.THRESHOLD(config.bpp, alp)
    else:
        Eo = net.THRESHOLD(config.bpp)
    logger.info("thresholds in units of noise variance: Eo=%g Emax=%g", Eo, Eo * 2)

    # set veto array
    TL = net.setVeto(config.iwindow)
    logger.info("live time in zero lag: %g", TL)
    if TL <= 0.:
        raise ValueError("live time is zero")

    # init sparse table (used in supercluster stage : set the TD filter size)
    sparse_table = []
    wdm.setTDFilter(config.TDSize, 1)
    for n in range(config.nIFO):
        ws = ROOT.WSeries(np.double)(tf_maps[n], wdm)
        ws.Forward()
        ss = ROOT.SSeries(np.double)()
        ss.SetMap(ws)
        ss.SetHalo(m_tau)
        sparse_table.append(ss)

    logger.info("lag | clusters | pixels ")

    csize_tot = 0
    psize_tot = 0

    pwc_list = []
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
        logger.info("%3d |%9d |%7d ", j, csize_tot, psize_tot)

        # add core pixels to sparse table
        for n in range(config.nIFO):
            sparse_table[n].AddCore(n, pwc)

        pwc.clear()

    for n in range(config.nIFO):
        sparse_table[n].UpdateSparseTable()
        sparse_table[n].Clean()

    return sparse_table, pwc_list