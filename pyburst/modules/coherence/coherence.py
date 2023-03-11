import copy
import time
from multiprocessing import Pool
import numpy as np
import ROOT
import logging
from pyburst.config import Config
from pyburst.modules.network import create_network
from pyburst.types import TimeFrequencySeries
from pyburst.utils import convert_to_wavearray


logger = logging.getLogger(__name__)


def coherence_parallel(config, tf_maps, wdm_list, nRMS_list):
    """
    Calculate the coherence parallelly, a temporary network object will be created for each process to avoid conflict.
    MRA won't be loaded for the temporary network object.

    :param config: user configuration
    :type config: Config
    :param tf_maps: list of strain
    :type tf_maps: list[TimeFrequencySeries]
    :param wdm_list: list of wdm
    :type wdm_list: list[WDM]
    :param nRMS_list: list of noise RMS
    :type nRMS_list: list[TimeFrequencySeries]
    :return:
    """
    timer_start = time.perf_counter()
    up_n = config.rateANA // 1024
    if up_n < 1:
        up_n = 1

    sparse_table_list = []
    pwc_list = []

    with Pool(processes=min(config.nproc, config.nRES)) as pool:
        tasks = []
        for i in range(config.nRES):
            tasks.append((i, config, tf_maps, nRMS_list, wdm_list[i], up_n))
        for sparse_table, pwc in pool.starmap(_coherence_single_res, tasks):
            sparse_table_list.append(sparse_table)
            pwc_list += pwc

    logger.info("Coherence time totally: %f s", time.perf_counter() - timer_start)
    return sparse_table_list, pwc_list


def coherence(config, tf_maps, wdm_list, nRMS_list, net=None):
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
    :param tf_maps: list of strain
    :type tf_maps: list[TimeFrequencySeries]
    :param wdm_list: list of wdm
    :type wdm_list: list[WDM]
    :param nRMS_list: list of noise RMS
    :type nRMS_list: list[TimeFrequencySeries]
    :param net: network, if None, create a temporary minimum network object with wdm_list and nRMS_list
    :type net: ROOT.network, optional
    :return: (sparse_table_list, pwc_list)
    :rtype: (list[ROOT.SSeries], list[ROOT.PWC])
    """
    # calculate upsample factor
    timer_start = time.perf_counter()
    up_n = config.rateANA // 1024
    if up_n < 1:
        up_n = 1

    sparse_table_list = []
    pwc_list = []

    for i in range(config.nRES):
        sparse_table, pwc_list_res = _coherence_single_res(i, config, tf_maps, nRMS_list, wdm_list[i], up_n, net)
        sparse_table_list.append(sparse_table)
        pwc_list += pwc_list_res

    logger.info("----------------------------------------")
    logger.info("Coherence time: %f s", time.perf_counter() - timer_start)
    logger.info("----------------------------------------")

    return sparse_table_list, pwc_list


def _coherence_single_res(i, config, tf_maps, nRMS_list, wdm, up_n, net=None):
    """
    Calculate the coherence for a single resolution

    :param i: index of resolution
    :type i: int
    :param config: configuration object
    :type config: Config
    :param net: network
    :type net: ROOT.network
    :param tf_maps: list of strain
    :type tf_maps: list[TimeFrequencySeries]
    :param wdm: wdm used for current resolution
    :type wdm: WDM
    :param m_tau: maximum delay
    :type m_tau: float
    :param up_n: upsample factor
    :type up_n: int
    :return: (sparse_table, pwc_list)
    :rtype: (ROOT.SSeries, list[ROOT.netcluster])
    """
    # timer
    timer_start = time.perf_counter()
    if net is None:
        # for paralleling, create a new network to avoid conflict
        net = create_network(1, config, tf_maps, nRMS_list, minimum=True)

    m_tau = net.getDelay('MAX')

    wc = ROOT.netcluster()

    # print level infos
    level = config.l_high - i
    layers = 2 ** level if level > 0 else 0
    rate = config.rateANA // 2 ** level

    # use string instead of directly logging to avoid messy output in parallel
    logger_info = "level : %d\t rate(hz) : %d\t layers : %d\t df(hz) : %f\t dt(ms) : %f \n" % (
        level, rate, layers, config.rateANA / 2. / (2 ** level), 1000. / rate)

    # produce TF maps with max over the sky energy
    alp = 0.0
    for n in range(len(config.ifo)):
        # tf_map = ROOT.WSeries(np.double)(tf_maps[n])
        ts = convert_to_wavearray(tf_maps[n])
        # alp += tf_map.maxEnergy(ts, wdm, m_tau, up_n, net.pattern)
        # tf_map.setlow(config.fLow)
        # tf_map.sethigh(config.fHigh)
        alp += net.getifo(n).getTFmap().maxEnergy(ts, wdm.wavelet,
                                                  m_tau, up_n,
                                                  net.pattern)
        net.getifo(n).getTFmap().setlow(config.fLow)
        net.getifo(n).getTFmap().sethigh(config.fHigh)

    logger_info += "max energy in units of noise variance: %g \n" % alp

    # logger.info("max energy in units of noise variance: %g", alp)
    alp = alp / config.nIFO

    if net.pattern != 0:
        Eo = net.THRESHOLD(config.bpp, alp)
    else:
        Eo = net.THRESHOLD(config.bpp)

    logger_info += "thresholds in units of noise variance: Eo=%g Emax=%g \n" % (Eo, Eo * 2)

    # set veto array
    TL = net.setVeto(config.iwindow)
    logger_info += "live time in zero lag: %g \n" % TL

    if TL <= 0.:
        raise ValueError("live time is zero")

    # init sparse table (used in supercluster stage : set the TD filter size)
    sparse_table = []
    wdm.set_td_filter(config.TDSize, 1)
    for n in range(config.nIFO):
        ws = ROOT.WSeries(np.double)(convert_to_wavearray(tf_maps[n]), wdm.wavelet)
        ws.Forward()
        ss = ROOT.SSeries(np.double)()
        ss.SetMap(ws)
        ss.SetHalo(m_tau)
        sparse_table.append(ss)

    logger_info += "lag | clusters | pixels \n"

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
        logger_info += "%3d |%9d |%7d \n" % (j, csize_tot, psize_tot)
        # logger.info("%3d |%9d |%7d ", j, csize_tot, psize_tot)

        # add core pixels to sparse table
        for n in range(config.nIFO):
            sparse_table[n].AddCore(n, pwc)

        pwc.clear()

    for n in range(config.nIFO):
        sparse_table[n].UpdateSparseTable()
        sparse_table[n].Clean()

    logger_info += "Coherence time for single level: %f s" % (time.perf_counter() - timer_start)

    logger.info(logger_info)
    return sparse_table, pwc_list
