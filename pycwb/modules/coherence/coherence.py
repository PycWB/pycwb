import time
from multiprocessing import Pool
import ROOT
import logging
from pycwb.config import Config
from pycwb.types.network import Network
from pycwb.modules.cwb_conversions import convert_to_wavearray, convert_netcluster_to_fragment_clusters
from pycwb.modules.multi_resolution_wdm import create_wdm_for_level

logger = logging.getLogger(__name__)


def coherence(config, tf_maps, nRMS_list, net=None):
    """
    Select the significant pixels

    Loop over resolution levels (nRES)

    * Loop over detectors (cwb::nIFO)

      * Compute the maximum energy of TF pixels (WSeries<double>::maxEnergy)
      * Set pixel energy selection threshold (network::THRESHOLD)
      * Loop over time lags (network::nLag)

      * Select the significant pixels (network::getNetworkPixels)
      * Single resolution clustering (network::cluster)

    Parameters
    ----------
    config : pycwb.config.Config
        Configuration object
    tf_maps : list of pycwb.types.time_frequency_series.TimeFrequencySeries
        List of time-frequency maps
    nRMS_list : list of pycwb.types.time_frequency_series.TimeFrequencySeries
        List of noise RMS
    net : pycwb.types.network.Network, optional
        Network object, by default None

    Returns
    -------
    fragment_clusters: list[list[pycwb.types.network_cluster.FragmentCluster]]
        List of fragment clusters
    """
    # calculate upsample factor
    timer_start = time.perf_counter()
    logger.info("Start coherence" + " in parallel" if config.nproc > 1 else "")

    # upper sample factor
    up_n = int(config.rateANA / 1024)
    if up_n < 1:
        up_n = 1

    if config.nproc > 1:
        with Pool(processes=min(config.nproc, config.nRES)) as pool:
            fragment_clusters_multi_res = pool.starmap(coherence_single_res,
                                                       [(i, config, tf_maps, nRMS_list, up_n) for i in
                                                        range(config.nRES)])
    else:
        fragment_clusters_multi_res = [coherence_single_res(i, config, tf_maps, nRMS_list, up_n, net) for i in
                                       range(config.nRES)]

    # flat the array
    # fragment_clusters = [item for sublist in fragment_clusters_multi_res for item in sublist]

    logger.info("----------------------------------------")
    logger.info("Coherence time totally: %f s", time.perf_counter() - timer_start)
    logger.info("----------------------------------------")

    return fragment_clusters_multi_res


def coherence_single_res_wrapper(i, config, data):
    return coherence_single_res(i, config, data[0], data[1])


def coherence_single_res(i, config, tf_maps, nRMS_list, up_n=None, net=None):
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
    :param up_n: upsample factor
    :type up_n: int
    :return: (sparse_table, fragment_clusters)
    :rtype: (ROOT.SSeries, list[ROOT.netcluster])
    """
    # timer
    timer_start = time.perf_counter()

    if up_n is None:
        # upper sample factor
        up_n = int(config.rateANA / 1024)
        if up_n < 1:
            up_n = 1

    wdm = create_wdm_for_level(config, config.WDM_level[i])

    if net is None:
        # for paralleling, create a new network to avoid conflict
        net = Network(config, tf_maps, nRMS_list, silent=True)

    # print level infos
    level = config.l_high - i
    layers = 2 ** level if level > 0 else 0
    rate = config.rateANA // 2 ** level

    # use string instead of directly logging to avoid messy output in parallel
    logger_info = "level : %d\t rate(hz) : %d\t layers : %d\t df(hz) : %f\t dt(ms) : %f \n" % (
        level, rate, layers, config.rateANA / 2. / (2 ** level), 1000. / rate)

    fragment_clusters = []
    ###############################
    # cWB2G coherence calculation #
    ###############################

    # produce TF maps with max over the sky energy
    alp = 0.0

    # FIXME: max time delay is different to pycbc
    config.max_delay = net.get_max_delay()
    for n in range(len(config.ifo)):
        ts = convert_to_wavearray(tf_maps[n])
        ts.Edge = config.segEdge
        # TODO: WSeries.putLayer is updated internally, here requires the wave packet pattern
        # https://gwburst.gitlab.io/documentation/latest/html/running.html#wave-packet-parameters
        # The max function is not just calculate the max values, but also set the whole TF map to
        # the max value over delayed time series, this is the most time consuming part in coherence
        alp += net.get_ifo(n).getTFmap().maxEnergy(ts, wdm.wavelet, config.max_delay, up_n, net.pattern)
        net.get_ifo(n).getTFmap().setlow(config.fLow)
        net.get_ifo(n).getTFmap().sethigh(config.fHigh)

    logger_info += "max energy in units of noise variance: %g \n" % alp

    alp = alp / config.nIFO

    # set threshold
    if net.pattern != 0:
        Eo = net.threshold(config.bpp, alp)
    else:
        Eo = net.threshold(config.bpp)

    logger_info += "thresholds in units of noise variance: Eo=%g Emax=%g \n" % (Eo, Eo * 2)

    # set veto array
    TL = net.set_veto(config.iwindow)
    logger_info += "live time in zero lag: %g \n" % TL

    if TL <= 0.:
        raise ValueError("live time is zero")

    logger_info += "lag | clusters | pixels \n"

    # temporary storage for sparse table
    wc = ROOT.netcluster()

    # loop over time lags
    for j in range(int(net.nLag)):
        # select pixels above Eo
        net.get_network_pixels(j, Eo)
        # get pixel list
        pwc = net.get_cluster(j)
        if net.pattern != 0:
            # cluster pixels
            net.cluster(j, 2, 3)
            wc.cpf(pwc, False)
            # remove pixels below subrho
            # TODO: keep in mind, subrho can be more flexible.
            # TODO: pythonize this algorithm in network cluster
            wc.select("subrho", config.select_subrho)
            # remove pixels below subnet
            wc.select("subnet", config.select_subnet)
            # copy selected pixels back to pwc
            pwc.cpf(wc, False)
        else:
            net.cluster(j, 1, 1)

        fragment_cluster = convert_netcluster_to_fragment_clusters(pwc)
        fragment_clusters.append(fragment_cluster)

        logger_info += "%3d |%9d |%7d \n" % (j, fragment_cluster.event_count(), fragment_cluster.pixel_count())

        pwc.clear()

    for j in range(int(net.nLag)):
        pwc = net.get_cluster(j)
        pwc.clean(1)

    ts.resize(0)  # clear the time series to free memory
    ###############################

    logger_info += "Coherence time for single level: %f s" % (time.perf_counter() - timer_start)

    logger.info(logger_info)
    return fragment_clusters


coherence_parallel = coherence
