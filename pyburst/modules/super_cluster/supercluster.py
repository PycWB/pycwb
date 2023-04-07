import copy

import ROOT
import logging
import time
import numpy as np
from pyburst.config import Config
from pyburst.constants import MIN_SKYRES_HEALPIX
from pyburst.conversions import convert_fragment_clusters_to_netcluster, convert_sparse_series_to_sseries
from pyburst.modules.network import update_sky_map, update_sky_mask, restore_skymap
from pyburst.types import FragmentCluster

logger = logging.getLogger(__name__)


def supercluster(config, network, wdm_list, fragment_clusters, sparse_table_list):
    """
    Multi resolution clustering & Rejection of the sub-threshold clusters

    Loop over time lags \n
    * Read clusters from job file (netcluster::read) \n
    * Multi resolution clustering (netcluster::supercluster) \n
    * Compute for each pixel the time delay amplitudes (netcluster::loadTDampSSE) \n
    * Rejection of the sub-threshold clusters (network::subNetCut) \n
    * Defragment clusters (netcluster::defragment) \n

    :param config: user configuration
    :type config: Config
    :param network: network
    :type network: Network
    :param wdm_list: list of wavelets
    :type wdm_list: list[WDM]
    :param fragment_clusters: fragment clusters
    :type fragment_clusters: list[FragmentCluster]
    :param sparse_table_list: list of sparse tables
    :type sparse_table_list: list[SparseTimeFrequencySeries]
    :return: the list of clusters
    :rtype: list[FragmentCluster]
    """
    # timer
    timer_start = time.perf_counter()

    # decrease skymap resolution to improve subNetCut performances
    skyres = MIN_SKYRES_HEALPIX if config.healpix > MIN_SKYRES_HEALPIX else 0
    if skyres > 0:
        network.update_sky_map(config, skyres)
        network.update_sky_mask(config, skyres)

    hot = []
    for n in range(config.nIFO):
        hot.append(network.get_ifo(n).getHoT())
    # set low-rate TD filters
    for wdm in wdm_list:
        wdm.set_td_filter(config.TDSize, 1)

    # merge cluster
    cluster = copy.deepcopy(fragment_clusters[0])
    if len(fragment_clusters) > 1:
        for fragment_cluster in fragment_clusters[1:]:
            cluster.clusters += fragment_cluster.clusters

    pwc_list = []

    ###############################
    # cWB2G supercluster
    ###############################
    # convert to netcluster
    cluster = convert_fragment_clusters_to_netcluster(cluster)

    # add wavelets to network
    for wdm in wdm_list:
        network.add_wavelet(wdm)

    # read sparse map to detector for pwc.loadTDampSSE
    for n in range(config.nIFO):
        det = network.get_ifo(n)
        det.sclear()
        for sparse_table in sparse_table_list:
            det.vSS.push_back(convert_sparse_series_to_sseries(sparse_table[n]))

    for j in range(int(network.nLag)):
        # cycle = cfg.simulation ? ifactor : Long_t(NET.wc_List[j].shift);
        cycle = int(network.get_cluster(j).shift)
        cycle_name = f"lag={cycle}"

        logger.info("-> Processing %s ...", cycle_name)
        logger.info("   --------------------------------------------------")
        logger.info("    coher clusters|pixels      : %6d|%d", cluster.esize(0), cluster.psize(0))

        if config.l_high == config.l_low:
            network.net.pair = False
        if network.pattern != 0:
            network.net.pair = False

        cluster.supercluster('L',network.net.e2or,config.TFgap,False)
        logger.info("    super clusters|pixels      : %6d|%d", cluster.esize(0), cluster.psize(0))

        # defragmentation for pattern != 0
        if network.pattern != 0:
            cluster.defragment(config.Tgap, config.Fgap)
            logger.info("   defrag clusters|pixels      : %6d|%d", cluster.esize(0), cluster.psize(0))

        # copy selected clusters to network
        pwc = network.get_cluster(j)
        pwc.cpf(cluster, False)

        # apply subNetCut() only for pattern=0 || cfg.subnet>0 || cfg.subcut>0 || cfg.subnorm>0 || cfg.subrho>=0
        if network.pattern == 0 or config.subnet > 0 or config.subcut > 0 or config.subnorm > 0 or config.subrho >= 0:
            # set Acore and netRHO
            if config.subacor > 0:
                network.net.acor = config.subacor
            if config.subrho > 0:
                network.net.netRHO = config.subrho

            network.set_delay_index(hot[0].rate())
            pwc.setcore(False)

            psel = 0
            while True:
                # TODO: pythonize this
                count = pwc.loadTDampSSE(network.net, 'a', config.BATCH, config.LOUD)
                psel += network.sub_net_cut(j, config.subnet, config.subcut, config.subnorm)
                if count < 10000:
                    break
            logger.info("   subnet clusters|pixels      : %6d|%d", network.n_events, pwc.psize(-1))

            # restore Acore and netRHO
            if config.subacor > 0:
                network.net.acor = config.Acore
            if config.subrho > 0:
                network.net.netRHO = config.netRHO

        if network.pattern == 0:
            # TODO: pythonize this
            pwc.defragment(config.Tgap, config.Fgap)
            logger.info("   defrag clusters|pixels      : %6d|%d", cluster.esize(0), cluster.psize(0))

        # convert to FragmentCluster and append to list
        fragment_cluster = copy.deepcopy(FragmentCluster().from_netcluster(pwc))
        pwc_list.append(fragment_cluster)

        pwc.clear()
    ###############################

    n_event = sum([c.event_count() for c in pwc_list])
    n_pixels = sum([c.pixel_count(-1) for c in pwc_list])
    frac = n_pixels / sum([c.pixel_count(1) + c.pixel_count(-1) for c in pwc_list])

    logger.info("Supercluster done")
    if frac:
        logger.info("total  clusters|pixels|frac : %6d|%d|%f", n_event, n_pixels, frac)
    else:
        logger.info("total  clusters             : %6d", n_event)

    # restore skymap resolution
    network.restore_skymap(config, skyres)

    # timer
    timer_stop = time.perf_counter()
    logger.info("----------------------------------------")
    logger.info("Supercluster time: %.2f s", timer_stop - timer_start)
    logger.info("----------------------------------------")

    return pwc_list
