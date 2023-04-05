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


def supercluster(config, net, wdm_list, fragment_clusters, sparse_table_list):
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
    :param net: network
    :type net: ROOT.network
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

    # add wavelets to network
    for wdm in wdm_list:
        net.add(wdm.wavelet)

    # decrease skymap resolution to improve subNetCut performances
    skyres = MIN_SKYRES_HEALPIX if config.healpix > MIN_SKYRES_HEALPIX else 0
    if skyres > 0:
        update_sky_map(config, net, skyres)
        update_sky_mask(config, net, skyres)

    hot = []
    for n in range(config.nIFO):
        hot.append(net.getifo(n).getHoT())
    # set low-rate TD filters
    for wdm in wdm_list:
        wdm.set_td_filter(config.TDSize, 1)

    # read sparse map to detector
    for n in range(config.nIFO):
        det = net.getifo(n)
        det.sclear()
        for sparse_table in sparse_table_list:
            det.vSS.push_back(convert_sparse_series_to_sseries(sparse_table[n]))

    # merge cluster
    cluster = copy.deepcopy(fragment_clusters[0])
    if len(fragment_clusters) > 1:
        for fragment_cluster in fragment_clusters[1:]:
            cluster.clusters += fragment_cluster.clusters

    # convert to netcluster
    cluster = convert_fragment_clusters_to_netcluster(cluster)

    nevt = 0
    nnn = 0
    mmm = 0
    pwc_list = []
    for j in range(int(net.nLag)):
        # cycle = cfg.simulation ? ifactor : Long_t(NET.wc_List[j].shift);
        cycle = int(net.wc_List[j].shift)
        cycle_name = f"lag={cycle}"

        logger.info("-> Processing %s ...", cycle_name)
        logger.info("   --------------------------------------------------")
        logger.info("    coher clusters|pixels      : %6d|%d", cluster.esize(0), cluster.psize(0))

        if config.l_high == config.l_low:
            net.pair = False
        if net.pattern != 0:
            net.pair = False

        cluster.supercluster('L',net.e2or,config.TFgap,False)
        logger.info("    super clusters|pixels      : %6d|%d", cluster.esize(0), cluster.psize(0))

        # defragmentation for pattern != 0
        if net.pattern != 0:
            cluster.defragment(config.Tgap, config.Fgap)
            logger.info("   defrag clusters|pixels      : %6d|%d", cluster.esize(0), cluster.psize(0))

        # copy selected clusters to network
        pwc = net.getwc(j)
        pwc.cpf(cluster, False)

        # apply subNetCut() only for pattern=0 || cfg.subnet>0 || cfg.subcut>0 || cfg.subnorm>0 || cfg.subrho>=0
        if net.pattern == 0 or config.subnet > 0 or config.subcut > 0 or config.subnorm > 0 or config.subrho >= 0:
            if config.subacor > 0:
                # set Acore for subNetCuts
                net.acor = config.subacor
            if config.subrho > 0:
                # set netRHO for subNetCuts
                net.netRHO = config.subrho
            net.setDelayIndex(hot[0].rate())
            pwc.setcore(False)
            psel = 0
            while True:
                count = pwc.loadTDampSSE(net, 'a', config.BATCH, config.LOUD)
                # FIXME: O4 code have addtional config.subnorm
                psel += net.subNetCut(j, config.subnet, config.subcut, config.subnorm, ROOT.nullptr)
                # ptot = cluster.psize(1) + cluster.psize(-1)
                # pfrac = ptot / psel if ptot > 0 else 0
                if count < 10000:
                    break
            logger.info("   subnet clusters|pixels      : %6d|%d", net.events(), pwc.psize(-1))
            if config.subacor > 0:
                # restore Acore
                net.acor = config.Acore
            if config.subrho > 0:
                # restore netRHO
                net.netRHO = config.netRHO

        if net.pattern == 0:
            pwc.defragment(config.Tgap, config.Fgap)
            logger.info("   defrag clusters|pixels      : %6d|%d", cluster.esize(0), cluster.psize(0))

        nevt += net.events()
        nnn += pwc.psize(-1)
        mmm += pwc.psize(1) + pwc.psize(-1)

        # convert to FragmentCluster and append to list
        pwc_list.append(copy.deepcopy(FragmentCluster().from_netcluster(pwc)))
        pwc.clear()

    logger.info("Supercluster done")
    if mmm:
        logger.info("total  clusters|pixels|frac : %6d|%d|%f", nevt, nnn, nnn / mmm)
    else:
        logger.info("total  clusters             : %6d", nevt)

    # restore skymap resolution
    restore_skymap(config, net, skyres)

    # timer
    timer_stop = time.perf_counter()
    logger.info("----------------------------------------")
    logger.info("Supercluster time: %.2f s", timer_stop - timer_start)
    logger.info("----------------------------------------")

    return pwc_list