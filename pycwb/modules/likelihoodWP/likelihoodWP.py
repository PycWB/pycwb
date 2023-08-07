import copy
import time
import logging
import numpy as np
import ROOT
import healpy as hp
from pycwb.types.network_cluster import FragmentCluster
from pycwb.types.network_event import Event

from pycwb.modules.cwb_conversions import convert_fragment_clusters_to_netcluster

logger = logging.getLogger(__name__)


def likelihood(config, network, fragment_clusters):
    """
    calculate likelihood

    :param config: user configuration
    :type config: Config
    :param network: network
    :type network: Network
    :param fragment_clusters: list of cluster
    :type fragment_clusters: list[FragmentCluster]
    :return: the list of events and clusters
    :rtype: list[Event], list[Cluster]
    """

    timer_start = time.perf_counter()

    events = []
    clusters = []

    skymap_statistics = []
    for j, fragment_cluster in enumerate(fragment_clusters):
        cycle = fragment_cluster.shift

        # print header
        logger.info("-------------------------------------------------------")
        logger.info("-> Processing %d clusters in lag=%d" % (len(fragment_clusters[j].clusters), cycle))
        logger.info("   ----------------------------------------------------")

        # loop over clusters to calculate likelihood
        for k, selected_cluster in enumerate(fragment_cluster.clusters):
            # skip if cluster is already rejected
            if selected_cluster.cluster_status > 0:
                continue

            event, cluster, skymap_statistic = _likelihoodWP(config, network, fragment_cluster.dump_cluster(k))
            events.append(event)
            clusters.append(cluster)

            # skip saving skymap_statistic if cluster is already rejected
            if cluster.cluster_status != -1:
                skymap_statistics.append(None)
                continue

            skymap_statistics.append(skymap_statistic)

    n_events = len([c for c in clusters if c.cluster_status == -1])

    # timer
    timer_end = time.perf_counter()
    logger.info("-------------------------------------------------------")
    logger.info("Total events: %d" % n_events)
    logger.info("Total time: %.2f s" % (timer_end - timer_start))
    logger.info("-------------------------------------------------------")

    return events, clusters, skymap_statistics


def _likelihoodWP(config, network, fragment_cluster):
    # set low-rate TD filters
    wdm_list = network.get_wdm_list()
    for wdm in wdm_list:
        wdm.setTDFilter(config.TDSize, config.upTDF)

    network.set_delay_index(config.TDRate)
    pwc = ROOT.netcluster()
    pwc.cpf(convert_fragment_clusters_to_netcluster(fragment_cluster), False)
    pwc.setcore(False, 1)
    pwc.loadTDampSSE(network.net, 'a', config.BATCH, config.BATCH)  # attach TD amp to pixels

    # skymap* nSkyStat, skymap* nSensitivity, skymap* nAlignment, skymap* nDisbalance,
    # skymap* nLikelihood, skymap* nNullEnergy, skymap* nCorrEnergy, skymap* nCorrelation,
    # skymap* nEllipticity, skymap* nPolarisation, skymap* nNetIndex, skymap* nAntenaPrior,
    # skymap* nProbability,
    keys = ['nSkyStat', 'nSensitivity', 'nAlignment', 'nDisbalance', 'nLikelihood', 'nNullEnergy', 'nCorrEnergy',
            'nCorrelation', 'nEllipticity', 'nPolarisation', 'nNetIndex', 'nAntenaPrior', 'nProbability']
    skymaps = {key: ROOT.skymap(7) for key in keys}
    ifos = [network.get_ifo(i) for i in range(len(config.ifo))]
    skymap_index_size = hp.nside2npix(2 ** 7)
    print("WDM size = ", network.net.wdmMRA.size(0))
    # double netCC, bool EFEC, double precision, double gamma, bool optim, double netRHO, double delta, double acor,
    count = ROOT.likelihoodWP(pwc, skymaps['nSkyStat'], skymaps['nSensitivity'], skymaps['nAlignment'],
                              skymaps['nDisbalance'], skymaps['nLikelihood'], skymaps['nNullEnergy'],
                              skymaps['nCorrEnergy'], skymaps['nCorrelation'], skymaps['nEllipticity'],
                              skymaps['nPolarisation'], skymaps['nNetIndex'], skymaps['nAntenaPrior'],
                              skymaps['nProbability'], len(ifos), ifos, skymap_index_size,
                              network.net.skyMask.data, network.net.skyMaskCC.data, network.net.wdmMRA,
                              network.net.netCC, network.net.EFEC, network.net.precision, network.net.gamma,
                              network.net.optim, network.net.netRHO, network.net.delta, network.net.acor,
                              config.search, 1, config.Search)

    cluster = copy.deepcopy(FragmentCluster().from_netcluster(pwc)).clusters[0]

    # TODO: dump event without network
    event = Event()

    for key in skymaps.keys():
        L = skymaps[key].size()
        skymaps[key] = [skymaps[key].get(i) for i in range(L)]

    return event, cluster, skymaps


def set_plus_regulator(acor, gamma, delta, nIFO):
    netwoek_energy_threshold = 2 * acor * acor * nIFO  # network energy threshold in the sky loop
    gamma_regulator = gamma * gamma * 2. / 3.  # gamma regulator for x componet
    delta_regulator = abs(delta)
    if delta_regulator > 1:
        delta_regulator = 1

    return delta_regulator * np.sqrt(2)


def set_cross_regulator(acor, gamma, delta, nIFO):
    pass
