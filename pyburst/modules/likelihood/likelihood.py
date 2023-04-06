import copy, os
import time
import pickle
import ROOT
import logging
from pyburst.config import Config
from pyburst.conversions import convert_fragment_clusters_to_netcluster
from pyburst.modules.netcluster import select_clusters, copy_metadata
from pyburst.modules.netevent import Event
from pyburst.modules.catalog import add_events_to_catalog
from pyburst.types import FragmentCluster

logger = logging.getLogger(__name__)


def likelihood(job_id, config, net, fragment_clusters):
    """
    calculate likelihood

    :param config: user configuration
    :type config: Config
    :param net: network
    :type net: ROOT.network
    :param fragment_clusters: list of cluster
    :type fragment_clusters: list[FragmentCluster]
    :return: the list of events and clusters
    :rtype: list[Event], list[Cluster]
    """

    timer_start = time.perf_counter()

    net.setDelayIndex(config.TDRate)

    n_events = 0
    events = []
    clusters = []
    for j in range(int(net.nLag)):
        cycle = net.wc_List[j].shift

        # print header
        logger.info("-------------------------------------------------------")
        logger.info("-> Processing %d clusters in lag=%d" % (len(fragment_clusters[j].clusters), cycle))
        logger.info("   ----------------------------------------------------")

        # loop over clusters to calculate likelihood
        for k in range(len(fragment_clusters[j].clusters)):
            event, cluster = _likelihood(job_id, config, net, j, k + 1, fragment_clusters[j])
            events.append(event)
            clusters.append(cluster)

    n_events = len([c for c in clusters if c.cluster_status == -1])

    # timer
    timer_end = time.perf_counter()
    logger.info("-------------------------------------------------------")
    logger.info("Total events: %d" % n_events)
    logger.info("Total time: %.2f s" % (timer_end - timer_start))
    logger.info("-------------------------------------------------------")

    return events, clusters


def _likelihood(job_id, config, net, lag, cluster_id, fragment_cluster):
    k = 0

    temp_cluster = copy.copy(fragment_cluster)
    temp_cluster.clusters = [fragment_cluster.clusters[cluster_id-1]]

    ####################
    # cWB2G likelihood #
    ####################
    pwc = net.getwc(lag)
    pwc.cpf(convert_fragment_clusters_to_netcluster(temp_cluster), False)

    pwc.setcore(False, k + 1)
    pwc.loadTDampSSE(net, 'a', config.BATCH, config.BATCH)  # attach TD amp to pixels

    ID = 0
    if net.pattern > 0:
        selected_core_pixels = net.likelihoodWP(config.search, lag, ID, ROOT.nullptr, config.Search)
    else:
        selected_core_pixels = net.likelihood2G(config.search, lag, ID, ROOT.nullptr)
    cluster = copy.deepcopy(FragmentCluster().from_netcluster(net.getwc(lag))).clusters[k]

    event = Event()
    event.output(net, k + 1, 0)

    pwc.clean(1)

    ####################

    logger.info("Selected core pixels: %d" % selected_core_pixels)

    detected = cluster.cluster_status == -1

    # print reconstructed event
    logger.info("   cluster-id|pixels: %5d|%d" % (cluster_id, len(cluster.pixels)))
    if detected:
        logger.info("\t -> SELECTED !!!")
    else:
        logger.info("\t <- rejected    ")

    # TODO: sky statistics, likelihood distribution, null-hypothesis distribution, waveform, etc.

    try:
        # save event to file
        with open(f'{config.outputDir}/event_{job_id}_{cluster_id}.json', 'w') as f:
            f.write(event.json())
        # save cluster to pickle
        with open(f'{config.outputDir}/cluster_{job_id}_{cluster_id}.pkl', 'wb') as f:
            pickle.dump(cluster, f)

        # save event to catalog if file exists
        if os.path.exists(f"{config.outputDir}/catalog.json"):
            add_events_to_catalog(f"{config.outputDir}/catalog.json", [event.summary(job_id, cluster_id)])
        else:
            logger.warning("Catalog file does not exist. Event will not be saved to catalog.")
    except Exception as e:
        logger.error(e)

    return event, cluster
