import copy
import time
import pickle
import logging
import orjson
from pycwb.config import Config
from pycwb.modules.cwb_conversions import convert_fragment_clusters_to_netcluster, \
    convert_netcluster_to_fragment_clusters
from pycwb.types.network_cluster import FragmentCluster
from pycwb.types.network_event import Event

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

            cluster_id = k + 1
            event, cluster = _likelihood(config, network, j, cluster_id, fragment_cluster.dump_cluster(k))
            events.append(event)
            clusters.append(cluster)

            # skip saving skymap_statistic if cluster is already rejected
            if cluster.cluster_status != -1:
                skymap_statistics.append(None)
                continue

            # save skymap statistic
            skymap_statistic = {
                "nSensitivity": [],
                "nAlignment": [],
                "nLikelihood": [],
                "nNullEnergy": [],
                "nCorrEnergy": [],
                "nCorrelation": [],
                "nSkyStat": [],
                "nProbability": [],
                "nDisbalance": [],
                "nNetIndex": [],
                "nEllipticity": [],
                "nPolarisation": []
            }

            for key in skymap_statistic:
                var = getattr(network.net, key)
                L = var.size()
                skymap_statistic[key] = [var.get(l) for l in range(L)]

            skymap_statistics.append(skymap_statistic)

    n_events = len([c for c in clusters if c.cluster_status == -1])

    # timer
    timer_end = time.perf_counter()
    logger.info("-------------------------------------------------------")
    logger.info("Total events: %d" % n_events)
    logger.info("Total time: %.2f s" % (timer_end - timer_start))
    logger.info("-------------------------------------------------------")

    return events, clusters, skymap_statistics


def _likelihood(config, network, lag, cluster_id, fragment_cluster):
    # dumb variables
    k = 0

    ####################
    # cWB2G likelihood #
    ####################

    # set low-rate TD filters
    wdm_list = network.get_wdm_list()
    for wdm in wdm_list:
        wdm.setTDFilter(config.TDSize, config.upTDF)

    network.set_delay_index(config.TDRate)

    # sparse_table_list = sparse_table_from_fragment_clusters(config, tf_maps, [fragment_cluster])
    # for n in range(config.nIFO):
    #     det = network.get_ifo(n)
    #     det.sclear()
    #     for sparse_table in sparse_table_list:
    #         det.vSS.push_back(convert_sparse_series_to_sseries(sparse_table[n]))
    #     print("vss Size", det.vSS.size())

    # load cluster to network
    pwc = network.get_cluster(lag)
    pwc.cpf(convert_fragment_clusters_to_netcluster(fragment_cluster), False)

    pwc.setcore(False, k + 1)
    # attach TD amp to pixels, which will be used in likelihood calculation to pa(_vtd, v00), pA(_vTD, v90)
    # todo: check how this is implemented
    # todo: why this complex amplitude is not loaded before?
    pwc.loadTDampSSE(network.net, 'a', config.BATCH, config.BATCH)

    if network.pattern > 0:
        selected_core_pixels = network.likelihoodWP(config.search, lag, config.Search)
    else:
        selected_core_pixels = network.likelihood2G(config.search, lag)

    cluster = copy.deepcopy(convert_netcluster_to_fragment_clusters(network.get_cluster(lag))).clusters[0]

    event = Event()
    event.output(network.net, k + 1, 0)

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

    return event, cluster


# def save_likelihood_data(job_id, cluster_id, output_dir, event, cluster):
#     """
#     save event and cluster to file
#
#     :param job_id: job id
#     :type job_id: int
#     :param cluster_id: cluster id
#     :type cluster_id: int
#     :param output_dir: output directory
#     :type output_dir: str
#     :param event: event
#     :type event: Event
#     :param cluster: cluster
#     :type cluster: Cluster
#     """
#     # TODO: sky statistics, likelihood distribution, null-hypothesis distribution, waveform, etc.
#     try:
#         # save event to file
#         with open(f'{output_dir}/event_{job_id}_{cluster_id}.json', 'wb') as f:
#             f.write(orjson.dumps(event, option=orjson.OPT_SERIALIZE_NUMPY))
#         # save cluster to pickle
#         # with open(f'{output_dir}/cluster_{job_id}_{cluster_id}.pkl', 'wb') as f:
#         #     pickle.dump(cluster, f)
#         with open(f'{output_dir}/cluster_{job_id}_{cluster_id}.json', 'wb') as f:
#             f.write(orjson.dumps(cluster, option=orjson.OPT_SERIALIZE_NUMPY))
#
#     except Exception as e:
#         logger.error(e)
