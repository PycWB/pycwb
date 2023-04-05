import copy
import time
import numpy as np
import ROOT
import logging
from pyburst.config import Config
from pyburst.conversions import convert_fragment_clusters_to_netcluster
from pyburst.modules.netcluster import select_clusters, copy_metadata
from pyburst.modules.netevent import Event
from pyburst.modules.catalog import add_events_to_catalog

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
    :return: the list of events
    :rtype: list[Event]
    """

    timer_start = time.perf_counter()

    # TODO: check if this is necessary
    # # set low-rate TD filters
    # for k in range(config.nRES):
    #     wdm_list[k].set_td_filter(config.TDSize, config.upTDF)
    net.setDelayIndex(config.TDRate)

    # load sparse table
    # logger.info("Loading sparse TF map ... ")
    # for n in range(config.nIFO):
    #     pD = net.getifo(n)
    #     pD.sclear()
    #     for i in range(config.nRES):
    #         pD.vSS.push_back(sparse_table_list[i][n])

    n_events = 0
    events = []
    for j in range(int(net.nLag)):
        cycle = net.wc_List[j].shift

        # add clusters to network for analysis
        pwc = net.getwc(j)
        pwc.cData.clear()
        pwc_temp = convert_fragment_clusters_to_netcluster(fragment_clusters[j])
        copy_metadata(pwc, pwc_temp)

        # print header
        logger.info("-------------------------------------------------------")
        logger.info("-> Processing %d clusters in lag=%d" % (len(fragment_clusters[j].clusters), cycle))
        logger.info("   ----------------------------------------------------")

        nmax = -1  # load all tdAmp
        npixels = 0  # total loaded pixels per lag
        nevents = 0  # total recontructed events per lag
        nselected_core_pixels = 0
        nrejected_weak_pixels = 0  # remove weak glitches
        nrejected_loud_pixels = 0  # remove loud glitches

        # loop over clusters to calculate likelihood
        for k in range(len(fragment_clusters[j].clusters)):
            # Todo: decouple for parallelization (or not)
            copy_metadata(pwc, pwc_temp)
            select_clusters(pwc, pwc_temp, k)

            event = _likelihood(job_id, config, net, j, pwc, k + 1, fragment_clusters[j])
            events.append(event)
        n_events += nevents

    # timer
    timer_end = time.perf_counter()
    logger.info("-------------------------------------------------------")
    logger.info("Total events: %d" % n_events)
    logger.info("Total time: %.2f s" % (timer_end - timer_start))
    logger.info("-------------------------------------------------------")

    return events


def _likelihood(job_id, config, net, lag, pwc, cluster_id, fragment_cluster):
    k = 0

    pwc.setcore(False, k+1)
    pwc.loadTDampSSE(net, 'a', config.BATCH, config.BATCH)  # attach TD amp to pixels

    ID = 0
    if net.pattern > 0:
        selected_core_pixels = net.likelihoodWP(config.search, lag, ID, ROOT.nullptr, config.Search)
    else:
        selected_core_pixels = net.likelihood2G(config.search, lag, ID, ROOT.nullptr)
    logger.info("Selected core pixels: %d" % selected_core_pixels)

    event = Event()
    event.output(net, k+1, 0)

    rejected_weak_pixels = 0
    rejected_loud_pixels = 0

    detected = (net.getwc(lag).sCuts[k] == -1)
    # Decoupling: detected = (net.getwc(j).sCuts[0] == -1)

    # print reconstructed event
    logger.info("   cluster-id|pixels: %5d|%d" % (cluster_id, int(pwc.size())))
    if detected:
        logger.info("\t -> SELECTED !!!")
        # print("-------------------------------------------------------")
        # print(event.dump())
        # print("-------------------------------------------------------")
    else:
        logger.info("\t <- rejected    ")

    # if detected:
    #     nevents += 1
    # npixels = pwc.size()
    # Decoupling: remove above line
    # TODO: sky statistics, likelihood distribution, null-hypothesis distribution, waveform, etc.

    try:
        output = event.json()
        with open(f'{config.outputDir}/event_{job_id}_{cluster_id}.json', 'w') as f:
            f.write(output)
        add_events_to_catalog(f"{config.outputDir}/catalog.json", [event.summary(job_id, cluster_id)])
    except Exception as e:
        logger.error(e)

    pwc.clean(1)

    return event


