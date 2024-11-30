import logging, time
from multiprocessing import Pool

from pycwb.modules.multi_resolution_wdm import create_wdm_for_level
from pycwb.types.sparse_series import SparseTimeFrequencySeries

logger = logging.getLogger(__name__)


def sparse_table_from_fragment_clusters(config, tf_maps, fragment_clusters, parallel=False):
    """Create sparse tables from fragment clusters

    :param config: config object
    :type config: Config
    :param tf_maps: time-frequency maps
    :type tf_maps: list[TimeFrequencySeries]
    :param wdm_list: list of wavelet-domain models
    :type wdm_list: list[WDM]
    :param fragment_clusters: fragment clusters
    :type fragment_clusters: list[FragmentCluster]
    :return: sparse tables
    :rtype: list[list[SparseTimeFrequencySeries]]
    """
    timer_start = time.perf_counter()

    if parallel:
        logger.info("Start sparse series in parallel")
        with Pool(processes=min(config.nproc, config.nRES)) as pool:
            sparse_tables = pool.map(_sparse_table_from_fragment_cluster,
                                     [(config, tf_maps, i, fragment_cluster)
                                      for i, fragment_cluster in enumerate(fragment_clusters)])
    else:
        sparse_tables = []

        for i, fragment_cluster in enumerate(fragment_clusters):
            sparse_tables.append(_sparse_table_from_fragment_cluster((config, tf_maps, i, fragment_cluster)))
        # sparse_tables = list(map(_sparse_table_from_fragment_cluster,
        #                          [(config, tf_maps, i, fragment_cluster)
        #                           for i, fragment_cluster in enumerate(fragment_clusters)]))

    timer_stop = time.perf_counter()
    logger.info("----------------------------------------")
    logger.info("Sparse series time: %.2f s", timer_stop - timer_start)
    logger.info("----------------------------------------")

    return sparse_tables


def _sparse_table_from_fragment_cluster(args):
    config, tf_maps, i, fragment_cluster = args
    wdm = create_wdm_for_level(config, config.WDM_level[i])

    return [
        SparseTimeFrequencySeries().from_fragment_cluster(wdm, tf_maps[n], fragment_cluster,
                                                          config.TDSize, config.max_delay, n)
        for n in range(config.nIFO)]
