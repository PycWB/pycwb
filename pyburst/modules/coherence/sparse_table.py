import logging, time
from multiprocessing import Pool

from pyburst.types.sparse_series import SparseTimeFrequencySeries

logger = logging.getLogger(__name__)


def sparse_table_from_fragment_clusters(config, m_tau, tf_maps, wdm_list, fragment_clusters):
    """Create sparse tables from fragment clusters

    :param config: config object
    :type config: Config
    :param m_tau: m_tau
    :type m_tau: float
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

    # sparse_tables = []
    with Pool(processes=min(config.nproc, config.nRES)) as pool:
        sparse_tables = pool.starmap(_sparse_table_from_fragment_cluster,
                                     [(config, m_tau, tf_maps, wdm_list[i], fragment_cluster)
                                      for i, fragment_cluster in enumerate(fragment_clusters)])
    # for i, fragment_cluster in enumerate(fragment_clusters):
    #     sparse_tables.append([
    #         SparseTimeFrequencySeries().from_fragment_cluster(wdm_list[i], tf_maps[n], fragment_cluster,
    #                                                           config.TDSize, m_tau, n)
    #         for n in range(config.nIFO)])

    timer_stop = time.perf_counter()
    logger.info("----------------------------------------")
    logger.info("Sparse series time: %.2f s", timer_stop - timer_start)
    logger.info("----------------------------------------")

    return sparse_tables


def _sparse_table_from_fragment_cluster(config, m_tau, tf_maps, wdm, fragment_cluster):
    return [
        SparseTimeFrequencySeries().from_fragment_cluster(wdm, tf_maps[n], fragment_cluster,
                                                          config.TDSize, m_tau, n)
        for n in range(config.nIFO)]
