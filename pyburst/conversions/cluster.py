import ROOT
from .pixel import convert_pixel_to_netpixel
from pyburst.types import Cluster


def convert_netcluster_to_cluster(c_cluster):
    cluster_list = []
    for c_id, pixel_ids in enumerate(c_cluster.cList):
        cluster = Cluster().from_netcluster(c_cluster, c_id)
        cluster_list.append(cluster)
    return cluster_list


def convert_cluster_to_netcluster(clusters):
    """
    Convert cluster to netcluster

    :param cluster: cluster
    :type cluster: Cluster
    :return: netcluster
    :rtype: ROOT.netcluster
    """
    netcluster = ROOT.netcluster()

    # derive cList
    i = 0
    for cluster in clusters:
        n = len(cluster.pixels)
        netcluster.cList.push_back(list(range(i, i + n)))
        i += n

    # add pixels
    netcluster.pList = [
        convert_pixel_to_netpixel(pixel, c_index + 1)
        for c_index, cluster in enumerate(clusters)
        for pixel in cluster.pixels
    ]

    # add others
    netcluster.cData = [cluster.cluster_meta for cluster in clusters]
    netcluster.sCuts = [cluster.cluster_status for cluster in clusters]
    netcluster.cTime = [cluster.cluster_time for cluster in clusters]
    netcluster.cFreq = [cluster.cluster_freq for cluster in clusters]
    for cluster in clusters:
        netcluster.cRate.push_back(cluster.cluster_rate)
        netcluster.sArea.push_back(cluster.sky_area)
        netcluster.p_Map.push_back(cluster.sky_pixel_map)
        netcluster.p_Ind.push_back(cluster.sky_pixel_index)
        netcluster.nTofF.push_back(cluster.sky_time_delay)

    return netcluster
