def append_cluster(cluster, c, n_max):
    """
    Append cluster to another cluster

    :param cluster: target cluster
    :type cluster: ROOT.netcluster
    :param c: source cluster
    :type c: ROOT.netcluster
    :param n_max: if -2, skip cleaning
    :type n_max: int
    :return: None
    """
    skip = n_max == -2

    for c_id, clist in enumerate(c.cList):
        if not skip:
            for i in cluster.pList:
                i.clean()
        new_cluster_id = cluster.cList.size() + 1
        for pixel in c.pList[clist[0]: clist[-1] + 1]:
            pixel.clusterID = new_cluster_id
            cluster.pList.push_back(pixel)
            # TODO: orate and crate
        cluster.sCuts.push_back(0)
        cluster.cList.push_back(clist)
        cluster.cRate.push_back(c.cRate[c_id])
        cluster.cTime.push_back(c.cTime[c_id])
        cluster.cFreq.push_back(c.cFreq[c_id])
        cluster.sArea.push_back(c.sArea[c_id])
        cluster.p_Map.push_back(c.p_Map[c_id])
        cluster.nTofF.push_back(c.nTofF[c_id])
        cluster.p_Ind.push_back(c.p_Ind[c_id])
        if not skip:
            cluster.cData.push_back(c.cData[c_id])


def copy_metadata(cluster, c):
    """
    Copy metadata from c to cluster

    :param cluster: target cluster
    :type cluster: ROOT.netcluster
    :param c: source cluster
    :type c: ROOT.netcluster
    :return: None
    """
    cluster.cpf(c, False, 0)
    cluster.clear()


def select_clusters(cluster, c, cluster_id):
    """
    Select cluster by cluster_id from c and append to cluster

    :param cluster: target cluster
    :type cluster: ROOT.netcluster
    :param c: source cluster
    :type c: ROOT.netcluster
    :param cluster_id: cluster id
    :type cluster_id: int
    :return: None
    """
    # Remember to copy cluster metadata (cData)
    new_cluster_id = cluster.cList.size() + 1
    new_pixel_id = cluster.pList.size()
    clist = c.cList[cluster_id]
    for pixel in c.pList[clist[0]: clist[-1] + 1]:
        pixel.clusterID = new_cluster_id
        cluster.pList.push_back(pixel)
    cluster.sCuts.push_back(0)
    cluster.cList.push_back([new_pixel_id + i for i in range(clist.size())])
    cluster.cRate.push_back(c.cRate[cluster_id])
    cluster.cTime.push_back(c.cTime[cluster_id])
    cluster.cFreq.push_back(c.cFreq[cluster_id])
    cluster.sArea.push_back(c.sArea[cluster_id])
    cluster.p_Map.push_back(c.p_Map[cluster_id])
    cluster.nTofF.push_back(c.nTofF[cluster_id])
    cluster.p_Ind.push_back(c.p_Ind[cluster_id])
    cluster.cData.push_back(c.cData[cluster_id])
