import ROOT
from .pixel import convert_pixel_to_netpixel
from pycwb.types.network_cluster import Cluster


def convert_fragment_clusters_to_netcluster(fragment_clusters):
    """
    Convert cluster to netcluster

    :param cluster: cluster
    :type cluster: Cluster
    :return: netcluster
    :rtype: ROOT.netcluster
    """
    netcluster = ROOT.netcluster()

    netcluster.rate = fragment_clusters.rate
    netcluster.start = fragment_clusters.start
    netcluster.stop = fragment_clusters.stop
    netcluster.bpp = fragment_clusters.bpp
    netcluster.shift = fragment_clusters.shift
    netcluster.flow = fragment_clusters.f_low
    netcluster.fhigh = fragment_clusters.f_high
    netcluster.nPIX = fragment_clusters.n_pix
    netcluster.run = fragment_clusters.run
    netcluster.pair = fragment_clusters.pair
    netcluster.nSUB = fragment_clusters.subnet_threshold

    clusters = fragment_clusters.clusters
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
    netcluster.cData = [convert_cluster_meta_to_cData(cluster.cluster_meta)
                        for cluster in clusters]
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


def convert_cluster_meta_to_cData(cluster_meta):
    c_data = ROOT.clusterdata()
    c_data.energy = cluster_meta.energy
    c_data.enrgsky = cluster_meta.energy_sky
    c_data.likenet = cluster_meta.like_net
    c_data.netecor = cluster_meta.net_ecor
    c_data.normcor = cluster_meta.norm_cor
    c_data.netnull = cluster_meta.net_null
    c_data.netED = cluster_meta.net_ed
    c_data.Gnoise = cluster_meta.g_noise
    c_data.likesky = cluster_meta.like_sky
    c_data.skycc = cluster_meta.sky_cc
    c_data.netcc = cluster_meta.net_cc
    c_data.skyChi2 = cluster_meta.sky_chi2
    c_data.subnet = cluster_meta.sub_net
    c_data.SUBNET = cluster_meta.sub_net2
    c_data.skyStat = cluster_meta.sky_stat
    c_data.netRHO = cluster_meta.net_rho
    c_data.netrho = cluster_meta.net_rho2
    c_data.theta = cluster_meta.theta
    c_data.phi = cluster_meta.phi
    c_data.iota = cluster_meta.iota
    c_data.psi = cluster_meta.psi
    c_data.ellipticity = cluster_meta.ellipticity
    c_data.cTime = cluster_meta.c_time
    c_data.cFreq = cluster_meta.c_freq
    c_data.gNET = cluster_meta.g_net
    c_data.aNET = cluster_meta.a_net
    c_data.iNET = cluster_meta.i_net
    c_data.norm = cluster_meta.norm
    c_data.nDoF = cluster_meta.ndof
    c_data.tmrgr = cluster_meta.tmrgr
    c_data.tmrgrerr = cluster_meta.tmrgrerr
    c_data.mchirp = cluster_meta.mchirp
    c_data.mchirperr = cluster_meta.mchirperr
    c_data.chi2chirp = cluster_meta.chi2chirp
    c_data.chirpEfrac = cluster_meta.chirp_efrac
    c_data.chirpPfrac = cluster_meta.chirp_pfrac
    c_data.chirpEllip = cluster_meta.chirp_ellip
    c_data.skySize = cluster_meta.sky_size
    c_data.skyIndex = cluster_meta.sky_index
    c_data.chirp = cluster_meta.chirp
    c_data.mchpdf = cluster_meta.mchpdf
    return c_data


