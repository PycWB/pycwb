def get_time_frequency(cluster, pix):
    """
    Get time and frequency of pixel in cluster

    :param cluster: cluster for the pixel
    :type cluster: ROOT.netcluster
    :param pix: pixel
    :type pix: ROOT.netpixel
    :return: (time, frequency)
    :rtype: tuple
    """
    return cluster.start + pix.time / (pix.rate * pix.layers), pix.frequency * pix.rate
