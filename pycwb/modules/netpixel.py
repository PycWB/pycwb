def get_time_frequency(cluster, pix):
    return cluster.start + pix.time / (pix.rate * pix.layers), pix.frequency * pix.rate
