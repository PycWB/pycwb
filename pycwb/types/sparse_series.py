import numpy as np
from scipy.sparse import coo_array


class SparseTimeFrequencySeries:
    """SparseSeries class

    :param core: bool, whether to use core functions
    """

    def __init__(self, core=None, wavelet=None,  # sparse_lookup=None, sparse_type=None,
                 sparse_index=None, sparse_map_00=None, sparse_map_90=None,
                 rate=0, w_rate=0, start=0, stop=0, edge=0,
                 time_halo=0, layer_halo=0, net_delay=0):
        self.wavelet = wavelet
        self.core = core
        self.sparse_table_00 = None
        self.sparse_table_90 = None
        if sparse_index and sparse_map_00 and sparse_map_90:
            self.set_sparse_table(sparse_index, sparse_map_00, sparse_map_90)

        self.rate = rate
        self.w_rate = w_rate
        self.start = start
        self.stop = stop
        self.edge = edge

        self.time_halo = time_halo
        self.layer_halo = layer_halo
        self.net_delay = net_delay

    def from_fragment_cluster(self, wdm, tf_map, fragment_clusters, td_size, m_tau, ifo_id):
        wdm.set_td_filter(td_size, 1)
        ws = tf_map.copy()
        ws.wavelet = wdm
        ws.forward()

        self.set_map(ws)
        self.set_halo(m_tau)
        if not isinstance(fragment_clusters, list):
            fragment_clusters = [fragment_clusters]
        for fragment_cluster in fragment_clusters:
            for cluster in fragment_cluster.clusters:
                self.add_core(ifo_id, cluster)
        self.update_sparse_table()
        return self

    def resize(self):
        """Resize the sparse map

        :param size: new size
        :type size: int
        """
        self.core = []
        self.sparse_table_00 = None
        self.sparse_table_90 = None

    def set_map(self, tf_map, reset=True):
        """Set the map

        :param tf_map: time-frequency map
        :type tf_map: TimeFrequencySeries
        :param reset: reset the sparse map
        :type reset: bool (default: True)
        """
        if reset:
            self.resize()

        if tf_map and reset:
            if self.wavelet:
                self.wavelet.release()

            self.wavelet = tf_map.wavelet.lightweight_dump()
            self.wavelet.allocate(n=tf_map.wavelet.nWWS, data=tf_map.wavelet.pWWS)
            self.time_halo = tf_map.wavelet.time_delay_filter_size

            self.rate = tf_map.data.sample_rate
            self.w_rate = tf_map.w_rate
            self.start = tf_map.start
            self.stop = tf_map.stop
            self.edge = tf_map.edge
        elif tf_map:
            print("Not implemented yet")

    def set_halo(self, max_tau=0.042, layer_halo=1, time_halo=-1):
        """Set the halo

        For each core pixels (if layerHalo = 1) the following pixels are saved

            core      = '.'
            extraHalo = '++++++++'
            timeHalo  = 'xxxx'

            ++++++++xxxx xxxx++++++++
            ++++++++xxxx.xxxx++++++++
            ++++++++xxxx xxxx++++++++

        :param max_tau: delay time (sec) is stored in net_Delay used to compute extraHalo = net_Delay*sparseRate
        :type max_tau: float
        :param layer_halo: number of layers above and below each core pixel is stored in layerHalo the total number of layers in the halo is 2*layerHalo+1
        :type layer_halo: int
        :param time_halo: number of slice on the right and on the left each core pixel is stored in timeHalo the total number of pixels on the time axis is 2*(timeHalo+extraHalo)+1 The default value is -1 : the value is automatically selected from the associated TF map with WDM::getTDFsize() method.
        :type time_halo: int
        """
        self.resize()
        self.layer_halo = layer_halo
        self.net_delay = max_tau
        if time_halo >= 0:
            self.time_halo = time_halo

    def add_core(self, ifo_id, cluster):
        """ Add core pixels of the detector ID=ifoID contained in the pwc netcluster with index ID

        :param ifo_id: detector ID
        :type ifo_id: int
        :param cluster: netcluster
        :type cluster: Cluster
        """

        r = int(self.w_rate + 0.1)

        # skip rejected clusters
        if cluster.cluster_status == 1:
            return False

        # loop over pixels
        for pixel in cluster.pixels:
            # skip pixel with bad rate, TODO: why the rate can be bad?
            if int(pixel.rate + 0.01) != r:
                continue

            # store pixel index
            self.core.append(pixel.data[ifo_id].index)

    def update_sparse_table(self):
        """
        Use the core pixels and halo parameters to update the sparse maps with core+halo pixels

        :return:
        """
        if self.time_halo == 0 and self.layer_halo == 0:
            return

        extra_halo = int(self.net_delay * self.w_rate) + 8  # init extra halo : WARNING value 8 ad hoc - to be fixed
        h_slice = self.time_halo + extra_halo  # halo slices
        layer_halo = self.layer_halo  # halo layers

        cluster = []
        n_layer = self.wavelet.max_level + 1  # number of WDM layers
        n_slice = self.wavelet.size_at_zero_layer  # number of samples in wavelet layer

        # start_time = time.perf_counter()
        for core_pixel in self.core:
            # i,j index of the core pixel
            t = core_pixel // n_layer  # slice
            l = core_pixel % n_layer  # layer

            # Calculate the starting and ending indices for the time and layer axes
            start_t = max(0, t - h_slice)
            start_l = max(0, l - layer_halo)
            end_t = min(n_slice - 1, t + h_slice)
            end_l = min(n_layer - 1, l + layer_halo)

            # Calculate the block indices in 1D array
            cluster += [(ll, tt) for ll in range(start_l, end_l + 1) for tt in range(start_t, end_t + 1)]

        # remove duplicates
        cluster = list(dict.fromkeys(cluster))
        cluster.sort()

        # get the sparse maps
        # sparse_map_00 = [self.wavelet.get_map_00(ll * n_layer + tt) for tt, ll in cluster]
        # sparse_map_90 = [self.wavelet.get_map_90(ll * n_layer + tt) for tt, ll in cluster]
        sparse_map_00 = []
        sparse_map_90 = []
        for tt, ll in cluster:
            sparse_map_00.append(self.wavelet.get_map_00(ll * n_layer + tt))
            sparse_map_90.append(self.wavelet.get_map_90(ll * n_layer + tt))

        self.set_sparse_table(cluster, sparse_map_00, sparse_map_90)

    def set_sparse_table(self, sparse_index, sparse_map_00, sparse_map_90):
        n_layer = self.wavelet.max_level + 1  # number of WDM layers
        n_slice = self.wavelet.size_at_zero_layer  # number of samples in wavelet layer

        # cluster = np.array(sparse_index)
        # time = cluster // n_layer
        # layer = cluster % n_layer
        # list comprehension is faster than zip
        layer = [i[0] for i in sparse_index]
        time = [i[1] for i in sparse_index]

        # create sparse table
        self.sparse_table_00 = coo_array((sparse_map_00, (layer, time)), shape=(n_layer, n_slice))
        self.sparse_table_90 = coo_array((sparse_map_90, (layer, time)), shape=(n_layer, n_slice))

    @property
    def sparse_lookup(self):
        """Get the sparse lookup
        """
        sparse_lookup = []
        n = 0
        n_layer = self.sparse_table_00.shape[0]
        for i in range(n_layer + 1):
            sparse_lookup.append(n)
            n += np.count_nonzero(self.sparse_table_00.row == i)
        return sparse_lookup

    @property
    def sparse_map_00(self):
        """Get the sparse map_00
        """
        return self.sparse_table_00.data

    @property
    def sparse_map_90(self):
        """Get the sparse map_90
        """
        return self.sparse_table_90.data

    @property
    def sparse_index(self):
        """Get the sparse index
        """
        n_layer = self.sparse_table_00.shape[0]
        return self.sparse_table_00.col * n_layer + self.sparse_table_00.row
