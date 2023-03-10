class NetworkCluster:
    __slots__ = ['rate', 'start', 'stop', 'bpp', 'shift', 'f_low', 'f_high', 'n_pix', 'run', 'pair',
                 'subnet_threshold', 'pixel_list', 'cluster_data', 'selection_cuts', 'cluster_list',
                 'cluster_rate', 'cluster_time', 'cluster_freq', 'sky_area', 'sky_pixel_map',
                 'sky_pixel_index', 'sky_time_delay']

    def __init__(self, rate, start, stop, bpp, shift, f_low, f_high, n_pix, run, pair, n_sub,
                 p_list, c_data, s_cuts, c_list, c_rate, c_time, c_freq, s_area, p_map, p_ind, n_toff):
        #: original Time series rate
        self.rate = rate
        #: interval start GPS time
        self.start = start
        #: interval stop  GPS time
        self.stop = stop
        #: black pixel probability
        self.bpp = bpp
        #: time shift
        self.shift = shift
        #: low frequency boundary
        self.f_low = f_low
        #: high frequency boundary
        self.f_high = f_high
        #: minimum number of pixels at all resolutions
        self.n_pix = n_pix
        #: run ID
        self.run = run
        #: true - 2 resolutions, false - 1 resolution
        self.pair = pair
        #: subnetwork threshold for a single network pixel
        self.subnet_threshold = n_sub
        #: pixel list
        self.pixel_list = p_list
        #: cluster metadata
        self.cluster_data = c_data
        #: cluster selection flags (cuts)
        self.selection_cuts = s_cuts
        #: cluster list defined by vector of pList references
        self.cluster_list = c_list
        #: cluster type defined by rate
        self.cluster_rate = c_rate
        #: supercluster central time
        self.cluster_time = c_time
        #: supercluster central frequency
        self.cluster_freq = c_freq
        #: sky error regions
        self.sky_area = s_area
        #: sky pixel map
        self.sky_pixel_map = p_map
        #: sky pixel index
        self.sky_pixel_index = p_ind
        #: sky time delay configuration for waveform backward correction
        self.sky_time_delay = n_toff
