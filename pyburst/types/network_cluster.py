from pyburst.types.network_pixel import Pixel


class FragmentCluster:
    """
    Class for clusters found in data fragment

    :param rate: original Time series rate
    :type rate: float
    :param start: interval start GPS time
    :type start: float
    :param stop: interval stop  GPS time
    :type stop: float
    :param bpp: black pixel probability
    :type bpp: float
    :param shift: time shift
    :type shift: float
    :param f_low: low frequency boundary
    :type f_low: float
    :param f_high: high frequency boundary
    :type f_high: float
    :param n_pix: minimum number of pixels at all resolutions
    :type n_pix: int
    :param run: run ID
    :type run: int
    :param pair: true - 2 resolutions, false - 1 resolution
    :type pair: bool
    :param n_sub: subnetwork threshold for a single network pixel
    :type n_sub: int
    :param clusters: cluster list
    :type clusters: list[Cluster]
    """
    __slots__ = ['rate', 'start', 'stop', 'bpp', 'shift', 'f_low', 'f_high', 'n_pix', 'run', 'pair',
                 'subnet_threshold', 'clusters']

    def __init__(self, rate=None, start=None, stop=None, bpp=None, shift=None, f_low=None, f_high=None,
                    n_pix=None, run=None, pair=None, n_sub=None, clusters=None):
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
        #: cluster list
        self.clusters = clusters

    def from_netcluster(self, c_cluster):
        """
        Create FragmentCluster from netcluster

        :param c_cluster: netcluster
        :type c_cluster: ROOT.netcluster
        """
        self.rate = c_cluster.rate
        self.start = c_cluster.start
        self.stop = c_cluster.stop
        self.bpp = c_cluster.bpp
        self.shift = c_cluster.shift
        self.f_low = c_cluster.flow
        self.f_high = c_cluster.fhigh
        self.n_pix = c_cluster.nPIX
        self.run = c_cluster.run
        self.pair = c_cluster.pair
        self.subnet_threshold = c_cluster.nSUB

        cluster_list = []
        for c_id, pixel_ids in enumerate(c_cluster.cList):
            cluster = Cluster().from_netcluster(c_cluster, c_id)
            cluster_list.append(cluster)
        self.clusters = cluster_list
        return self


class Cluster:
    """
    Class for one cluster of pixels

    :param pixels: list of pixels
    :type pixels: list[Pixel]
    :param cluster_meta: cluster metadata
    :type cluster_meta: ClusterMeta
    :param cluster_status: cluster selection flags (cuts)  1 - rejected, 0 - not processed / accepted,
     -1 - not complete, -2 - ready for processing
    :type cluster_status: int
    :param cluster_rate: cluster type defined by rate
    :type cluster_rate: list[int]
    :param cluster_time: supercluster central time
    :type cluster_time: float
    :param cluster_freq: supercluster central frequency
    :type cluster_freq: float
    :param sky_area: sky error area
    :type sky_area: list[float]
    :param sky_pixel_map: sky pixel map
    :type sky_pixel_map: list[float]
    :param sky_pixel_index: sky pixel index
    :type sky_pixel_index: list[int]
    :param sky_time_delay: sky time delay
    :type sky_time_delay: list[int]

    """
    __slots__ = ['pixels', 'cluster_meta', 'cluster_status',
                 'cluster_rate', 'cluster_time', 'cluster_freq', 'sky_area', 'sky_pixel_map',
                 'sky_pixel_index', 'sky_time_delay']

    def __init__(self, pixels=None, cluster_meta=None, cluster_status=None, cluster_rate=None,
                 cluster_time=None, cluster_freq=None, sky_area=None, sky_pixel_map=None,
                 sky_pixel_index=None, sky_time_delay=None):
        #: pixel list
        self.pixels = pixels
        #: cluster metadata
        self.cluster_meta = cluster_meta
        #: cluster selection flags (cuts)
        self.cluster_status = cluster_status
        #: cluster type defined by rate
        self.cluster_rate = cluster_rate
        #: supercluster central time
        self.cluster_time = cluster_time
        #: supercluster central frequency
        self.cluster_freq = cluster_freq
        #: sky error regions
        self.sky_area = sky_area
        #: sky pixel map
        self.sky_pixel_map = sky_pixel_map
        #: sky pixel index
        self.sky_pixel_index = sky_pixel_index
        #: sky time delay configuration for waveform backward correction
        self.sky_time_delay = sky_time_delay



    def __repr__(self):
        return self.to_dict().__repr__()

    def to_dict(self):
        return {key: getattr(self, key) for key in self.__slots__}

    def from_netcluster(self, netcluster, c_id):
        self.pixels = [Pixel().from_netpixel(netcluster.pList[pixel_id]) for pixel_id in netcluster.cList[c_id]]
        self.cluster_meta = netcluster.cData[c_id]
        self.cluster_status = netcluster.sCuts[c_id]
        self.cluster_rate = list(netcluster.cRate[c_id])
        self.cluster_time = netcluster.cTime[c_id]
        self.cluster_freq = netcluster.cFreq[c_id]
        self.sky_area = list(netcluster.sArea[c_id])
        self.sky_pixel_map = list(netcluster.p_Map[c_id])
        self.sky_pixel_index = list(netcluster.p_Ind[c_id])
        self.sky_time_delay = list(netcluster.nTofF[c_id])
        return self


class ClusterMeta:
    __slots__ = ['energy', 'energy_sky', 'like_net', 'net_ecor', 'norm_cor', 'net_null', 'net_ed',
                 'g_noise', 'like_sky', 'sky_cc', 'net_cc', 'sky_chi2', 'sub_net', 'sub_net2',
                 'sky_stat', 'net_rho', 'net_rho2', 'theta', 'phi', 'iota', 'psi', 'ellipticity',
                 'c_time', 'c_freq', 'g_net', 'a_net', 'i_net', 'norm', 'ndof', 'tmrgr', 'tmrgrerr',
                 'mchirp', 'mchirperr', 'chi2chirp', 'chirp_efrac', 'chirp_pfrac', 'chirp_ellip',
                 'sky_size', 'sky_index']

    def __init__(self, energy=None, energy_sky=None, like_net=None, net_ecor=None, norm_cor=None,
                 net_null=None, net_ed=None, g_noise=None, like_sky=None, sky_cc=None, net_cc=None,
                 sky_chi2=None, sub_net=None, sub_net2=None, sky_stat=None, net_rho=None,
                 net_rho2=None, theta=None, phi=None, iota=None, psi=None, ellipticity=None,
                 c_time=None, c_freq=None, g_net=None, a_net=None, i_net=None, norm=None, ndof=None,
                 tmrgr=None, tmrgrerr=None, mchirp=None, mchirperr=None, chi2chirp=None,
                 chirp_efrac=None, chirp_pfrac=None, chirp_ellip=None, sky_size=None,
                 sky_index=None):
        #: total cluster energy
        self.energy = energy
        #: cluster energy in all resolutions
        self.energy_sky = energy_sky
        #: signal energy
        self.like_net = like_net
        #: network coherent energy
        self.net_ecor = net_ecor
        #: normalized coherent energy
        self.norm_cor = norm_cor
        #: null energy in the sky loop with Gauss correction
        self.net_null = net_null
        #: energy disbalance
        self.net_ed = net_ed
        #: estimated contribution of Gaussian noise
        self.g_noise = g_noise
        #: likelihood at all resolutions
        self.like_sky = like_sky
        #: network cc from the sky loop (all resolutions)
        self.sky_cc = sky_cc
        #: network cc for MRA or SRA analysis
        self.net_cc = net_cc
        #: chi2 stat from the sky loop (all resolutions)
        self.sky_chi2 = sky_chi2
        #: first subNetCut statistic
        self.sub_net = sub_net
        #: second subNetCut statistic
        self.sub_net2 = sub_net2
        #: localization statistic
        self.sky_stat = sky_stat
        #: coherent SNR per detector
        self.net_rho = net_rho
        #: reduced coherent SNR per detector
        self.net_rho2 = net_rho2
        #: sky position theta
        self.theta = theta
        #: sky position phi
        self.phi = phi
        #: sky position iota
        self.iota = iota
        #: sky position psi
        self.psi = psi
        #: sky position ellipticity
        self.ellipticity = ellipticity
        #: supercluster central time
        self.c_time = c_time
        #: supercluster central frequency
        self.c_freq = c_freq
        #: network acceptance
        self.g_net = g_net
        #: network alignment
        self.a_net = a_net
        #: network index
        self.i_net = i_net
        #: packet norm
        self.norm = norm
        #: cluster degrees of freedom
        self.ndof = ndof
        #: merger time
        self.tmrgr = tmrgr
        #: merger time error
        self.tmrgrerr = tmrgrerr
        #: chirp mass
        self.mchirp = mchirp
        #: chirp mass error
        self.mchirperr = mchirperr
        #: chi2 over NDF
        self.chi2chirp = chi2chirp
        #: chirp energy fraction
        self.chirp_efrac = chirp_efrac
        #: chirp pixel fraction
        self.chirp_pfrac = chirp_pfrac
        #: chirp ellipticity
        self.chirp_ellip = chirp_ellip
        #: number of sky pixels
        self.sky_size = sky_size
        #: index in the skymap
        self.sky_index = sky_index
