import copy

import numpy as np
from scipy.sparse import coo_array

from pycwb.types.network_pixel import Pixel
from pycwb.utils.image import resize_resolution, align_images, merge_images


class FragmentCluster:
    """
    Class for clusters found in data fragment

    Parameters
    ----------
    rate : float
        original Time series rate
    start : float
        interval start GPS time
    stop : float
        interval stop  GPS time
    bpp : float
        black pixel probability
    shift : float
        time shift
    f_low : float
        low frequency boundary
    f_high : float
        high frequency boundary
    n_pix : int
        minimum number of pixels at all resolutions
    run : int
        run ID
    pair : bool
        true - 2 resolutions, false - 1 resolution
    n_sub : int
        subnetwork threshold for a single network pixel
    clusters : list of Cluster
        cluster list
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

    def event_count(self, event_status=None):
        """
        Count number of events in clusters

        :param event_status: event status, 1 - rejected, 0 - not processed / accepted, -1 - not complete, -2 - ready for processing, None - all
        :type event_status: int
        :return:
        """
        if event_status is not None:
            if not isinstance(event_status, int) or event_status > 1 or event_status < -2:
                raise ValueError('event_status must be -2, -1, 0, 1 or None')

        if event_status is None:
            return len([c.cluster_status for c in self.clusters if c.cluster_status < 1])
        else:
            return len([c.cluster_status for c in self.clusters if c.cluster_status == event_status])

    def pixel_count(self, event_status=None):
        """
        Count number of pixels in clusters

        :param event_status: event status, 1 - rejected, 0 - not processed / accepted, -1 - not complete, -2 - ready for processing, None - all
        :type event_status: int
        :return:
        """
        if event_status is not None:
            if not isinstance(event_status, int) or event_status > 1 or event_status < -2:
                raise ValueError('event_status must be -2, -1, 0, 1 or None')

        pixel_count = 0

        for c in self.clusters:
            if event_status is None:
                if c.cluster_status < 1:
                    pixel_count += len(c.pixels)
            else:
                if c.cluster_status == event_status:
                    pixel_count += len(c.pixels)
        return pixel_count

    def dump_cluster(self, cluster_id):
        """
        Select cluster by id

        Parameters
        ----------
        cluster_id : int
            cluster id

        Returns
        -------
        FragmentCluster
            fragment cluster with one cluster
        """
        temp_cluster = copy.copy(self)
        temp_cluster.clusters = [self.clusters[cluster_id]]

        return temp_cluster

    def remove_rejected(self):
        """
        Remove rejected clusters
        """
        self.clusters = [c for c in self.clusters if c.cluster_status < 1]


class Cluster:
    """
    Class for one cluster of pixels

    Parameters
    ----------
    pixels : list of Pixel
        list of pixels
    cluster_meta : ClusterMeta
        cluster metadata
    cluster_status : int
        cluster selection flags (cuts)  1 - rejected, 0 - not processed / accepted, -1 - not complete, -2 - ready for processing
    cluster_rate : list of int
        cluster type defined by rate
    cluster_time : float
        supercluster central time
    cluster_freq : float
        supercluster central frequency
    sky_area : list of float
        sky error area
    sky_pixel_map : list of float
        sky pixel map
    sky_pixel_index : list of int
        sky pixel index
    sky_time_delay : list of float
        sky time delay
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
        """
        Convert Cluster to dict

        Returns
        -------
        dict
            dict with Cluster attributes
        """
        return {key: getattr(self, key) for key in self.__slots__}

    def from_netcluster(self, netcluster, c_id):
        """
        Convert ROOT.netcluster to Cluster

        Parameters
        ----------
        netcluster : ROOT.netcluster
            ROOT.netcluster object
        c_id : int
            cluster id to convert

        Returns
        -------
        Cluster
            converted Cluster
        """
        self.pixels = [Pixel().from_netpixel(netcluster.pList[pixel_id]) for pixel_id in netcluster.cList[c_id]]
        self.cluster_meta = ClusterMeta().from_cData(netcluster.cData[c_id])
        self.cluster_status = netcluster.sCuts[c_id]
        self.cluster_rate = list(netcluster.cRate[c_id])
        self.cluster_time = netcluster.cTime[c_id]
        self.cluster_freq = netcluster.cFreq[c_id]
        self.sky_area = list(netcluster.sArea[c_id])
        self.sky_pixel_map = list(netcluster.p_Map[c_id])
        self.sky_pixel_index = list(netcluster.p_Ind[c_id])
        self.sky_time_delay = list(netcluster.nTofF[c_id])
        return self

    def get_pixel_rates(self):
        """
        Get all pixel rates

        Returns
        -------
        list of int
            list of pixel rates
        """
        return [p.rate for p in self.pixels]

    def get_pixels_with_rate(self, rate):
        """
        Get pixels with rate

        Parameters
        ----------
        rate : int
            pixel rate to select

        Returns
        -------
        list of Pixel
            list of pixels with selected rate
        """
        return [p for p in self.pixels if p.rate == rate]

    def get_sparse_map_by_rate(self, key='likelihood'):
        """
        Get sparse map for selected key

        Returns
        -------
        v_maps: scipy.sparse.coo_matrix
            sparse map
        t_starts: list of int
            list of start times
        dts: list of float
            list of time steps
        dfs: list of float
            list of frequency steps
        """
        pixels = self.pixels

        # get the sort the rates
        rates = sorted(set([p.rate for p in pixels]))

        # get the pixels for each rate
        res_layers = []
        for r in rates:
            res_layers.append([p for p in pixels if p.rate == r])

        # generate the sparse map for each rate
        v_maps = []
        dts = []
        dfs = []
        t_starts = []

        for res in res_layers:
            one_pixel = res[0]
            dt = 1 / one_pixel.rate
            df = one_pixel.rate / 2
            times = [int(p.time / p.layers) for p in res]
            freqs = [p.frequency for p in res]
            values = [getattr(p, key) for p in res]
            v_map = coo_array((values, (times, freqs)), shape=(max(times) + 1, max(freqs) + 1))

            # strip the zeros
            t_start = v_map.nonzero()[0].min()
            d = v_map.toarray()[t_start:]

            # append for each rate
            v_maps.append(d)
            dts.append(dt)
            dfs.append(df)
            t_starts.append(t_start)
        return v_maps, t_starts, dts, dfs

    def get_sparse_map(self, key='likelihood'):
        """
        Get sparse map for selected key

        Parameters
        ----------
        key : str
            key to select

        Returns
        -------
        mergerd_map: np.ndarray
            merged sparse map for all resoltions
        t_start: int
            start time
        dt: float
            time step
        df: float
            frequency step
        """
        v_maps, t_starts, dts, dfs = self.get_sparse_map_by_rate(key=key)

        min_dt = min(dts)
        min_df = min(dfs)

        # resize to the same resolution
        resized_maps = [resize_resolution(v_map, dts[i], dfs[i], min_dt, min_df) for i, v_map in enumerate(v_maps)]

        # calculate shift
        t_starts_shifted = ((np.array(t_starts) * np.array(dts) - np.array(dts) / 2) / min_dt).astype(int)

        # align images
        aligned_images = align_images(resized_maps, t_starts_shifted)

        # merge images
        merged_map = merge_images(aligned_images)

        # start time
        t_start_new = min(t_starts_shifted) * min_dt

        return merged_map, t_start_new, min_dt, min_df

    # def generate_multi_resolution_map(self):
    #     pixels = self.pixels
    #     one_pixel = pixels[0]
    #     rate = (one_pixel.layers - 1) * one_pixel.rate
    #
    #     times = [p.time_in_seconds for p in pixels]
    #     freqs = [p.frequency_in_hz for p in pixels]
    #     dt = [1/p.rate for p in pixels]
    #     df = [p.rate / 2 for p in pixels]
    #     layers = [p.layers for p in pixels]
    #     max_layers = max(layers)
    #     min_layers = min(layers)
    #     max_rate = rate / (min_layers - 1)
    #     min_rate = rate / (max_layers - 1)
    #     min_dt = 1 / max_rate
    #     max_dt = 1 / min_rate
    #     min_df = min_rate / 2
    #     max_df = max_rate / 2
    #
    #     min_time = min(times) - max_dt
    #     max_time = max(times) + max_dt
    #     max_freq = max(freqs)
    #     min_freq = min(freqs)
    #     # max_freq = rate / 2
    #     # min_freq = 0
    #     n_times = int((max_time - min_time) / min_dt)
    #     n_freqs = 2 * (max_layers - 1)
    #
    #     df_plot = max((max_freq - min_freq) / 10., 2 * max_df)
    #     dt_plot = max((max_time - min_time) / 10., 2 * max_dt)
    #     min_freq_plot = max(0, min_freq - df_plot)
    #     max_freq_plot = min(rate / 2, max_freq + df_plot)
    #     min_time_plot = min(times) - max(max_dt, dt_plot)
    #     max_time_plot = max(times) + min(max_dt, dt_plot)


class ClusterMeta:
    """
    Cluster meta data
    """
    __slots__ = ['energy', 'energy_sky', 'like_net', 'net_ecor', 'norm_cor', 'net_null', 'net_ed',
                 'g_noise', 'like_sky', 'sky_cc', 'net_cc', 'sky_chi2', 'sub_net', 'sub_net2',
                 'sky_stat', 'net_rho', 'net_rho2', 'theta', 'phi', 'iota', 'psi', 'ellipticity',
                 'c_time', 'c_freq', 'g_net', 'a_net', 'i_net', 'norm', 'ndof', 'tmrgr', 'tmrgrerr',
                 'mchirp', 'mchirperr', 'chi2chirp', 'chirp_efrac', 'chirp_pfrac', 'chirp_ellip',
                 'sky_size', 'sky_index', 'chirp', 'mchpdf']

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

        self.chirp = None
        self.mchpdf = None

    def __dict__(self):
        return {k: getattr(self, k) for k in self.__slots__}

    def from_cData(self, c_data):
        self.energy = c_data.energy
        self.energy_sky = c_data.enrgsky
        self.like_net = c_data.likenet
        self.net_ecor = c_data.netecor
        self.norm_cor = c_data.normcor
        self.net_null = c_data.netnull
        self.net_ed = c_data.netED
        self.g_noise = c_data.Gnoise
        self.like_sky = c_data.likesky
        self.sky_cc = c_data.skycc
        self.net_cc = c_data.netcc
        self.sky_chi2 = c_data.skyChi2
        self.sub_net = c_data.subnet
        self.sub_net2 = c_data.SUBNET
        self.sky_stat = c_data.skyStat
        self.net_rho = c_data.netRHO
        self.net_rho2 = c_data.netrho
        self.theta = c_data.theta
        self.phi = c_data.phi
        self.iota = c_data.iota
        self.psi = c_data.psi
        self.ellipticity = c_data.ellipticity
        self.c_time = c_data.cTime
        self.c_freq = c_data.cFreq
        self.g_net = c_data.gNET
        self.a_net = c_data.aNET
        self.i_net = c_data.iNET
        self.norm = c_data.norm
        self.ndof = c_data.nDoF
        self.tmrgr = c_data.tmrgr
        self.tmrgrerr = c_data.tmrgrerr
        self.mchirp = c_data.mchirp
        self.mchirperr = c_data.mchirperr
        self.chi2chirp = c_data.chi2chirp
        self.chirp_efrac = c_data.chirpEfrac
        self.chirp_pfrac = c_data.chirpPfrac
        self.chirp_ellip = c_data.chirpEllip
        self.sky_size = c_data.skySize
        self.sky_index = c_data.skyIndex
        # TODO: pythonize these
        self.chirp = c_data.chirp
        self.mchpdf = c_data.mchpdf
        return self
