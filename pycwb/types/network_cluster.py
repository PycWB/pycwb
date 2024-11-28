import copy
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_array

from pycwb.types.network_pixel import Pixel
from pycwb.utils.image import resize_resolution, align_images, merge_images


@dataclass
class ClusterMeta:
    """
    Cluster meta data

    energy: float
        total cluster energy
    energy_sky: float
        cluster energy in all resolutions
    like_net: float
        signal energy
    net_ecor: float
        network coherent energy
    norm_cor: float
        normalized coherent energy
    net_null: float
        null energy in the sky loop with Gauss correction
    net_ed: float
        energy disbalance
    g_noise: float
        estimated contribution of Gaussian noise
    like_sky: float
        likelihood at all resolutions
    sky_cc: float
        network cc from the sky loop (all resolutions)
    net_cc: float
        network cc for MRA or SRA analysis
    sky_chi2: float
        chi2 stat from the sky loop (all resolutions)
    sub_net: float
        first subNetCut statistic
    sub_net2: float
        second subNetCut statistic
    sky_stat: float
        localization statistic
    net_rho: float
        coherent SNR per detector
    net_rho2: float
        reduced coherent SNR per detector
    theta: float
        sky position theta
    phi: float
        sky position phi
    iota: float
        sky position iota
    psi: float
        sky position psi
    ellipticity: float
        sky position ellipticity
    c_time: float
        supercluster central time
    c_freq: float
        supercluster central frequency
    g_net: float
        network acceptance
    a_net: float
        network alignment
    i_net: float
        network index
    norm: float
        packet norm
    ndof: float
        cluster degrees of freedom
    sky_size: float
        number of sky pixels
    sky_index: float
        index in the skymap
    """
    energy: float = 0.
    energy_sky: float = 0.
    like_net: float = 0.
    net_ecor: float = 0.
    norm_cor: float = 0.
    net_null: float = 0.
    net_ed: float = 0.
    g_noise: float = 0.
    like_sky: float = 0.
    sky_cc: float = 0.
    net_cc: float = 0.
    sky_chi2: float = 0.
    sub_net: float = 0.
    sub_net2: float = 0.
    sky_stat: float = 0.
    net_rho: float = 0.
    net_rho2: float = 0.
    theta: float = 0.
    phi: float = 0.
    iota: float = 0.
    psi: float = 0.
    ellipticity: float = 0.
    c_time: float = 0.
    c_freq: float = 0.
    g_net: float = 0.
    a_net: float = 0.
    i_net: float = 0.
    norm: float = 0.
    ndof: float = 0.
    sky_size: int = 0
    sky_index: int = 0


@dataclass
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
    pixels: list[Pixel]
    cluster_meta: ClusterMeta
    cluster_status: int = 0
    cluster_rate: list[int] = field(default_factory=list)
    cluster_time: float = 0.0
    cluster_freq: float = 0.0
    sky_area: list[float] = field(default_factory=list)
    sky_pixel_map: list[float] = field(default_factory=list)
    sky_pixel_index: list[int] = field(default_factory=list)
    sky_time_delay: list[float] = field(default_factory=list)

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

            # if all zeros, skip
            if len(v_map.nonzero()[0]) == 0:
                continue
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
        merged_map[merged_map == 0] = np.nan
        # start time
        t_start_new = min(t_starts_shifted) * min_dt

        return merged_map, t_start_new, min_dt, min_df

    def get_size(self):
        """
        Get cluster size

        Returns
        -------
        int
            cluster size
        """
        return len(self.pixels)

    def get_analyzed_size(self):
        """
        Get analyzed pixels

        Returns
        -------
        int
            analyzed pixels
        """
        return len([p for p in self.pixels if p.likelihood > 0 and p.core])


@dataclass
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
    subnet_threshold : int
        subnetwork threshold for a single network pixel
    clusters : list of Cluster
        cluster list
    """
    rate: float
    start: float
    stop: float
    bpp: float
    shift: float
    f_low: float
    f_high: float
    n_pix: int
    run: int
    pair: bool
    subnet_threshold: float
    clusters: list[Cluster]


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


@dataclass(slots=True)
class ClusterChirp:
    """
    Class for cluster chirp statistics
    """
    tmrgr: float   # merger time
    tmrgrerr: float  # merger time error
    mchirp: float  # chirp mass
    mchirperr: float  # chirp mass error
    chi2chirp: float   # chi2 over NDF
    chirp_efrac: float  # chirp energy fraction
    chirp_pfrac: float  # chirp pixel fraction
    chirp_ellip: float  # chirp ellipticity
    chirp: list[float]  # chirp graph
    mchpdf: list[float]  # chirp mass PDF
        # self.tmrgr = c_data.tmrgr
        # self.tmrgrerr = c_data.tmrgrerr
        # self.mchirp = c_data.mchirp
        # self.mchirperr = c_data.mchirperr
        # self.chi2chirp = c_data.chi2chirp
        # self.chirp_efrac = c_data.chirpEfrac
        # self.chirp_pfrac = c_data.chirpPfrac
        # self.chirp_ellip = c_data.chirpEllip
        # # TODO: pythonize these
        # self.chirp = c_data.chirp
        # self.mchpdf = c_data.mchpdf

