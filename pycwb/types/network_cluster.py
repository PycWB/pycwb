import copy
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np
from scipy.sparse import coo_array

from pycwb.types.network_pixel import Pixel
from pycwb.types.pixel_arrays import PixelArrays, empty_pixel_arrays
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
    l_max: int = 0
    # Per-IFO xtalk-corrected waveform energies (getMRAwave equivalents, set by fill_detection_statistic)
    wave_snr: list = field(default_factory=list)    # data energy per IFO (C++ d->enrg = get_XX())
    signal_snr: list = field(default_factory=list)  # signal energy per IFO (C++ d->sSNR = get_SS())
    cross_snr: list = field(default_factory=list)   # xSNR per IFO (C++ d->xSNR = get_XS())
    signal_energy_physical: list = field(default_factory=list)  # physical strain energy per IFO for hrss
    null_energy: list = field(default_factory=list)  # null energy per IFO (C++ d->null)


@dataclass(init=False)
class Cluster:
    """
    Class for one cluster of pixels.

    Primary storage is ``pixel_arrays`` (a :class:`~pycwb.types.pixel_arrays.PixelArrays`
    struct-of-arrays).  The legacy ``pixels`` attribute is a computed property that
    reconstructs a ``list[Pixel]`` on demand for backward-compatible downstream code.

    Parameters
    ----------
    pixel_arrays : PixelArrays
        SoA pixel data.  Provide *either* this *or* the legacy ``pixels`` kwarg.
    pixels : list[Pixel], optional
        Backward-compat: if supplied, converted to ``pixel_arrays`` automatically.
    cluster_meta : ClusterMeta
        Cluster statistics metadata.
    cluster_status : int
        Selection flag: 1 rejected, 0 accepted/not-processed, -1 incomplete, -2 ready.
    """
    # ---- primary storage ----
    pixel_arrays: PixelArrays = field(default=None, repr=False)

    # ---- metadata ----
    cluster_meta: ClusterMeta = field(default_factory=ClusterMeta)
    cluster_status: int = 0
    cluster_id: int = 0
    cluster_rate: list[int] = field(default_factory=list)
    cluster_time: float = 0.0
    cluster_freq: float = 0.0
    sky_area: list[float] = field(default_factory=list)
    sky_pixel_map: list[float] = field(default_factory=list)
    sky_pixel_index: list[int] = field(default_factory=list)
    sky_time_delay: list[float] = field(default_factory=list)

    def __init__(
        self,
        cluster_meta: ClusterMeta | None = None,
        pixel_arrays: PixelArrays | None = None,
        pixels: list | None = None,          # backward-compat alias
        cluster_status: int = 0,
        cluster_id: int = 0,
        cluster_rate: list | None = None,
        cluster_time: float = 0.0,
        cluster_freq: float = 0.0,
        sky_area: list | None = None,
        sky_pixel_map: list | None = None,
        sky_pixel_index: list | None = None,
        sky_time_delay: list | None = None,
    ) -> None:
        if pixel_arrays is None:
            if pixels:
                n_ifo = len(pixels[0].data)
                pixel_arrays = PixelArrays.from_pixels(pixels, n_ifo)
            else:
                pixel_arrays = empty_pixel_arrays(0)
        self.pixel_arrays    = pixel_arrays
        self.cluster_meta    = cluster_meta if cluster_meta is not None else ClusterMeta()
        self.cluster_status  = cluster_status
        self.cluster_id      = cluster_id
        self.cluster_rate    = list(cluster_rate) if cluster_rate is not None else []
        self.cluster_time    = cluster_time
        self.cluster_freq    = cluster_freq
        self.sky_area        = list(sky_area)        if sky_area        is not None else []
        self.sky_pixel_map   = list(sky_pixel_map)   if sky_pixel_map   is not None else []
        self.sky_pixel_index = list(sky_pixel_index) if sky_pixel_index is not None else []
        self.sky_time_delay  = list(sky_time_delay)  if sky_time_delay  is not None else []

    # ------------------------------------------------------------------ #
    # pixels: backward-compat property backed by pixel_arrays
    # ------------------------------------------------------------------ #

    @property
    def pixels(self) -> list[Pixel]:
        """Reconstruct and return ``list[Pixel]`` from the SoA storage.

        This is a compatibility shim for downstream code that iterates over
        individual ``Pixel`` objects.  Hot paths should read ``pixel_arrays``
        directly.
        """
        return self.pixel_arrays.to_pixel_list() if self.pixel_arrays else []

    def get_pixel_rates(self):
        """Return all unique pixel rates."""
        return list(np.unique(self.pixel_arrays.rate))

    def get_pixels_with_rate(self, rate):
        """Return pixels (as list[Pixel]) with the given rate."""
        mask = self.pixel_arrays.rate == rate
        return self.pixel_arrays[mask].to_pixel_list()

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
        pa = self.pixel_arrays
        rates = sorted(set(pa.rate.tolist()))

        v_maps = []
        dts = []
        dfs = []
        t_starts = []

        for r in rates:
            mask = pa.rate == r
            sub = pa[mask]
            if len(sub) == 0:
                continue

            dt = 1.0 / r
            df = r / 2.0
            times  = (sub.time  // sub.layers).tolist()
            freqs  = sub.frequency.tolist()
            values = getattr(sub, key).tolist() if hasattr(sub, key) else [0.0] * len(sub)
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
        """Return the total number of pixels."""
        return len(self.pixel_arrays)

    def get_analyzed_size(self):
        """Return the number of core pixels with positive likelihood."""
        pa = self.pixel_arrays
        return int(np.sum(pa.core & (pa.likelihood > 0)))

    def get_core_size(self):
        """Return the number of core pixels."""
        return int(np.sum(self.pixel_arrays.core))

    @property
    def start_time(self):
        """Cluster start time (seconds)."""
        pa = self.pixel_arrays
        dt = 1.0 / pa.rate  # (n_pix,)
        t_sec = (pa.time.astype(np.float64) / pa.layers.astype(np.float64)) / pa.rate
        return float(np.min(t_sec + dt / 2))

    @property
    def stop_time(self):
        """Cluster stop time (seconds)."""
        pa = self.pixel_arrays
        dt = 1.0 / pa.rate
        t_sec = (pa.time.astype(np.float64) / pa.layers.astype(np.float64)) / pa.rate
        return float(np.max(t_sec + dt / 2 + dt))

    @property
    def duration(self):
        """Cluster duration (seconds)."""
        return self.stop_time - self.start_time

    @property
    def low_frequency(self):
        """Cluster low frequency (Hz)."""
        pa = self.pixel_arrays
        return float(np.min((pa.frequency.astype(np.float64) - 0.5) * pa.rate / 2))

    @property
    def high_frequency(self):
        """Cluster high frequency (Hz)."""
        pa = self.pixel_arrays
        return float(np.max((pa.frequency.astype(np.float64) - 0.5) * pa.rate / 2 + pa.rate / 2))

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
    rate: float = 0.0
    start: float = 0.0
    stop: float = 0.0
    bpp: float = 1.0
    shift: float = 0.0
    f_low: float = 0.0
    f_high: float = 0.0
    n_pix: int = 0
    run: int = 0
    pair: bool = False
    subnet_threshold: float = 0.0
    clusters: list[Cluster] = field(default_factory=list)


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
            return sum(1 for cluster in self.clusters if cluster.cluster_status < 1)
        return sum(1 for cluster in self.clusters if cluster.cluster_status == event_status)

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

        if event_status is None:
            return sum(len(cluster.pixel_arrays) for cluster in self.clusters if cluster.cluster_status < 1)
        return sum(len(cluster.pixel_arrays) for cluster in self.clusters if cluster.cluster_status == event_status)

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

    def select(self, name: str, threshold: float) -> np.ndarray:
        """
        Select/reject clusters by a scalar cluster statistic.

        This is a Python-native counterpart of cWB `netcluster::select` for
        common filters used in coherence (`subrho`, `subnet`).

        Parameters
        ----------
        name : str
            Selection name. Supported: `subrho`, `subnet`.
        threshold : float
            Rejection threshold. Cluster is rejected when value < threshold.

        Returns
        -------
        np.ndarray
            Array of per-cluster values used by the selection.
        """
        key = str(name).strip().lower()

        if key == "subrho":
            values = np.array([c.cluster_meta.net_rho for c in self.clusters], dtype=float)
        elif key == "subnet":
            values = np.array([c.cluster_meta.sub_net for c in self.clusters], dtype=float)
        else:
            raise ValueError(f"Unsupported selection '{name}'. Supported: subrho, subnet")

        for cluster, value in zip(self.clusters, values):
            if cluster.cluster_status > 0:
                continue
            if value < threshold:
                cluster.cluster_status = 1

        return values


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

