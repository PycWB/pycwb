import base64
import hashlib

import numpy as np
import json
from dataclasses import dataclass, field
from pycwb.types.network_cluster import Cluster
from pycwb.types.job import WaveSegment


@dataclass()
class Event:
    """
    This class is used to store the results of an event.
    """
    id: str = ""  # event ID
    nevent: int = 0  # number of events
    ndim: int = field(default_factory=int)  # number of dimensions
    run: int = field(default_factory=int)  # run ID
    rho: list[float] = field(default_factory=list)  # effective correlated SNR
    netcc: list[float] = field(default_factory=list)  # network correlation coefficients: 0-net,1-pc,2-cc,3-net2
    neted: list[float] = field(default_factory=list)  # network energy disbalance: 0 - total, 1 - 00-phase, 2 - 90-phase
    gnet: float = field(default_factory=float)  # network sensitivity
    anet: float = field(default_factory=float)  # network alignment factor
    inet: float = field(default_factory=float)  # network index
    ecor: float = field(default_factory=float)  # correlated energy
    norm: float = field(default_factory=float)  # norm Factor or ellipticity
    ECOR: float = field(default_factory=float)  # effective correlated energy
    penalty: float = field(default_factory=float)  # penalty factor
    likelihood: float = field(default_factory=float) # network likelihood
    factor: float = field(default_factory=float)  # Multiplicative amplitude factor - simulation only
    range: list[int] = field(default_factory=list) # range to source: [0/1]-rec/inj
    chirp: list[float] = field(default_factory=list) # chirp array: 0-injmass,1-recmass,2-merr,3-tmrgr,4-terr,5-chi2
    eBBH: list[float] = field(default_factory=list)  # eBBH array
    usize: float = 0.0  # size of the universe
    ifo_list: list[str] = field(default_factory=list)  # list of ifos
    eventID: list[int] = field(default_factory=list) # event ID
    type: list[int] = field(default_factory=list)  # event type
    name: list[str] = field(default_factory=list)  # event name
    log: list[str] = field(default_factory=list)
    rate: list[float] = field(default_factory=list)
    volume: list[float] = field(default_factory=list)
    size: list[float] = field(default_factory=list)
    gap: list[float] = field(default_factory=list)
    lag: list[float] = field(default_factory=list)
    slag: list[float] = field(default_factory=list)
    strain: list[float] = field(default_factory=list)
    phi: list[float] = field(default_factory=list)
    theta: list[float] = field(default_factory=list)
    psi: list[float] = field(default_factory=list)
    iota: list[float] = field(default_factory=list)
    bp: list[float] = field(default_factory=list)
    bx: list[float] = field(default_factory=list)
    time: list[float] = field(default_factory=list)
    gps: list[float] = field(default_factory=list)
    right: list[float] = field(default_factory=list)
    left: list[float] = field(default_factory=list)
    duration: list[float] = field(default_factory=list)
    start: list[float] = field(default_factory=list)
    stop: list[float] = field(default_factory=list)
    frequency: list[float] = field(default_factory=list)
    low: list[float] = field(default_factory=list)
    high: list[float] = field(default_factory=list)
    bandwidth: list[float] = field(default_factory=list)
    hrss: list[float] = field(default_factory=list)
    noise: list[float] = field(default_factory=list)
    erA: list[float] = field(default_factory=list)
    Psm: list[float] = field(default_factory=list)
    null: list[float] = field(default_factory=list)
    nill: list[float] = field(default_factory=list)
    mass: list[float] = field(default_factory=list)
    spin: list[float] = field(default_factory=list)
    snr: list[float] = field(default_factory=list)
    xSNR: list[float] = field(default_factory=list)
    sSNR: list[float] = field(default_factory=list)
    iSNR: list[float] = field(default_factory=list)
    oSNR: list[float] = field(default_factory=list)
    ioSNR: list[float] = field(default_factory=list)
    Deff: list[float] = field(default_factory=list)
    injection: dict = field(default_factory=dict)
    job_id: int = None


    # def __init__(self):
    #     self.nevent = 0  # number of events
    #     self.ndim = None  # number of dimensions
    #     self.run = None  # run ID
    #     self.rho = [0, 0, 0]  # effective correlated SNR
    #     self.netcc = [0, 0, 0]  # network correlation coefficients: 0-net,1-pc,2-cc,3-net2
    #     self.neted = [0, 0, 0]  # network energy disbalance: 0 - total, 1 - 00-phase, 2 - 90-phase
    #     self.gnet = None  # network sensitivity
    #     self.anet = None  # network alignment factor
    #     self.inet = None  # network index
    #     self.ecor = None  # correlated energy
    #     self.norm = None  # norm Factor or ellipticity
    #     self.ECOR = None  # effective correlated energy
    #     self.penalty = None  # penalty factor
    #     self.likelihood = None  # network likelihood
    #     self.factor = None  # Multiplicative amplitude factor - simulation only
    #     self.range = None  # range to source: [0/1]-rec/inj
    #     self.chirp = None  # chirp array: 0-injmass,1-recmass,2-merr,3-tmrgr,4-terr,5-chi2
    #     self.eBBH = None  # eBBH array
    #     self.usize = None  # size of the universe
    #     self.ifo_list = []  # list of ifos
    #     self.eventID = [0, 0, 0]  # event ID
    #     self.type = [0, 0, 0]  # event type
    #     self.name = [0, 0, 0]  # event name
    #     self.log = []
    #     self.rate = []
    #     self.volume = []
    #     self.size = []
    #     self.gap = []
    #     self.lag = []
    #     self.slag = [0, 0, 0]
    #     self.strain = []
    #     self.phi = []
    #     self.theta = []
    #     self.psi = []
    #     self.iota = []
    #     self.bp = []
    #     self.bx = []
    #     self.time = []
    #     self.gps = []
    #     self.right = []
    #     self.left = []
    #     self.duration = []
    #     self.start = []
    #     self.stop = []
    #     self.frequency = []
    #     self.low = []
    #     self.high = []
    #     self.bandwidth = []
    #     self.hrss = []
    #     self.noise = []
    #     self.erA = []
    #     self.Psm = []
    #     self.null = []
    #     self.nill = []
    #     self.mass = []
    #     self.spin = []
    #     self.snr = []
    #     self.xSNR = []
    #     self.sSNR = []
    #     self.iSNR = []
    #     self.oSNR = []
    #     self.ioSNR = []
    #     self.Deff = []

    def output(self, net, ID, LAG, shifts):
        """
        Generate event parameters from ROOT.network object

        :param net: network containing the event
        :type net: ROOT.network
        :param ID: cluster ID
        :type ID: int
        :param LAG: lag of the analysis
        :type LAG: int
        :return:
        """
        self.rho = [0, 0, 0]
        self.netcc = [0, 0, 0]
        self.neted = [0, 0, 0]
        self.eventID = [0, 0, 0]
        self.type = [0, 0, 0]
        self.slag = shifts if shifts is not None else [0, 0, 0]


        pwc = net.getwc(LAG)
        inRate = net.getifo(0).rate
        pat0 = True if net.pattern == 0 else False

        n_ifo = int(net.ifoListSize())
        # read cluster parameters

        rate_net = pwc.get("rate", 0, 'R', 0)
        vol0_net = pwc.get("volume", 0, 'R', 0)  # stored in volume[0]
        vol1_net = pwc.get("VOLUME", 0, 'R', 0)  # stored in volume[1]
        size_net = pwc.get("size", 0, 'R', 0)  # stored in size[0]
        low_net = pwc.get("low", 0, 'R', 0)
        high_net = pwc.get("high", 0, 'R', 0)
        cFreq_net = pwc.get("freq", 0, 'L', 0, False)
        duration_net = pwc.get("duration", 0, 'L', 0, False)
        bandwidth_net = pwc.get("bandwidth", 0, 'L', 0, False)

        cluster_ids = [int(i) for i in list(pwc.get("ID", 0, 'S', 0))]
        if ID not in cluster_ids:
            return

        kid = cluster_ids.index(ID)

        start_net = []
        stop_net = []
        noise_net = []
        NOISE_net = []
        for i in range(n_ifo):
            start_net.append(list(pwc.get("start", i+1, 'L', 0)))
            stop_net.append(list(pwc.get("stop", i+1, 'L', 0)))
            noise_net.append(list(pwc.get("noise", i+1, 'S', 0)))
            NOISE_net.append(list(pwc.get("NOISE", i+1, 'S', 0)))

        # print('duration ', np.array(start_net) - np.array(stop_net))

        psm = net.getifo(0).tau
        vI = net.wc_List[LAG].p_Ind[ID - 1]
        ind = vI[0]

        for i in range(0, n_ifo):
            # FIXME: the gps and some other parameters are same for all ifos
            self.gps.append(pwc.start + (self.slag[i] - self.slag[0]))
        pcd = pwc.cData[ID - 1]
        self.ndim = n_ifo
        psm.gps = pcd.cTime + self.gps[0]
        self.ecor = pcd.netecor
        self.nevent += 1
        self.eventID = [ID, 0]
        self.iota = [pcd.iota]
        self.psi = [pcd.psi]
        self.phi = [psm.getPhi(ind), 0, psm.getRA(ind), pcd.phi]
        self.theta = [psm.getTheta(ind), 0, psm.getDEC(ind), pcd.theta]
        self.gnet = pcd.gNET
        self.anet = pcd.aNET
        self.inet = pcd.iNET
        self.norm = pcd.norm
        self.likelihood = pcd.likenet
        self.volume = [int(vol0_net.data[kid] + 0.5), int(vol1_net.data[kid] + 0.5)]
        self.size = [int(size_net.data[kid] + 0.5), pcd.skySize]
        self.chirp = [0, pcd.mchirp, pcd.mchirperr, pcd.chirpEllip, pcd.chirpPfrac, pcd.chirpEfrac]
        self.range = [0]

        TAU = psm.get(self.theta[0], self.phi[0])
        M = 0
        gC = 0
        self.strain = [0]
        self.penalty = 0
        self.neted = [0, 0, 0, 0, 0]
        self.lag = [pwc.shift] * n_ifo
        net.getMRAwave(ID, LAG, 's', net.optim)
        for i in range(n_ifo):
            pd = net.getifo(i)
            Aa = pd.antenna(self.theta[0], self.phi[0], self.psi[0])
            self.type = [1]
            self.rate.append(rate_net.data[kid] if net.optim else 0)
            self.gap.append(0)
            self.lag.append(pd.lagShift.data[LAG])
            self.snr.append(pd.enrg)
            self.nill.append(pd.xSNR - pd.sSNR)
            self.null.append(pd.null)
            self.xSNR.append(pd.xSNR)
            self.sSNR.append(pd.sSNR)
            self.time.append(pcd.cTime + self.gps[i])

            if i > 0:
                self.time[i] += net.getifo(i).tau.get(self.theta[0], self.phi[0]) - TAU
            # print("start_net size = %d" % len(start_net[i]))
            # print("pwc size = %d" % len(pwc.get("ID", 0, 'S', 0)))
            # print("indexes i = %d, kid = %d" % (i, kid))
            self.left.append(start_net[i][kid])
            self.right.append(pwc.stop - pwc.start - stop_net[i][kid])
            self.duration.append(stop_net[i][kid] - start_net[i][kid])
            self.start.append(start_net[i][kid] + self.gps[i])
            self.stop.append(stop_net[i][kid] + self.gps[i])

            # take lag shift into account
            xstart = self.gps[i] + net.Edge  # start data
            xstop = self.gps[i] + pwc.stop - pwc.start - net.Edge  # end data
            self.time[i] += self.lag[i]
            if self.time[i] > xstop:
                self.time[i] = xstart + (self.time[i] - xstop)

            self.frequency.append(cFreq_net.data[kid])
            self.low.append(low_net.data[kid])
            self.high.append(high_net.data[kid])
            self.bandwidth.append(high_net.data[kid] - low_net.data[kid])

            self.hrss.append(np.sqrt(pd.get_SS() / inRate))
            self.noise.append(np.power(10., noise_net[i][kid]) / np.sqrt(inRate))
            self.bp.append(Aa.real())
            self.bx.append(Aa.imag())
            self.strain[0] += self.hrss[i] * self.hrss[i]

            # Aa /= np.power(10., NOISE_net[i][kid])
            # gC += Aa * Aa.conj()

            # psm.gps = pcd.cTime + self.gps[0]
            self.duration[0] = duration_net.data[kid]
            self.bandwidth[0] = bandwidth_net.data[kid]
            self.frequency[0] = pcd.cFreq


            #    this->ECOR     = pcd->normcor;                         // normalized coherent energy
            #    this->netcc[0] = pcd->netcc;                           // MRA or SRA cc statistic 
            #    this->netcc[1] = pcd->skycc;                           // all-resolution cc statistic
            #    this->netcc[2] = pcd->subnet;                          // MRA or SRA sub-network statistic 
            #    this->netcc[3] = pcd->SUBNET;                          // all-resolution sub-network statistic

            #    this->neted[0] = pcd->netED;                           // network ED
            #    this->neted[1] = pcd->netnull;                         // total null energy with Gaussian bias correction
            #    this->neted[2] = pcd->energy;                          // total event energy
            #    this->neted[3] = pcd->likesky;                         // total likelihood at all resolutions
            #    this->neted[4] = pcd->enrgsky;                         // total energy at all resolutions

            
            # for i in range(11):
            self.erA.append(np.array(pwc.sArea[ID - 1]))
            # ind = pwc.sArea[ID - 1].size(); 
            # for i in range(11):
            #     if i < ind:
            #         self.erA.append(pwc.sArea[ID - 1][i])
            #     else:
            #         self.erA.append(0)

            self.ECOR = pcd.normcor  # normalized coherent energy
            self.netcc = [
                pcd.netcc,  # MRA or SRA cc statistic
                pcd.skycc,  # all-resolution cc statistic
                pcd.subnet,  # MRA or SRA sub-network statistic
                pcd.SUBNET  # all-resolution sub-network statistic
            ]
            self.neted = [
                pcd.netED,  # network ED
                pcd.netnull,  # total null energy with Gaussian bias correction
                pcd.energy,  # total event energy
                pcd.likesky,  # total likelihood at all resolutions
                pcd.enrgsky  # total energy at all resolutions
            ]

            self.penalty = pcd.netnull / n_ifo
            self.penalty /= self.size[0] if pat0 else pcd.nDoF # cluster chi2/nDoF
        chrho = self.chirp[3] * np.sqrt(self.chirp[5])  # reduction factor for chirp events
        if pcd.netRHO >= 0:  # original 2G
            self.rho[0] = pcd.netRHO  # reduced coherent SNR per detector
            self.rho[
                1] = pcd.netrho if pat0 else pcd.netRHO * chrho  # reduced coherent SNR per detector for chirp events
        else:  # (XGB.rho0)
            self.rho[0] = -pcd.netRHO  # reduced coherent SNR per detector
            self.rho[1] = pcd.netrho  # reduced coherent SNR per detector # GV original 2G rho, only for tests

        # Take sqrt of strain (C++ netevent.cc line 983)
        self.strain[0] = np.sqrt(self.strain[0])

        self.id = self.long_id
    # def json(self):
    #     """
    #     Return a JSON representation of the event
    #
    #     :return: JSON format of the instance
    #     :rtype: str
    #     """
    #     return json.dumps(self.__dict__)

    def output_py(self, job_segment: WaveSegment, cluster: Cluster, config=None):
        """
        Populate event fields from a native (ROOT-free) Cluster object.

        Parameters
        ----------
        job_segment : WaveSegment
            The job segment that produced this cluster (provides GPS times, IFO list).
        cluster : Cluster
            The detected cluster whose ``cluster_meta`` and ``pixels`` are already
            populated by the native likelihood pipeline.
        config : Config, optional
            Pipeline configuration object.  Required to compute per-IFO amplitude
            fields (hrss, snr, sSNR, xSNR, noise, bp, bx) and timing corrections.
            If ``None`` those fields will be left as empty lists.
        """
        meta = cluster.cluster_meta
        n_ifo = len(job_segment.ifos)

        # --- GPS epoch per IFO ---
        self.gps = np.array([job_segment.physical_start_times[ifo] for ifo in job_segment.ifos])

        # --- Time-window geometry ---
        self.left = [float(cluster.start_time)] * n_ifo
        self.right = [float(job_segment.duration - cluster.stop_time)] * n_ifo
        self.start = [float(cluster.start_time + self.gps[i]) for i in range(n_ifo)]
        self.stop = [float(cluster.stop_time + self.gps[i]) for i in range(n_ifo)]
        self.low = [float(cluster.low_frequency)] * n_ifo
        self.high = [float(cluster.high_frequency)] * n_ifo

        # --- Network-level coherence statistics ---
        self.ecor = meta.net_ecor
        self.ECOR = meta.norm_cor
        self.likelihood = meta.like_net       # waveform likelihood (Lw = sum sSNR)
        self.norm = meta.norm                 # packet norm (= (Eo-Eh)/Ew * 2)
        self.netcc = [meta.net_cc, meta.sky_cc, meta.sub_net, meta.sub_net2]
        # rho[1]: netrho (raw rho) for pat0 mode; netRHO * chirp_factor for non-pat0 (mirrors output() logic)
        # rho[1] = netrho (raw rho before cc reduction); same as pat0 branch in output().
        # For chirp events: rho[1] = netRHO * chrho, but chirp not yet computed so use net_rho2.
        pat0 = (getattr(config, 'pattern', 10) == 0) if config is not None else False
        self.rho = [meta.net_rho, meta.net_rho2]  # rho[0]=netRHO=rho/sqrt(cc), rho[1]=netrho=raw rho
        self.anet = meta.a_net
        self.gnet = meta.g_net
        self.inet = meta.i_net
        self.ndim = n_ifo
        self.size = [cluster.get_core_size(), meta.sky_size]
        self.volume = [cluster.get_size(), meta.sky_size]  # [all pixels, core pixels]
        self.neted = [meta.net_ed, meta.net_null, meta.energy, meta.like_sky, meta.energy_sky]
        self.iota = [meta.iota]
        self.psi = [meta.psi]
        self.type = [1]
        self.range = [0]
        self.chirp = [0, 0, 0, 0, 0, 0]       # placeholder; mchirp not yet computed
        self.gap = [0] * n_ifo
        n_ifo_val = config.nIFO if config is not None else n_ifo
        ndof_val = max(meta.ndof, 1e-10)
        if pat0:
            self.penalty = meta.net_null / max(n_ifo_val * cluster.get_core_size(), 1e-10)
        else:
            self.penalty = meta.net_null / max(n_ifo_val * ndof_val, 1e-10)

        # --- Sky localisation ---
        # meta.theta / meta.phi are in the Earth-fixed (geographic) frame,
        # matching CWB's convention: phi_geo = geographic longitude,
        # theta_geo = geographic co-latitude (0 = N pole, 180 = S pole).
        # Equatorial RA = phi_geo + GMST(t_event), dec = 90 - theta_geo.
        theta_deg = meta.theta  # geographic co-latitude [0, 180] degrees
        phi_deg = meta.phi      # geographic longitude [0, 360) degrees
        dec_deg = 90.0 - theta_deg   # declination (equatorial; same Z-axis)
        # Convert geographic longitude to equatorial RA using GMST at event time.
        # mirrors CWB skymap::phi2RA = fmod(phi_geo + GMST, 360)
        from pycwb.types.detector import gmst_accurate as _gmst_accurate
        _t_event_gps = float(meta.c_time if meta.c_time != 0.0 else cluster.cluster_time) + float(self.gps[0])
        _gmst_deg = np.degrees(_gmst_accurate(_t_event_gps)) % 360.0
        ra_deg = (phi_deg + _gmst_deg) % 360.0
        # [skymap_value, 0, equatorial, detector_frame]  — matches CWB output() layout:
        #   phi[0] = geographic phi;  phi[2] = equatorial RA
        self.theta = [theta_deg, 0.0, dec_deg, theta_deg]
        self.phi = [phi_deg, 0.0, ra_deg, phi_deg]

        # --- Event IDs and IFO list ---
        self.nevent = 1
        self.job_id = job_segment.index
        self.ifo_list = list(job_segment.ifos)
        # eventID[0]: sequential cluster ID within job; eventID[1]: 0 placeholder
        self.eventID = [getattr(cluster, 'cluster_id', 1), 0]

        # --- Super-lag shifts ---
        self.slag = list(getattr(job_segment, 'shift', [0.0] * n_ifo))

        # --- Timing and frequency per IFO (network-level in slot [0]) ---
        c_time = meta.c_time if meta.c_time != 0.0 else cluster.cluster_time
        c_freq = meta.c_freq if meta.c_freq != 0.0 else cluster.cluster_freq
        net_bw = cluster.high_frequency - cluster.low_frequency

        # Pre-extract core pixels (used below)
        all_pixels = cluster.pixels
        core_pixels = [p for p in all_pixels if p.core]

        # Sky-time delays: ml[i, l_max] are integer lag indices
        sky_td = list(cluster.sky_time_delay) if cluster.sky_time_delay else []
        td_rate = float(config.TDRate) if config is not None else 1.0
        # C++ lag correction: tau_i(theta,phi) - tau_0(theta,phi) in seconds
        # For IFO 0: no correction; for IFO i>0: (sky_td[i] - sky_td[0]) / TDRate
        tau_ref = float(sky_td[0]) / td_rate if sky_td else 0.0
        tau_ifo = [float(sky_td[i]) / td_rate if i < len(sky_td) else 0.0
                   for i in range(n_ifo)]
        # C++ lag: [pwc->shift]*n_ifo + [pd->lagShift.data[LAG]]*n_ifo
        # lagShift = data stream lag (0 for zero-lag); ToF delays only go into self.time.
        self.lag = [0.0] * (2 * n_ifo)

        # Time per IFO (mirrors C++ netevent.cc):
        #   time[0] = pcd->cTime + gps[0]                          (no delay for ref)
        #   time[i] = pcd->cTime + gps[i] + tau_i - tau_0          (ToF correction)
        self.time = [c_time + float(self.gps[i]) + (tau_ifo[i] - tau_ref if i > 0 else 0.0)
                     for i in range(n_ifo)]

        # Duration per IFO: min/max of per-IFO pixel time bins using p.data[i].index
        # (mirrors C++ pwc->get("start",i,'L',0) / pwc->get("stop",i,'L',0))
        # start = min of (1/rate * floor(data[i].index / layers))
        # stop  = max of (1/rate * (floor(data[i].index / layers) + 1))
        per_ifo_duration = []
        for ifo_idx in range(n_ifo):
            t_starts = []
            t_stops = []
            for p in core_pixels:
                if p.rate > 0 and p.layers > 0 and ifo_idx < len(p.data):
                    dt = 1.0 / float(p.rate)
                    time_bin = int(p.data[ifo_idx].index) // int(p.layers)
                    t_starts.append(dt * time_bin)
                    t_stops.append(dt * (time_bin + 1))
            if t_starts:
                per_ifo_duration.append(max(t_stops) - min(t_starts))
            else:
                per_ifo_duration.append(cluster.stop_time - cluster.start_time)

        # Frequency per IFO:
        # C++ sets frequency[i] = cFreq_net.data[kid] for all IFOs (WDM sub-bin likelihood-weighted mean),
        # then overwrites frequency[0] = pcd->cFreq (supercluster central freq from supercluster stage).
        # We compute the WDM sub-bin mean below (a_f/b_f) for slots [1+],
        # and use cluster.cluster_freq for slot [0].
        # Placeholder until the sub-bin accumulators are filled below:
        self.frequency = [c_freq] * n_ifo  # will be updated after sub-bin loop

        # Bandwidth per IFO: C++ uses high-low for all IFOs, overwrites [0] with energy-weighted RMS
        self.bandwidth = [net_bw] * n_ifo

        # Duration: per-IFO from pixel index ranges; [0] overwritten below with energy-weighted value
        self.duration = per_ifo_duration[:]

        # Compute energy-weighted duration (RMS) and bandwidth (RMS) for slot [0]
        # Mirrors C++ pwc->get("duration",0,'L',0) and pwc->get("bandwidth",0,'L',0)
        # Case 'D': a = sum(t*x), b = sum(x), d = sum(t^2*x); result = sqrt((d-a^2/b)/b) * b / b
        # = sqrt(variance) (actually it's the weighted RMS)
        if core_pixels:
            # Duration RMS
            a_t = b_t = d_t = 0.0
            a_f = b_f = d_f = 0.0
            max_rate = max((p.rate for p in core_pixels), default=0)
            min_rate = min((p.rate for p in core_pixels if p.rate > 0), default=1)
            for p in core_pixels:
                dt = 1.0 / float(p.rate) if p.rate > 0 else 0.0
                mm = int(p.layers)
                mp = int(max_rate * dt + 0.5) if dt > 0 else 1  # n sub-time bins
                # WDM: dT=0.5 bin offset (C++: dT = mm==mp ? 0 : 0.5)
                dT = 0.0 if mm == mp else 0.5
                # C++ uses INTEGER division: pList[M].time/mm (both size_t)
                time_bin = int(p.time) // mm
                iT = (float(time_bin) - dT) * dt
                n_sub = max(mp, 1)
                sub_dt = 1.0 / float(max_rate) if max_rate > 0 else dt
                iT += sub_dt / 2.0  # central bin
                x = p.likelihood / (n_sub * n_sub) if p.likelihood > 0 else 0.0
                for _ in range(n_sub):
                    a_t += iT * x
                    b_t += x
                    d_t += iT * iT * x
                    iT += sub_dt

                # Frequency sub-bins
                mp_f = int(1.0 / (float(min_rate) * dt) + 0.5) if dt > 0 else 1
                dF = 0.5  # WDM offset
                iF = (float(p.frequency) - dF) / dt / 2.0
                df = float(min_rate) / 2.0
                n_sub_f = max(mp_f, 1)
                iF += df / 2.0
                x_f = p.likelihood / (n_sub_f * n_sub_f) if p.likelihood > 0 else 0.0
                for _ in range(n_sub_f):
                    a_f += iF * x_f
                    b_f += x_f
                    d_f += iF * iF * x_f
                    iF += df

            if b_t > 0:
                var_t = (d_t - a_t * a_t / b_t) / b_t
                dur0 = float(np.sqrt(var_t) * b_t / b_t) if var_t > 0 else 1.0 / (max_rate if max_rate > 0 else 1.0)
                self.duration[0] = dur0
            if b_f > 0:
                var_f = (d_f - a_f * a_f / b_f) / b_f
                bw0 = float(np.sqrt(var_f) * b_f / b_f) if var_f > 0 else float(min_rate) / 2.0
                self.bandwidth[0] = bw0
                # cFreq_net = likelihood-weighted WDM sub-bin mean frequency (C++ case 'f' / 'L')
                # used for frequency[i>0]; frequency[0] is overwritten with c_freq (= meta.c_freq) below
                lh_freq_net = a_f / b_f
                self.frequency = [lh_freq_net] * n_ifo

        # frequency[0] = pcd->cFreq = Fo from likelihoodWP (MRA spectral centroid) = meta.c_freq
        self.frequency[0] = c_freq

        # --- Per-IFO amplitude fields from pixel data ---
        if config is not None:
            in_rate = float(config.inRate)
            # C++ get_SS()/get_GW()/get_NN() sum over ALL pixels, not just core.
            # Use core pixels only for noise_rms sampling (TF map lookup).
            all_pixels = cluster.pixels
            core_pixels = [p for p in all_pixels if p.core]

            # Prefer xtalk-corrected per-IFO energies from fill_detection_statistic
            # (mirrors C++ getMRAwave 'W'/'S' + get_XX()/get_SS()/get_XS()).
            # Fall back to diagonal pixel sums when not set (backwards compatibility).
            have_xtalk_snr = (len(meta.wave_snr) == n_ifo
                              and len(meta.signal_snr) == n_ifo
                              and len(meta.cross_snr) == n_ifo)

            for i in range(n_ifo):
                if have_xtalk_snr:
                    # Xtalk-corrected energies: mirrors C++ d->enrg, d->sSNR, d->xSNR
                    wave_sq_xt  = float(meta.wave_snr[i])    # C++ d->enrg = get_XX()
                    asnr_sq_xt  = float(meta.signal_snr[i])  # C++ d->sSNR = get_SS()
                    xsnr_sq_xt  = float(meta.cross_snr[i])   # C++ d->xSNR = get_XS()
                    self.snr.append(wave_sq_xt)
                    self.sSNR.append(asnr_sq_xt)
                    self.xSNR.append(xsnr_sq_xt)
                    self.nill.append(float(xsnr_sq_xt - asnr_sq_xt))  # C++ pd.xSNR - pd.sSNR
                    # Use per-IFO null from cluster metadata (C++ pd.null)
                    if hasattr(meta, 'null_energy') and len(meta.null_energy) > i:
                        null_val = float(meta.null_energy[i])
                    else:
                        # Fallback: split total null evenly (backwards compatibility)
                        null_val = sum(p.null for p in all_pixels) / n_ifo
                    self.null.append(null_val)
                    # hrss from physical strain energy (not whitened)
                    if hasattr(meta, 'signal_energy_physical') and len(meta.signal_energy_physical) > i:
                        hrss_sq_physical = float(meta.signal_energy_physical[i])
                    else:
                        # Fallback: whitened energy (will be wrong units but backwards compatible)
                        hrss_sq_physical = asnr_sq_xt
                    self.hrss.append(float(np.sqrt(hrss_sq_physical / in_rate)))
                else:
                    # Fallback: diagonal pixel sum (no xtalk cross-terms)
                    asnr_sq = sum(p.data[i].asnr ** 2 + p.data[i].a_90 ** 2 for p in all_pixels)
                    wave_sq = sum(p.data[i].wave ** 2 + p.data[i].w_90 ** 2 for p in all_pixels)
                    self.snr.append(float(wave_sq))
                    self.sSNR.append(float(asnr_sq))
                    self.xSNR.append(float(np.sqrt(max(wave_sq * asnr_sq, 0.0))))
                    xsnr_sq = float(np.sqrt(max(wave_sq * asnr_sq, 0.0)))
                    self.nill.append(float(xsnr_sq - asnr_sq))
                    # null: split total evenly
                    null_val = sum(p.null for p in all_pixels) / n_ifo
                    self.null.append(null_val)
                    # hrss from pixel-based signal energy (whitened amplitudes * noise_rms)
                    hrss_sq = sum((p.data[i].asnr * p.data[i].noise_rms) ** 2 
                                  + (p.data[i].a_90 * p.data[i].noise_rms) ** 2 for p in all_pixels)
                    self.hrss.append(float(np.sqrt(hrss_sq / in_rate)))

                # Noise floor: RMS of noiserms across core pixels for this IFO, scaled to strain/rtHz
                # C++ get("noise",i,'S',0): sum += r*r; sum/=mp → log10(sqrt(sum)) → pow(10,...)/sqrt(inRate)
                # = sqrt(mean(noiserms^2)) / sqrt(inRate)
                rms_vals = [p.data[i].noise_rms for p in core_pixels
                            if p.data[i].noise_rms > 0]
                mean_nrms = float(np.sqrt(np.mean(np.array(rms_vals) ** 2))) if rms_vals else 1.0
                # Convert from TF-domain noise amplitude to strain/rtHz: divide by sqrt(inRate)
                self.noise.append(float(mean_nrms / np.sqrt(in_rate)))

            # erA: per-IFO sky error region from l_max (placeholder 11-element zeros)
            self.erA = [list(np.zeros(11)) for _ in range(n_ifo)]

            # Strain = sqrt(sum(hrss^2))  [C++ takes sqrt at end of loop]
            self.strain = [float(np.sqrt(sum(h ** 2 for h in self.hrss[:n_ifo])))]

            # Antenna patterns at best-fit sky location
            # C++ uses detector::antenna(theta, phi, psi) in GEOGRAPHIC coordinates:
            #   theta = co-latitude (0-180 deg), phi = longitude (0-360 deg)
            # These are CWB's meta.theta / meta.phi (Earth-fixed frame, NOT equatorial RA/Dec).
            # Formula: fp = -(a·D·a - b·D·b), fx = 2*(a·D·b), return (fp/2, fx/2)
            # where D = Det tensor (Ex⊗Ex - Ey⊗Ey, no 1/2), a=e_theta, b=e_phi (geographic).
            # Python det.response = 0.5*(x⊗x - y⊗y), so already carries the /2:
            #   fp/2 = -(a·D_py·a - b·D_py·b), fx/2 = 2*(a·D_py·b)
            try:
                from pycwb.types.detector import Detector
                theta_geo = float(np.radians(theta_deg))  # C++ theta (geographic co-latitude)
                phi_geo = float(np.radians(phi_deg))      # C++ phi (geographic longitude)
                psi_rad = float(np.radians(meta.psi)) if hasattr(meta, 'psi') else 0.0
                cT = np.cos(theta_geo); sT = np.sin(theta_geo)
                cP = np.cos(phi_geo);  sP = np.sin(phi_geo)
                # Polarization basis vectors in geographic Cartesian frame
                e_th = np.array([cT * cP, cT * sP, -sT])  # e_theta (C++ a)
                e_ph = np.array([-sP, cP, 0.0])            # e_phi   (C++ b)
                for ifo in job_segment.ifos:
                    det = Detector(ifo)
                    D = det.response  # 0.5*(x⊗x - y⊗y), respects the C++ /2 factor
                    Da = D @ e_th
                    Db = D @ e_ph
                    f_plus  = np.dot(e_th, Da) - np.dot(e_ph, Db)   # a·D·a - b·D·b (= fp_C++/2)
                    f_cross = 2.0 * np.dot(e_th, Db)                 # 2*(a·D·b)      (= fx_C++/2)
                    # C++ convention: fp = -fp (LIGO-T010110), before psi rotation
                    fp = -f_plus
                    fx = f_cross
                    if abs(psi_rad) > 1e-15:
                        a_rot = fp * np.cos(2 * psi_rad) + fx * np.sin(2 * psi_rad)
                        b_rot = -fp * np.sin(2 * psi_rad) + fx * np.cos(2 * psi_rad)
                        fp, fx = a_rot, b_rot
                    self.bp.append(float(fp))
                    self.bx.append(float(fx))
            except Exception:
                self.bp = [0.0] * n_ifo
                self.bx = [0.0] * n_ifo

            # Pixel rate per IFO (modal rate of core pixels)
            core_rates = [p.rate for p in core_pixels]
            if core_rates:
                from statistics import mode as stat_mode
                try:
                    modal_rate = float(stat_mode(core_rates))
                except Exception:
                    modal_rate = float(core_rates[0])
            else:
                modal_rate = 0.0
            self.rate = [modal_rate] * n_ifo

        self.id = self.long_id


    def summary(self):
        """
        Return a summary of the event for building the catalog

        :param job_id: Job ID
        :param id: Event ID
        :return: Summary of the event
        :rtype: dict
        """
        return {
            "job_id": self.job_id,
            "id": self.long_id,
            "ifo": self.ifo_list,
            "nevent": self.nevent,
            "rho": self.rho[0],
            "lag": self.lag[0],
            "slag": self.slag,
            "start": self.start,
            "stop": self.stop,
            "low": self.low,
            "high": self.high,
            "sSNR": self.sSNR,
            "hrss": self.hrss,
            "phi": self.phi,
            "theta": self.theta,
            "psi": self.psi,
            "injection": self.injection,
        }

    @property
    def hash_id(self):
        """
        Return a hash ID of the event

        :return: Hash ID of the event
        :rtype: str
        """
        hash_object = hashlib.md5()
        hash_object.update(f"{self.start[0]}_{self.stop[0]}_{self.low[0]}_{self.high[0]}".encode("utf-8"))  # Encoding the string to bytes
        return hash_object.hexdigest()[-10:]

    @property
    def long_id(self):
        """
        Return a long ID of the event

        :return: Long ID of the event
        :rtype: str
        """
        if len(self.stop) == 0:
            return "unknown"
        return f"{self.stop[0]}_{self.hash_id}"

    def dump(self):
        """
        Return a string representation of the event

        :return: String representation of the instance
        :rtype: str
        """
        return f"""
nevent: \t\t {self.nevent}
ndim: \t\t {self.ndim}
run: \t\t {self.run}
rho: \t\t {self.rho[0]}
netCC: \t\t {self.netcc[0]}
netED: \t\t {self.neted}
penalty: \t\t {self.penalty}
gnet: \t\t {self.gnet}
anet: \t\t {self.anet}
inet: \t\t {self.inet}
likelihood: \t\t {self.likelihood}
ecor: \t\t {self.ecor}
ECOR: \t\t {self.ECOR}
factor: \t\t {self.factor}
range: \t\t {self.range}
mchirp: \t\t {self.chirp}
norm: \t\t {self.norm}
usize: \t\t {self.usize}

ifo: \t\t {self.ifo_list}
eventID: \t\t {self.eventID}
rho: \t\t {self.rho}
type: \t\t {self.type}
rate: \t\t {self.rate}
volume: \t\t {self.volume}
size: \t\t {self.size}
lag: \t\t {self.lag}
slag: \t\t {self.slag}
phi: \t\t {self.phi}
theta: \t\t {self.theta}
psi: \t\t {self.psi}
iota: \t\t {self.iota}
bp: \t\t {self.bp}
bx: \t\t {self.bx}
chirp: \t\t {self.chirp}
range: \t\t {self.range}
Deff: \t\t {self.Deff}
mass: \t\t {self.mass}
spin: \t\t {self.spin}
eBBH: \t\t {self.eBBH}
null: \t\t {self.null}
strain: \t\t {self.strain}
hrss: \t\t {self.hrss}
noise: \t\t {self.noise}

start: \t\t {self.start}
stop: \t\t {self.stop}
left: \t\t {self.left}
right: \t\t {self.right}
duration: \t\t {self.duration}
frequency: \t\t {self.frequency}
low: \t\t {self.low}
high: \t\t {self.high}
bandwidth: \t\t {self.bandwidth}

snr: \t\t {self.snr}
xSNR: \t\t {self.xSNR}
sSNR: \t\t {self.sSNR}
iSNR: \t\t {self.iSNR}
oSNR: \t\t {self.oSNR}
ioSNR: \t\t {self.ioSNR}

netcc: \t\t {self.netcc}
neted: \t\t {self.neted}
        """
