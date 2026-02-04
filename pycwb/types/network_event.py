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

        self.id = self.long_id
    # def json(self):
    #     """
    #     Return a JSON representation of the event
    #
    #     :return: JSON format of the instance
    #     :rtype: str
    #     """
    #     return json.dumps(self.__dict__)

    def output_py(self, job_segment: WaveSegment, cluster: Cluster):
        #    print(f"[old] start: {event.left[0]}, stop: {event.stop[0] - event.gps[0]}, low_freq: {event.low[0]}, high_freq: {event.high[0]}")
        #         print(f"[new] start: {cluster.start_time}, stop: {cluster.stop_time}, low_freq: {cluster.low_frequency}, high_freq: {cluster.high_frequency}")
        #         print(f"[old] ecor: {event.ecor:.6f}, subnet: {event.netcc[2]:.6f}, SUBNET: {event.netcc[3]:.6f}, rho0: {event.rho[0]:.6f}")
        #         print(f"[new] ecor: {cluster.cluster_meta.net_ecor:.6f}, subnet: {cluster.cluster_meta.sub_net:.6f}, SUBNET: {cluster.cluster_meta.sub_net2:.6f}, rho0: {cluster.cluster_meta.net_rho:.6f}")
        #         print(f"[old] a_net: {event.anet:.6f}, g_net: {event.gnet:.6f}, i_net: {event.inet:.6f}")
        #         print(f"[new] a_net: {cluster.cluster_meta.a_net:.6f}, g_net: {cluster.cluster_meta.g_net:.6f}, i_net: {cluster.cluster_meta.i_net:.6f}")
        #         print(f"[old] skysize: {event.size[0]}, {event.size[1]}")
        #         print(f"[new] skysize: {cluster.get_core_size()}, {cluster.cluster_meta.sky_size}")
        self.gps = np.array([job_segment.physical_start_times[ifo] for ifo in job_segment.ifos])
        self.left = cluster.start_time
        self.right = job_segment.duration - cluster.stop_time
        self.start = cluster.start_time + self.gps
        self.stop = cluster.stop_time + self.gps
        self.low = cluster.low_frequency
        self.high = cluster.high_frequency
        self.ecor = cluster.cluster_meta.net_ecor
        self.netcc = [None, None, cluster.cluster_meta.sub_net, cluster.cluster_meta.sub_net2]
        self.rho = [cluster.cluster_meta.net_rho]
        self.anet = cluster.cluster_meta.a_net
        self.gnet = cluster.cluster_meta.g_net
        self.inet = cluster.cluster_meta.i_net
        self.size = [cluster.get_core_size(), cluster.cluster_meta.sky_size]
        self.nevent = 1
        self.job_id = job_segment.index


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
