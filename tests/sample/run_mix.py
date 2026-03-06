import sys
import time
import psutil
from time import perf_counter

sys.path.insert(0, "../..")

import logging
from pycwb.config import Config
from pycwb.modules.cwb_coherence.coherence import cluster_pixels, compute_threshold, select_network_pixels
from pycwb.modules.read_data.data_check import check_and_resample
from pycwb.config import Config
from pycwb.modules.logger import logger_init
from pycwb.config import Config
from pycwb.modules.catalog import add_events_to_catalog
from pycwb.modules.super_cluster.super_cluster import supercluster_wrapper
from pycwb.modules.super_cluster.supercluster import supercluster
from pycwb.modules.xtalk.monster import load_catalog
from pycwb.modules.coherence.coherence import coherence
from pycwb.modules.read_data import generate_injections, generate_noise_for_job_seg, read_from_job_segment, check_and_resample
from pycwb.modules.data_conditioning import data_conditioning
from pycwb.modules.likelihoodWP.likelihood import likelihood
from pycwb.modules.qveto.qveto import get_qveto
from pycwb.types.job import WaveSegment
from pycwb.types.network import Network
from pycwb.modules.workflow_utils.job_setup import print_job_info
from pycwb.utils.dataclass_object_io import save_dataclass_to_json
from pycwb.workflow.subflow.postprocess_and_plots import plot_trigger_flow, reconstruct_waveforms_flow, plot_skymap_flow
from pycwb.modules.cwb_conversions import convert_fragment_clusters_to_netcluster, convert_netcluster_to_fragment_clusters
from pycwb.types.network_event import Event
import numpy as np

logger_init()

logger = logging.getLogger(__name__)

config = Config()
config.load_from_yaml('./user_parameters_injection.yaml')
config.nproc = 1

from pycwb.modules.read_data import generate_injection, generate_noise_for_job_seg
from pycwb.modules.job_segment import create_job_segment_from_config

job_segments = create_job_segment_from_config(config)

data = generate_noise_for_job_seg(job_segments[0], config.inRate, f_low=config.fLow)
data = generate_injection(config, job_segments[0], data)

from pycwb.modules.data_conditioning import data_conditioning as data_conditioning_cwb


data = [check_and_resample(data[i], config, i) for i in range(len(job_segments[0].ifos))]

tf_maps, nRMS_list = data_conditioning_cwb(config, data)

# initialize network object 
network = Network(config, tf_maps, nRMS_list)

# coherence
fragment_clusters = coherence(config, tf_maps, nRMS_list, net=network)
logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

# supercluster
super_fragment_clusters = supercluster(config, network, fragment_clusters, tf_maps)

for lag, fragment_cluster in enumerate(super_fragment_clusters):
    print(f"Processing lag {lag+1}/{len(super_fragment_clusters)}")
    for k, selected_cluster in enumerate(fragment_cluster.clusters):
        print(f"  Processing cluster {k+1}/{len(fragment_cluster.clusters)}")
        # skip if cluster is already rejected
        if selected_cluster.cluster_status > 0:
            continue

        cluster_id = k + 1

        wdm_list = network.get_wdm_list()
        for wdm in wdm_list:
            wdm.setTDFilter(config.TDSize, config.upTDF)

        # load delay index
        network.set_delay_index(config.TDRate)                
        pwc = network.get_cluster(lag)


        # load time delay data
        pwc.cpf(convert_fragment_clusters_to_netcluster(fragment_cluster.dump_cluster(k)), False)
        pwc.setcore(False, 1)
        pwc.loadTDampSSE(network.net, 'a', config.BATCH, config.BATCH)
        cluster = convert_netcluster_to_fragment_clusters(pwc)

        # Compute likelihood using old ROOT code
        network.net.MRA = True
        start_time = perf_counter()
        selected_core_pixels = network.likelihoodWP(config.search, lag, config.Search)
        cluster_old = convert_netcluster_to_fragment_clusters(network.get_cluster(lag)).clusters[0]
        if cluster_old.cluster_status > 0:
            logger.info(f"Cluster {k} at lag {lag} rejected by old likelihood computation.")
        end_time = perf_counter()

        # extract event information using old ROOT code
        event = Event()
        event.output(network.net, 1, lag, shifts=job_segments[0].shift)
        pwc.clean(1)
        pwc.clear()

        # Extract sky delay/pattern arrays from the ROOT network object
        from pycwb.modules.cwb_conversions.series import convert_wavearray_to_nparray
        ml = np.array([convert_wavearray_to_nparray(network.get_ifo(i).index, short=True) for i in range(config.nIFO)])
        FP = np.array([convert_wavearray_to_nparray(network.get_ifo(i).fp) for i in range(config.nIFO)])
        FX = np.array([convert_wavearray_to_nparray(network.get_ifo(i).fx) for i in range(config.nIFO)])

        # Compute likelihood using new python code
        start_time_new = perf_counter()
        cluster, _ = likelihood(config.nIFO, cluster.clusters[0], config.MRAcatalog, ml=ml, FP=FP, FX=FX, config=config)
        event2 = Event()
        event2.output_py(job_segments[0], cluster, config)
        end_time_new = perf_counter()
        if cluster is None:
            logger.info(f"Cluster {k} at lag {lag} rejected by new likelihood computation.")
            continue
        # compare the results
        print(f"[old] start: {event.left[0]}, stop: {event.stop[0] - event.gps[0]}, low_freq: {event.low[0]}, high_freq: {event.high[0]}")
        print(f"[new] start: {cluster.start_time}, stop: {cluster.stop_time}, low_freq: {cluster.low_frequency}, high_freq: {cluster.high_frequency}")
        print(f"[old] ecor: {event.ecor:.6f}, subnet: {event.netcc[2]:.6f}, SUBNET: {event.netcc[3]:.6f}, rho0: {event.rho[0]:.6f}, rho1: {event.rho[1]:.6f}")
        print(f"[new] ecor: {cluster.cluster_meta.net_ecor:.6f}, subnet: {cluster.cluster_meta.sub_net:.6f}, SUBNET: {cluster.cluster_meta.sub_net2:.6f}, rho0: {cluster.cluster_meta.net_rho:.6f}, rho1: {cluster.cluster_meta.net_rho2:.6f}")
        print(f"[old] a_net: {event.anet:.6f}, g_net: {event.gnet:.6f}, i_net: {event.inet:.6f}")
        print(f"[new] a_net: {cluster.cluster_meta.a_net:.6f}, g_net: {cluster.cluster_meta.g_net:.6f}, i_net: {cluster.cluster_meta.i_net:.6f}")
        print(f"[old] neted[0]: {event.neted[0]:.6f}, neted[3]: {event.neted[3]:.6f}")
        print(f"[new] neted[0]: {cluster.cluster_meta.net_ed:.6f}, neted[3]: {cluster.cluster_meta.like_sky:.6f}")
        print(f"[old] neted[1]: {event.neted[1]:.6f}, neted[2]: {event.neted[2]:.6f}, neted[4]: {event.neted[4]:.6f}")
        print(f"[new] neted[1]: {cluster.cluster_meta.net_null:.6f}, neted[2]: {cluster.cluster_meta.energy:.6f}, neted[4]: {cluster.cluster_meta.energy_sky:.6f}")
        print(f"[new-debug] Gn={cluster.cluster_meta.g_noise:.6f}, Dc={cluster.cluster_meta.net_ed - cluster.cluster_meta.net_null + cluster.cluster_meta.ndof * 2:.6f}, N_eff={cluster.cluster_meta.ndof:.1f}")
        print(f"[new-debug] snr_ifo={[f'{v:.4f}' for v in cluster.cluster_meta.wave_snr]}")
        print(f"[new-debug] sSNR_ifo={[f'{v:.4f}' for v in cluster.cluster_meta.signal_snr]}")
        print(f"[new-debug] Ew_wf={sum(cluster.cluster_meta.wave_snr):.4f}, Lw={cluster.cluster_meta.like_net:.4f}")
        print(f"[old] snr:  {[f'{v:.6f}' for v in event.snr]}")
        print(f"[new] snr:  {[f'{v:.6f}' for v in event2.snr]}")
        print(f"[old] sSNR: {[f'{v:.6f}' for v in event.sSNR]}")
        print(f"[new] sSNR: {[f'{v:.6f}' for v in event2.sSNR]}")
        print(f"[old] xSNR: {[f'{v:.6f}' for v in event.xSNR]}")
        print(f"[new] xSNR: {[f'{v:.6f}' for v in event2.xSNR]}")
        print(f"[old] skysize: {event.size[0]}, {event.size[1]}")
        print(f"[new] skysize: {cluster.get_core_size()}, {cluster.cluster_meta.sky_size}")
        print(f"[old] hrss: {[f'{v:.6e}' for v in event.hrss]}")
        print(f"[new] hrss: {[f'{v:.6e}' for v in event2.hrss]}")
        print(f"[old] strain: {event.strain[0]:.6f}, phi: {event.phi[0]:.4f}, theta: {event.theta[0]:.4f}")
        print(f"[new] strain: {event2.strain[0]:.6f}, phi: {event2.phi[0]:.4f}, theta: {event2.theta[0]:.4f}")
        print("likelihood pixels: ", len([p for p in cluster.pixels if p.likelihood > 0]))
        print("core pixels: ", len([p for p in cluster.pixels if p.core]))
        # Compare strain values with scientific notation (can be very small, e.g. 1e-44)
        print(f"[old] strain: {[f'{v:.6e}' for v in event.strain]}")
        print(f"[new] strain: {[f'{v:.6e}' for v in event2.strain]}")
        strain_rel_err = [abs(event.strain[i] - event2.strain[i]) / abs(event.strain[i]) if abs(event.strain[i]) > 1e-50 else abs(event.strain[i] - event2.strain[i]) 
                          for i in range(len(event.strain))]
        print(f"[strain] relative errors: {[f'{v:.6e}' for v in strain_rel_err]}")
        # snr/sSNR/xSNR now use get_MRA_wave (exact WDM reconstruction equivalent to C++),
        # so they should match to floating-point precision (default rtol=1e-5).
        snr_rtol = 1e-4  # small tolerance for float32/float64 accumulation differences
        if np.isclose(event.left[0], cluster.start_time) and \
            np.isclose(event.stop[0] - event.gps[0], cluster.stop_time) and \
            np.isclose(event.low[0], cluster.low_frequency) and \
            np.isclose(event.high[0], cluster.high_frequency) and \
            np.isclose(event.ecor, cluster.cluster_meta.net_ecor) and \
            np.isclose(event.netcc[2], cluster.cluster_meta.sub_net) and \
            np.isclose(event.netcc[3], cluster.cluster_meta.sub_net2) and \
            np.isclose(event.rho[0], cluster.cluster_meta.net_rho) and \
            np.isclose(event.rho[1], cluster.cluster_meta.net_rho2) and \
            np.isclose(event.neted[0], cluster.cluster_meta.net_ed) and \
            np.isclose(event.neted[3], cluster.cluster_meta.like_sky) and \
            all(np.isclose(event.snr[i], event2.snr[i], rtol=snr_rtol) for i in range(config.nIFO)) and \
            all(np.isclose(event.sSNR[i], event2.sSNR[i], rtol=snr_rtol) for i in range(config.nIFO)) and \
            all(np.isclose(event.xSNR[i], event2.xSNR[i], rtol=snr_rtol) for i in range(config.nIFO)) and \
            all(np.isclose(event.strain[i], event2.strain[i], rtol=snr_rtol, atol=1e-50) for i in range(len(event.strain))) and \
            np.isclose(event.anet, cluster.cluster_meta.a_net) and \
            np.isclose(event.gnet, cluster.cluster_meta.g_net) and \
            np.isclose(event.inet, cluster.cluster_meta.i_net) and \
            event.size[0] == cluster.get_core_size() and \
            event.size[1] == cluster.cluster_meta.sky_size:
            print("✅ Results match between old and new likelihood code.")
        else:
            print("❌ Results do NOT match between old and new likelihood code.")
