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
        cluster, skymap_stats = likelihood(config.nIFO, cluster.clusters[0], config.MRAcatalog, ml=ml, FP=FP, FX=FX, config=config)
        event2 = Event()
        event2.output_py(job_segments[0], cluster, config)
        end_time_new = perf_counter()
        if cluster is None:
            logger.info(f"Cluster {k} at lag {lag} rejected by new likelihood computation.")
            continue

        # --- Sky pixel mismatch diagnostics ---
        from pycwb.types.detector import _build_sky_directions
        _healpix_order = int(getattr(config, 'healpix', 0)) if hasattr(config, 'healpix') else None
        n_sky_diag = int(ml.shape[1])
        _ra_diag, _dec_diag = _build_sky_directions(n_sky_diag, _healpix_order)
        _phi_arr_deg = np.degrees(_ra_diag) % 360.0
        _theta_arr_deg = (90.0 - np.degrees(_dec_diag)) % 180.0
        l_max_py = int(cluster.cluster_meta.l_max)
        l_max_py_sky = skymap_stats.nSkyStat[l_max_py]
        # Find top-5 nSkyStat indices to compare with C++
        top5_idx = np.argsort(skymap_stats.nSkyStat)[-5:][::-1]
        print(f"[sky-debug] Python l_max={l_max_py}: phi={_phi_arr_deg[l_max_py]:.4f} theta={_theta_arr_deg[l_max_py]:.4f} nSkyStat={l_max_py_sky:.6f}")
        print(f"[sky-debug] Top-5 sky pixels by nSkyStat:")
        for idx in top5_idx:
            print(f"  l={idx}: phi={_phi_arr_deg[idx]:.4f} theta={_theta_arr_deg[idx]:.4f} nSkyStat={skymap_stats.nSkyStat[idx]:.6f}")
        # Find all pixels within 2 degrees of the C++ phi/theta
        cpp_phi = event.phi[0]
        cpp_theta = event.theta[0]
        print(f"[sky-debug] C++ selected: phi={cpp_phi:.4f} theta={cpp_theta:.4f}")
        # Find the sky pixel closest to C++ phi/theta
        diffs = ((_phi_arr_deg - cpp_phi + 180) % 360 - 180)**2 + (_theta_arr_deg - cpp_theta)**2
        l_max_cpp_approx = int(np.argmin(diffs))
        print(f"[sky-debug] Closest Python pixel to C++ sky: l={l_max_cpp_approx}: phi={_phi_arr_deg[l_max_cpp_approx]:.4f} theta={_theta_arr_deg[l_max_cpp_approx]:.4f} nSkyStat={skymap_stats.nSkyStat[l_max_cpp_approx]:.6f}")
        print(f"[sky-debug] nSkyStat diff: py_max={skymap_stats.nSkyStat[l_max_py]:.8f} vs cpp_approx={skymap_stats.nSkyStat[l_max_cpp_approx]:.8f} (diff={skymap_stats.nSkyStat[l_max_py]-skymap_stats.nSkyStat[l_max_cpp_approx]:.2e})")
        # --- End sky diagnostics ---

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
        print(f"[old] noise: {[f'{v:.6e}' for v in event.noise]}")
        print(f"[new] noise: {[f'{v:.6e}' for v in event2.noise]}")
        print(f"[old] null: {[f'{v:.6e}' for v in event.null]}")
        print(f"[new] null: {[f'{v:.6e}' for v in event2.null]}")
        print(f"[old] nill: {[f'{v:.6e}' for v in event.nill]}")
        print(f"[new] nill: {[f'{v:.6e}' for v in event2.nill]}")
        print(f"[old] bp: {[f'{v:.6f}' for v in event.bp]}")
        print(f"[new] bp: {[f'{v:.6f}' for v in event2.bp]}")
        print(f"[old] bx: {[f'{v:.6f}' for v in event.bx]}")
        print(f"[new] bx: {[f'{v:.6f}' for v in event2.bx]}")
        print(f"[old] time: {[f'{v:.6f}' for v in event.time]}")
        print(f"[new] time: {[f'{v:.6f}' for v in event2.time]}")
        print(f"[old] duration: {[f'{v:.6f}' for v in event.duration]}")
        print(f"[new] duration: {[f'{v:.6f}' for v in event2.duration]}")
        print(f"[old] frequency: {[f'{v:.6f}' for v in event.frequency]}")
        print(f"[new] frequency: {[f'{v:.6f}' for v in event2.frequency]}")
        print(f"[old] bandwidth: {[f'{v:.6f}' for v in event.bandwidth]}")
        print(f"[new] bandwidth: {[f'{v:.6f}' for v in event2.bandwidth]}")
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
        hrss_rel_err = [abs(event.hrss[i] - event2.hrss[i]) / abs(event.hrss[i]) if abs(event.hrss[i]) > 1e-50 else abs(event.hrss[i] - event2.hrss[i]) 
                        for i in range(min(len(event.hrss), len(event2.hrss)))]
        print(f"[hrss] relative errors: {[f'{v:.6e}' for v in hrss_rel_err]}")
        # snr/sSNR/xSNR now use get_MRA_wave (exact WDM reconstruction equivalent to C++),
        # so they should match to floating-point precision (default rtol=1e-5).
        snr_rtol = 1e-4  # small tolerance for float32/float64 accumulation differences
        # null and nill rely on MRA waveform reconstruction energy differences.
        # Python avx_setAMP_ps runs in float64 while C++ _avx_setAMP_ps uses float32 SSE;
        # this causes tiny waveform reconstruction differences that are amplified in:
        #   null  (= sum((data-signal)²)), which is tiny relative to signal energy → ~1e-3 rel err
        #   nill  (= xSNR - sSNR), catastrophic cancellation (xSNR ≈ sSNR ~ 806) → ~1e-2 rel err
        null_rtol = 1e-3   # limited by float32 vs float64 in avx_setAMP_ps waveform reconstruction
        nill_rtol = 1e-2   # limited by catastrophic cancellation in xSNR-sSNR (≈ 0.042 vs 806)
        # Debug: print each comparison result
        checks = {
            'start': np.isclose(event.left[0], cluster.start_time),
            'stop': np.isclose(event.stop[0] - event.gps[0], cluster.stop_time),
            'low_freq': np.isclose(event.low[0], cluster.low_frequency),
            'high_freq': np.isclose(event.high[0], cluster.high_frequency),
            'ecor': np.isclose(event.ecor, cluster.cluster_meta.net_ecor),
            'sub_net': np.isclose(event.netcc[2], cluster.cluster_meta.sub_net),
            'sub_net2': np.isclose(event.netcc[3], cluster.cluster_meta.sub_net2),
            'rho0': np.isclose(event.rho[0], cluster.cluster_meta.net_rho),
            'rho1': np.isclose(event.rho[1], cluster.cluster_meta.net_rho2),
            'net_ed': np.isclose(event.neted[0], cluster.cluster_meta.net_ed, rtol=snr_rtol),
            'like_sky': np.isclose(event.neted[3], cluster.cluster_meta.like_sky),
            'snr': all(np.isclose(event.snr[i], event2.snr[i], rtol=snr_rtol) for i in range(config.nIFO)),
            'sSNR': all(np.isclose(event.sSNR[i], event2.sSNR[i], rtol=snr_rtol) for i in range(config.nIFO)),
            'xSNR': all(np.isclose(event.xSNR[i], event2.xSNR[i], rtol=snr_rtol) for i in range(config.nIFO)),
            'hrss': all(np.isclose(event.hrss[i], event2.hrss[i], rtol=snr_rtol, atol=1e-50) for i in range(config.nIFO)),
            'noise': all(np.isclose(event.noise[i], event2.noise[i], rtol=snr_rtol) for i in range(config.nIFO)),
            'null': all(np.isclose(event.null[i], event2.null[i], rtol=null_rtol, atol=1e-10) for i in range(config.nIFO)),
            'nill': all(np.isclose(event.nill[i], event2.nill[i], rtol=nill_rtol, atol=1e-10) for i in range(config.nIFO)),
            'bp': all(np.isclose(event.bp[i], event2.bp[i], rtol=snr_rtol) for i in range(config.nIFO)),
            'bx': all(np.isclose(event.bx[i], event2.bx[i], rtol=snr_rtol) for i in range(config.nIFO)),
            'time': all(np.isclose(event.time[i], event2.time[i], rtol=1e-5) for i in range(config.nIFO)),
            'duration': all(np.isclose(event.duration[i], event2.duration[i], rtol=1e-5) for i in range(config.nIFO)),
            'frequency': all(np.isclose(event.frequency[i], event2.frequency[i], rtol=1e-5) for i in range(config.nIFO)),
            'bandwidth': all(np.isclose(event.bandwidth[i], event2.bandwidth[i], rtol=1e-5) for i in range(config.nIFO)),
            'strain': all(np.isclose(event.strain[i], event2.strain[i], rtol=snr_rtol, atol=1e-50) for i in range(len(event.strain))),
            'anet': np.isclose(event.anet, cluster.cluster_meta.a_net),
            'gnet': np.isclose(event.gnet, cluster.cluster_meta.g_net),
            'inet': np.isclose(event.inet, cluster.cluster_meta.i_net),
            'core_size': event.size[0] == cluster.get_core_size(),
            'sky_size': event.size[1] == cluster.cluster_meta.sky_size,
        }
        failed = [k for k, v in checks.items() if not v]
        if failed:
            print(f"[diag] Failing checks: {failed}")
            for k in failed:
                if k in ('anet', 'gnet', 'inet'):
                    print(f"  {k}: old={getattr(event, k):.10f} new={getattr(cluster.cluster_meta, {'anet':'a_net','gnet':'g_net','inet':'i_net'}[k]):.10f}")
                elif k in ('net_ed', 'like_sky'):
                    field = 'net_ed' if k == 'net_ed' else 'like_sky'
                    print(f"  {k}: old={event.neted[0 if k=='net_ed' else 3]:.10f} new={getattr(cluster.cluster_meta, field):.10f}")
                elif k == 'bp':
                    for i in range(config.nIFO):
                        rel = abs(event.bp[i] - event2.bp[i]) / abs(event.bp[i]) if event.bp[i] != 0 else abs(event.bp[i] - event2.bp[i])
                        print(f"  bp[{i}]: old={event.bp[i]:.12f} new={event2.bp[i]:.12f} rel={rel:.3e}")
                elif k == 'bx':
                    for i in range(config.nIFO):
                        rel = abs(event.bx[i] - event2.bx[i]) / abs(event.bx[i]) if event.bx[i] != 0 else abs(event.bx[i] - event2.bx[i])
                        print(f"  bx[{i}]: old={event.bx[i]:.12f} new={event2.bx[i]:.12f} rel={rel:.3e}")
        if np.isclose(event.left[0], cluster.start_time) and \
            np.isclose(event.stop[0] - event.gps[0], cluster.stop_time) and \
            np.isclose(event.low[0], cluster.low_frequency) and \
            np.isclose(event.high[0], cluster.high_frequency) and \
            np.isclose(event.ecor, cluster.cluster_meta.net_ecor) and \
            np.isclose(event.netcc[2], cluster.cluster_meta.sub_net) and \
            np.isclose(event.netcc[3], cluster.cluster_meta.sub_net2) and \
            np.isclose(event.rho[0], cluster.cluster_meta.net_rho) and \
            np.isclose(event.rho[1], cluster.cluster_meta.net_rho2) and \
            np.isclose(event.neted[0], cluster.cluster_meta.net_ed, rtol=snr_rtol) and \
            np.isclose(event.neted[3], cluster.cluster_meta.like_sky) and \
            all(np.isclose(event.snr[i], event2.snr[i], rtol=snr_rtol) for i in range(config.nIFO)) and \
            all(np.isclose(event.sSNR[i], event2.sSNR[i], rtol=snr_rtol) for i in range(config.nIFO)) and \
            all(np.isclose(event.xSNR[i], event2.xSNR[i], rtol=snr_rtol) for i in range(config.nIFO)) and \
            all(np.isclose(event.hrss[i], event2.hrss[i], rtol=snr_rtol, atol=1e-50) for i in range(config.nIFO)) and \
            all(np.isclose(event.noise[i], event2.noise[i], rtol=snr_rtol) for i in range(config.nIFO)) and \
            all(np.isclose(event.null[i], event2.null[i], rtol=null_rtol, atol=1e-10) for i in range(config.nIFO)) and \
            all(np.isclose(event.nill[i], event2.nill[i], rtol=nill_rtol, atol=1e-10) for i in range(config.nIFO)) and \
            all(np.isclose(event.bp[i], event2.bp[i], rtol=snr_rtol) for i in range(config.nIFO)) and \
            all(np.isclose(event.bx[i], event2.bx[i], rtol=snr_rtol) for i in range(config.nIFO)) and \
            all(np.isclose(event.time[i], event2.time[i], rtol=1e-5) for i in range(config.nIFO)) and \
            all(np.isclose(event.duration[i], event2.duration[i], rtol=1e-5) for i in range(config.nIFO)) and \
            all(np.isclose(event.frequency[i], event2.frequency[i], rtol=1e-5) for i in range(config.nIFO)) and \
            all(np.isclose(event.bandwidth[i], event2.bandwidth[i], rtol=1e-5) for i in range(config.nIFO)) and \
            all(np.isclose(event.strain[i], event2.strain[i], rtol=snr_rtol, atol=1e-50) for i in range(len(event.strain))) and \
            np.isclose(event.anet, cluster.cluster_meta.a_net) and \
            np.isclose(event.gnet, cluster.cluster_meta.g_net) and \
            np.isclose(event.inet, cluster.cluster_meta.i_net) and \
            event.size[0] == cluster.get_core_size() and \
            event.size[1] == cluster.cluster_meta.sky_size:
            print("✅ Results match between old and new likelihood code.")
        else:
            print("❌ Results do NOT match between old and new likelihood code.")

        # ---- Qveto / Qfactor comparison ----
        # Reconstruct whitened waveforms from both old (ROOT) and new (Python) clusters,
        # then compute Qveto/Qfactor using the same get_qveto function used in production.
        print("\n--- Qveto / Qfactor ---")
        reconst_old = reconstruct_waveforms_flow(
            None, config, job_segments[0].ifos, event, cluster_old,
            epoch=0., wave_file='', save=False, plot=False,
        )
        reconst_new = reconstruct_waveforms_flow(
            None, config, job_segments[0].ifos, event2, cluster,
            epoch=0., wave_file='', save=False, plot=False,
        )
        qveto_match = True
        for ifo in job_segments[0].ifos:
            for a_type in ['DAT', 'REC']:
                key = f'{ifo}_wf_{a_type}_whiten'
                qveto_old, qfactor_old = get_qveto(reconst_old[key])
                qveto_new, qfactor_new = get_qveto(reconst_new[key])
                print(f"[old] {ifo} {a_type}: qveto={qveto_old:.6f}, qfactor={qfactor_old:.6f}")
                print(f"[new] {ifo} {a_type}: qveto={qveto_new:.6f}, qfactor={qfactor_new:.6f}")
                if not np.isclose(qveto_old, qveto_new, rtol=1e-2, atol=1e-6) or \
                   not np.isclose(qfactor_old, qfactor_new, rtol=1e-2, atol=1e-6):
                    qveto_match = False
        if qveto_match:
            print("✅ Qveto/Qfactor match between old and new likelihood code.")
        else:
            print("❌ Qveto/Qfactor do NOT match between old and new likelihood code.")
