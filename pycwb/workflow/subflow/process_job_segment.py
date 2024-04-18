from pycwb.config import Config
from pycwb.modules.super_cluster.super_cluster import supercluster_wrapper
from pycwb.modules.xtalk.monster import load_catalog
from pycwb.modules.coherence.coherence import coherence
from pycwb.modules.read_data import generate_injection, generate_noise_for_job_seg, read_from_job_segment
from pycwb.modules.data_conditioning import data_conditioning
from pycwb.modules.likelihood import likelihood
from pycwb.types.job import WaveSegment
from pycwb.types.network import Network
from pycwb.modules.workflow_utils.job_setup import print_job_info
from pycwb.workflow.subflow.postprocess_and_plots import plot_trigger_flow, reconstruct_waveforms_flow
from pycwb.workflow.subflow.supercluster_and_likelihood import save_trigger


def process_job_segment(working_dir: str, config: Config, job_seg: WaveSegment, compress_json: bool = True):
    print_job_info(job_seg)

    if not job_seg.frames and not job_seg.noise and not job_seg.injections:
        raise ValueError("No data to process")

    data = None

    if job_seg.frames:
        data = read_from_job_segment(config, job_seg)
    if job_seg.noise:
        data = generate_noise_for_job_seg(job_seg, config.inRate, data=data)
    if job_seg.injections:
        data = generate_injection(config, job_seg, data)

    xtalk_coeff, xtalk_lookup_table, layers, _ = load_catalog(config.MRAcatalog)

    tf_maps, nRMS_list = data_conditioning(config, data)

    fragment_clusters = coherence(config, tf_maps, nRMS_list)

    network = Network(config, tf_maps, nRMS_list)

    super_fragment_clusters = supercluster_wrapper(config, network, fragment_clusters, tf_maps,
                                                   xtalk_coeff, xtalk_lookup_table, layers)

    # super_fragment_clusters = supercluster(config, network, fragment_clusters, tf_maps)

    events, clusters, skymap_statistics = likelihood(config, network, super_fragment_clusters)

    # only return selected events
    events_data = []
    for i, cluster in enumerate(clusters):
        if cluster.cluster_status != -1:
            continue
        event = events[i]
        event_skymap_statistics = skymap_statistics[i]
        events_data.append((event, cluster, event_skymap_statistics))

        # associate the injections if there are any
        if job_seg.injections:
            for injection in job_seg.injections:
                if event.start[0] - 0.1 < injection['gps_time'] < event.stop[0] + 0.1:
                    event.injection = injection

    # save triggers
    trigger_folders = []
    for trigger in events_data:
        trigger_folders.append(
            save_trigger(working_dir, config.trigger_dir, config.catalog_dir, job_seg, trigger,
                         save_sky_map=config.save_sky_map)
        )

    for trigger_folder, trigger in zip(trigger_folders, events_data):
        event, cluster, event_skymap_statistics = trigger
        reconstruct_waveforms_flow(trigger_folder, config, job_seg,
                                   event, cluster,
                                   save=config.save_waveform, plot=config.plot_waveform,
                                   save_injection=config.save_injection, plot_injection=config.plot_injection)
        if config.plot_sky_map:
            plot_trigger_flow(trigger_folder, event, cluster, event_skymap_statistics)

    return trigger_folders

