from distributed import Client, LocalCluster

from pycwb.modules.logger import logger_init, log_prints
from pycwb.modules.super_cluster.super_cluster import supercluster_wrapper
from pycwb.modules.super_cluster.supercluster import supercluster
from pycwb.modules.xtalk.monster import load_catalog
from pycwb.modules.coherence.coherence import coherence_single_res_wrapper, coherence_single_res, coherence
from pycwb.modules.read_data import generate_injection, merge_frames, \
    read_single_frame_from_job_segment, generate_noise_for_job_seg, read_from_job_segment
from pycwb.modules.data_conditioning import data_conditioning
from pycwb.modules.likelihood import likelihood
from pycwb.types.network import Network
from pycwb.modules.workflow_utils.job_setup import print_job_info
from pycwb.workflow.subflow import prepare_job_runs, supercluster_and_likelihood
from pycwb.workflow.subflow.postprocess_and_plots import plot_trigger_flow, reconstruct_waveforms_flow
from pycwb.workflow.subflow.supercluster_and_likelihood import save_trigger


def process_job_segment_dask(working_dir, config, job_seg, plot=False, compress_json=True, client=None):
    print_job_info(job_seg)

    if not job_seg.frames and not job_seg.noise and not job_seg.injections:
        raise ValueError("No data to process")

    if job_seg.frames:
        frame_data = client.map(read_single_frame_from_job_segment,
                                [config] * len(job_seg.frames),
                                [job_seg] * len(job_seg.frames),
                                job_seg.frames)
        data = client.submit(merge_frames, job_seg, frame_data, config.segEdge)
    else:
        data = None

    if job_seg.noise:
        data = client.submit(generate_noise_for_job_seg, job_seg, config.inRate, data=data)
    if job_seg.injections:
        data = client.submit(generate_injection, config, job_seg, data)

    xtalk_catalog = client.submit(load_catalog, config.MRAcatalog)
    conditioned_data = client.submit(data_conditioning, config, data)
    fragment_clusters_multi_res = client.map(coherence_single_res_wrapper, list(range(config.nRES)),
                                             [config] * config.nRES, [conditioned_data] * config.nRES)

    trigger_folders = client.submit(supercluster_and_likelihood, working_dir, config, job_seg,
                                 fragment_clusters_multi_res, conditioned_data, xtalk_catalog)

    return client.gather(trigger_folders)


def process_job_segment(working_dir, config, job_seg, plot=False, compress_json=True):
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

    # save triggers
    trigger_folders = []
    for trigger in events_data:
        trigger_folders.append(save_trigger(working_dir, config, job_seg, trigger))

    if plot:
        for trigger_folder, trigger in zip(trigger_folders, events_data):
            event, cluster, event_skymap_statistics = trigger
            reconstruct_waveforms_flow(trigger_folder, config, job_seg,
                                       event, cluster, save=True, plot=True)
            plot_trigger_flow(trigger_folder, event, cluster, event_skymap_statistics)

    return trigger_folders


def search(file_name, working_dir='.', overwrite=False, submit=False, log_file=None, log_level="INFO",
           n_proc=1, plot=False, compress_json=True, dry_run=False):
    job_segments, config, working_dir = prepare_job_runs(working_dir, file_name, n_proc, dry_run, overwrite)
    logger_init(log_file, log_level)
    # log_prints()
    # cluster = LocalCluster(n_workers=n_proc, processes=True, threads_per_worker=1)
    # cluster.scale(n_proc)
    # client = Client(cluster)

    for job_seg in job_segments:
        process_job_segment(working_dir, config, job_seg, plot, compress_json)

    # client.close()