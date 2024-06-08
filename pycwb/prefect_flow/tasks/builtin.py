from prefect import task
import os
from prefect.utilities.annotations import quote


from pycwb.modules.plot.cluster_statistics import plot_statistics
from pycwb.modules.plot_map.world_map import plot_skymap_contour
from pycwb.modules.super_cluster.super_cluster import supercluster_wrapper
from pycwb.modules.xtalk.monster import load_catalog
from pycwb.modules.coherence.coherence import coherence_single_res
from pycwb.modules.reconstruction import get_network_MRA_wave
from pycwb.modules.read_data import generate_injection, merge_frames, \
    read_single_frame_from_job_segment, generate_noise_for_job_seg
from pycwb.modules.data_conditioning import regression, whitening
from pycwb.modules.likelihood import likelihood
from pycwb.modules.catalog import add_events_to_catalog

from pycwb.types.network import Network
from pycwb.utils.dataclass_object_io import save_dataclass_to_json
from pycwb.workflow.subflow.postprocess_and_plots import reconstruct_waveforms_flow, plot_trigger_flow


@task
def load_xtalk_catalog(MRAcatalog):
    print("Loading cross-talk catalog")
    xtalk_coeff, xtalk_lookup_table, layers, nRes = load_catalog(MRAcatalog)
    print("Cross-talk catalog loaded")
    return xtalk_coeff, xtalk_lookup_table, layers, nRes

@task
def print_job_info(job_seg):
    job_id = job_seg.index
    print(f"Job ID: {job_id}")
    print(f"Start time: {job_seg.start_time}")
    print(f"End time: {job_seg.end_time}")
    print(f"Duration: {job_seg.end_time - job_seg.start_time}")
    print(f"Frames: {job_seg.frames}")
    print(f"Noise: {job_seg.noise}")
    print(f"Injections: {job_seg.injections}")


@task
def generate_injection_task(config, job_seg, data=None):
    print(f"Generating injection for job segment {job_seg.index}")
    data = generate_injection(config, job_seg, data)
    print(f"Generated injection for job segment {job_seg.index}")

    # avoid prefect inspect the internal data
    # return [quote(d) for d in data]
    return data


@task
def read_file_from_job_segment(config, job_seg, frame):
    print(f"Reading frame {frame} from job segment {job_seg.index}")
    single_frame = read_single_frame_from_job_segment(config, frame, job_seg)
    print(f"Finished reading frame {frame} from job segment {job_seg.index}")
    return single_frame


@task
def merge_frame_task(job_seg, data, seg_edge) -> list:
    print(f"Merging frames for job segment {job_seg.index}")
    merged_data = merge_frames(job_seg, data, seg_edge)
    print(f"Merged frames for job segment {job_seg.index}")
    return merged_data


@task
def generate_noise_for_job_seg_task(job_seg, config, f_low=2.0, data=None):
    sample_rate = config.inRate
    print(f"Generating noise for job segment {job_seg.index}")
    if 'seeds' in job_seg.noise:
        print(f"Using seeds {job_seg.noise['seeds']}")
    print(f"Sample rate: {sample_rate}")
    print(f"Low frequency: {f_low}")
    data = generate_noise_for_job_seg(job_seg, sample_rate, f_low, data)
    print(f"Generated noise for job segment {job_seg.index}")
    return data


@task
def data_conditioning_task(config, strains, ifo):
    strain = strains[ifo]
    print(f"Data conditioning for {ifo}")
    print(f'Performing regression for {ifo}')
    data_regression = regression(config, strain)
    print(f'Performing whitening for {ifo}')
    whitened_data = whitening(config, data_regression)
    print(f"Data conditioning for {ifo} done")
    return whitened_data


@task
def coherence_task(config, conditioned_data, res):
    tf_maps, nRMS_list = zip(*conditioned_data)

    clusters = coherence_single_res(res, config, tf_maps, nRMS_list)

    n_pixels = 0
    n_clusters = 0
    for sc in clusters:
        n_clusters += len(sc.clusters)
        for cluster in sc.clusters:
            n_pixels += len(cluster.pixels)

    print(f"Resolutions {res}: {n_clusters} clusters, {n_pixels} pixels")
    return clusters


@task
def supercluster_and_likelihood_task(config, fragment_clusters_multi_res, conditioned_data,
                                        xtalk_catalog):
    # cross-talk catalog
    xtalk_coeff, xtalk_lookup_table, layers, nRes = xtalk_catalog

    # flatten the fragment clusters from different resolutions
    # fragment_clusters = [item for sublist in fragment_clusters_multi_res for item in sublist]

    # extract the tf_maps and nRMS_list from conditioned_data
    tf_maps, nRMS_list = zip(*conditioned_data)

    # prepare the network object required for cwb likelihood, will be removed in the future
    print("Creating network object")
    network = Network(config, tf_maps, nRMS_list)

    # perform supercluster
    print("Performing supercluster")
    super_fragment_clusters = supercluster_wrapper(config, network, fragment_clusters_multi_res, tf_maps,
                                                   xtalk_coeff, xtalk_lookup_table, layers)

    # perform likelihood
    print("Performing likelihood")
    events, clusters, skymap_statistics = likelihood(config, network, super_fragment_clusters)

    # only return selected events
    events_data = []
    for i, cluster in enumerate(clusters):
        if cluster.cluster_status != -1:
            continue
        event = events[i]
        event_skymap_statistics = skymap_statistics[i]
        events_data.append((event, cluster, event_skymap_statistics))

    return events_data


@task
def save_trigger(working_dir, config, job_seg, trigger_data, index=None):
    if index is None:
        event, cluster, event_skymap_statistics = trigger_data
    else:
        event, cluster, event_skymap_statistics = trigger_data[index]

    if cluster.cluster_status != -1:
        return 0

    print(f"Saving trigger {event.hash_id}")

    trigger_folder = f"{working_dir}/{config.outputDir}/trigger_{job_seg.index}_{event.stop[0]}_{event.hash_id}"
    print(f"Creating trigger folder: {trigger_folder}")
    if not os.path.exists(trigger_folder):
        os.makedirs(trigger_folder)
    else:
        print(f"Trigger folder {trigger_folder} already exists, skip")

    print(f"Saving trigger data")
    save_dataclass_to_json(event, f"{trigger_folder}/event.json")
    save_dataclass_to_json(cluster, f"{trigger_folder}/cluster.json")
    save_dataclass_to_json(event_skymap_statistics, f"{trigger_folder}/skymap_statistics.json")

    print(f"Adding event to catalog")
    add_events_to_catalog(f"{working_dir}/{config.outputDir}/catalog.json",
                          event.summary(job_seg.index, f"{event.stop[0]}_{event.hash_id}"))

    return trigger_folder


@task
def reconstruct_waveform(trigger_folders, config, job_seg, trigger_data, index=None, plot=False):
    if index is None:
        event, cluster, event_skymap_statistics = trigger_data
        trigger_folder = trigger_folders
    else:
        trigger_folder = trigger_folders[index]
        event, cluster, event_skymap_statistics = trigger_data[index]

    # trigger_folder = f"{working_dir}/{config.outputDir}/trigger_{job_seg.index}_{event.stop[0]}_{event.hash_id}"
    return reconstruct_waveforms_flow(trigger_folder, config, job_seg.ifos, event, cluster, save=True, plot=plot)


@task
def plot_triggers(trigger_folders, trigger_data, index):
    trigger_folder = trigger_folders[index]
    event, cluster, event_skymap_statistics = trigger_data[index]

    plot_trigger_flow(trigger_folder, event, cluster, event_skymap_statistics)