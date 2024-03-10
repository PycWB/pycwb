from prefect import task
import shutil
import os
from prefect.utilities.annotations import quote


from pycwb.modules.plot.cluster_statistics import plot_statistics
from pycwb.modules.plot.waveform import plot_reconstructed_waveforms
from pycwb.modules.plot_map.world_map import plot_skymap_contour
from pycwb.modules.super_cluster.super_cluster import supercluster_wrapper
from pycwb.modules.xtalk.monster import load_catalog
from pycwb.modules.coherence.coherence import coherence_single_res
from pycwb.modules.superlag import generate_slags
from pycwb.modules.reconstruction import get_network_MRA_wave
from pycwb.modules.read_data import read_from_job_segment, generate_injection, merge_frames, \
    read_single_frame_from_job_segment, generate_noise, generate_noise_for_job_seg
from pycwb.modules.data_conditioning import regression, whitening
from pycwb.modules.likelihood import likelihood
from pycwb.modules.job_segment import create_job_segment_from_config
from pycwb.modules.catalog import create_catalog, add_events_to_catalog
from pycwb.modules.web_viewer.create import create_web_viewer

from pycwb.types.network import Network
from pycwb.utils.dataclass_object_io import save_dataclass_to_json
from pycwb.config import Config


@task
def read_config(file_name):
    return Config(file_name)


@task
def job_generator(nifo, slag_min, slag_max, slag_off, slag_size):
    return generate_slags(nifo, slag_min, slag_max, slag_off, slag_size)


@task
def create_working_directory(working_dir):
    working_dir = os.path.abspath(working_dir)
    if not os.path.exists(working_dir):
        print(f"Creating working directory: {working_dir}")
        os.makedirs(working_dir)
    os.chdir(working_dir)


@task
def check_if_output_exists(working_dir, output_dir, overwrite=False):
    output_dir = f"{working_dir}/{output_dir}"
    if os.path.exists(output_dir) and os.listdir(output_dir):
        if overwrite:
            print(f"Overwrite output directory {output_dir}")
        else:
            print(f"Output directory {output_dir} is not empty")
            raise ValueError(f"Output directory {output_dir} is not empty")


@task
def create_output_directory(working_dir, output_dir, log_dir, user_parameter_file):
    # create folder for output and log
    print(f"Output folder: {working_dir}/{output_dir}")
    print(f"Log folder: {working_dir}/{log_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # if not os.path.exists(f"{output_dir}/user_parameters.yaml"):
    shutil.copyfile(user_parameter_file, f"{output_dir}/user_parameters.yaml")
    # else:
    #     logger.warning(f"User parameters file already exists in {working_dir}/{output_dir}")


@task
def create_job_segment(config):
    job_segments = create_job_segment_from_config(config)
    return job_segments


@task
def create_catalog_file(working_dir, config, job_segments):
    print("Creating catalog file")
    create_catalog(f"{working_dir}/{config.outputDir}/catalog.json", config, job_segments)


@task
def load_xtalk_catalog(MRAcatalog):
    print("Loading cross-talk catalog")
    xtalk_coeff, xtalk_lookup_table, layers, nRes = load_catalog(MRAcatalog)
    print("Cross-talk catalog loaded")
    return xtalk_coeff, xtalk_lookup_table, layers, nRes


@task
def create_web_dir(working_dir, output_dir):
    print("Creating web directory")
    create_web_viewer(f"{working_dir}/{output_dir}")


@task
def check_env():
    if not os.environ.get('HOME_WAT_FILTERS'):
        print("HOME_WAT_FILTERS is not set.")
        print("Please download the latest version of cwb config "
                    "and set HOME_WAT_FILTERS to the path of folder XTALKS.")
        print("Make sure you have installed git lfs before cloning the repository.")
        print("For example:")
        print("    git lfs install")
        print("    git clone https://gitlab.com/gwburst/public/config_o3")
        print("    export HOME_WAT_FILTERS=$(pwd)/config_o3/XTALKS")
        raise ValueError("HOME_WAT_FILTERS is not set.")
    return True


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
    return clusters


@task
def supercluster_and_likelihood_task(config, fragment_clusters_multi_res, conditioned_data,
                                        xtalk_catalog):
    # cross-talk catalog
    xtalk_coeff, xtalk_lookup_table, layers, nRes = xtalk_catalog

    # flatten the fragment clusters from different resolutions
    fragment_clusters = [item for sublist in fragment_clusters_multi_res for item in sublist]

    # extract the tf_maps and nRMS_list from conditioned_data
    tf_maps, nRMS_list = zip(*conditioned_data)

    # prepare the network object required for cwb likelihood, will be removed in the future
    print("Creating network object")
    network = Network(config, tf_maps, nRMS_list)

    # perform supercluster
    print("Performing supercluster")
    super_fragment_clusters = supercluster_wrapper(config, network, fragment_clusters, tf_maps,
                                                   xtalk_coeff, xtalk_lookup_table, layers)

    # perform likelihood
    print("Performing likelihood")
    events, clusters, skymap_statistics = likelihood(config, network, [super_fragment_clusters])

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
    if not index:
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
    if not index:
        event, cluster, event_skymap_statistics = trigger_data
        trigger_folder = trigger_folders
    else:
        trigger_folder = trigger_folders[index]
        event, cluster, event_skymap_statistics = trigger_data[index]

    # trigger_folder = f"{working_dir}/{config.outputDir}/trigger_{job_seg.index}_{event.stop[0]}_{event.hash_id}"

    print(f"Reconstructing waveform for event {event.hash_id}")
    reconstructed_waves = get_network_MRA_wave(config, cluster, config.rateANA, config.nIFO, config.TDRate,
                                               'signal', 0, True, whiten=False)

    print(f"Reconstructing whitened waveform for event {event.hash_id}")
    reconstructed_waves_whiten = get_network_MRA_wave(config, cluster, config.rateANA, config.nIFO, config.TDRate,
                                                        'signal', 0, True, whiten=True)

    reconstructed_waves_whiten_00 = get_network_MRA_wave(config, cluster, config.rateANA, config.nIFO, config.TDRate,
                                                        'signal', -1, True, whiten=True)
    reconstructed_waves_whiten_90 = get_network_MRA_wave(config, cluster, config.rateANA, config.nIFO, config.TDRate,
                                                        'signal', 1, True, whiten=True)

    if not os.path.exists(trigger_folder):
        os.makedirs(trigger_folder)
        print(f"Creating trigger folder: {trigger_folder}")

    for i, ts in enumerate(reconstructed_waves):
        print(f"Saving reconstructed waveform for {job_seg.ifos[i]}")
        ts.save(f"{trigger_folder}/reconstructed_waveform_{job_seg.ifos[i]}.txt")

    for i, ts in enumerate(reconstructed_waves_whiten):
        print(f"Saving reconstructed waveform for {job_seg.ifos[i]} (whitened)")
        ts.save(f"{trigger_folder}/reconstructed_waveform_{job_seg.ifos[i]}_whitened.txt")

    for i, (hp, hc) in enumerate(zip(reconstructed_waves_whiten_00, reconstructed_waves_whiten_90)):
        # save strain = hp + 1j hc
        print(f"Saving reconstructed strain for {job_seg.ifos[i]} (whitened)")
        hp = hp + 1j * hc
        hp.save(f"{trigger_folder}/reconstructed_strain_{job_seg.ifos[i]}_whitened.txt")

    if plot:
        from matplotlib import pyplot as plt

        for j, reconstructed_wave in enumerate(reconstructed_waves):
            plt.plot(reconstructed_wave.sample_times, reconstructed_wave.data)
            plt.xlim((event.left[0], event.left[0] + event.stop[0] - event.start[0]))
            plt.savefig(f'{trigger_folder}/reconstructed_wave_ifo_{j+1}.png')
            plt.close()

        for j, reconstructed_wave in enumerate(reconstructed_waves_whiten):
            plt.plot(reconstructed_wave.sample_times, reconstructed_wave.data)
            plt.xlim((event.left[0], event.left[0] + event.stop[0] - event.start[0]))
            plt.savefig(f'{trigger_folder}/reconstructed_wave_whiten_ifo_{j+1}.png')
            plt.close()

    return reconstructed_waves


@task
def plot_triggers(trigger_folders, trigger_data, index):
    trigger_folder = trigger_folders[index]
    event, cluster, event_skymap_statistics = trigger_data[index]

    print(f"Making plots for event {event.hash_id}")
    # trigger_folder = f"{working_dir}/{config.outputDir}/trigger_{job_seg.index}_{event.stop[0]}_{event.hash_id}"

    # plot the likelihood map
    plot_statistics(cluster, 'likelihood', filename=f'{trigger_folder}/likelihood_map.png')
    plot_statistics(cluster, 'null', filename=f'{trigger_folder}/null_map.png')

    # plot_world_map(event.phi[0], event.theta[0], filename=f'{config.outputDir}/world_map_{job_id}_{i+1}.png')
    for key in event_skymap_statistics.keys():
        plot_skymap_contour(event_skymap_statistics,
                            key=key,
                            reconstructed_loc=(event.phi[0], event.theta[0]),
                            detector_loc=(event.phi[3], event.theta[3]),
                            resolution=1,
                            filename=f'{trigger_folder}/{key}.png')