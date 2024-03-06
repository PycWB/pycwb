from prefect import task, get_run_logger
import shutil
import os

from pycwb.modules.super_cluster.super_cluster import supercluster_wrapper
from pycwb.modules.xtalk.monster import load_catalog
from pycwb.modules.coherence.coherence import coherence_single_res
from pycwb.modules.superlag import generate_slags
from pycwb.modules.reconstruction import get_network_MRA_wave
from pycwb.modules.read_data import read_from_job_segment, generate_injection, merge_frames, \
    read_single_frame_from_job_segment
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
    logger = get_run_logger()

    working_dir = os.path.abspath(working_dir)
    if not os.path.exists(working_dir):
        logger.info(f"Creating working directory: {working_dir}")
        os.makedirs(working_dir)
    os.chdir(working_dir)


@task
def check_if_output_exists(working_dir, output_dir, overwrite=False):
    logger = get_run_logger()
    output_dir = f"{working_dir}/{output_dir}"
    if os.path.exists(output_dir) and os.listdir(output_dir):
        if overwrite:
            logger.info(f"Overwrite output directory {output_dir}")
        else:
            logger.info(f"Output directory {output_dir} is not empty")
            raise ValueError(f"Output directory {output_dir} is not empty")


@task
def create_output_directory(working_dir, output_dir, log_dir, user_parameter_file):
    logger = get_run_logger()
    # create folder for output and log
    logger.info(f"Output folder: {working_dir}/{output_dir}")
    logger.info(f"Log folder: {working_dir}/{log_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(f"{output_dir}/user_parameters.yaml"):
        shutil.copyfile(user_parameter_file, f"{output_dir}/user_parameters.yaml")
    else:
        logger.warning(f"User parameters file already exists in {working_dir}/{output_dir}")


@task
def create_job_segment(config):
    job_segments = create_job_segment_from_config(config)
    return job_segments


@task
def create_catalog_file(working_dir, config, job_segments):
    logger = get_run_logger()
    logger.info("Creating catalog file")
    create_catalog(f"{working_dir}/{config.outputDir}/catalog.json", config, job_segments)


@task
def load_xtalk_catalog(MRAcatalog):
    logger = get_run_logger()
    logger.info("Loading cross-talk catalog")
    xtalk_coeff, xtalk_lookup_table, layers, nRes = load_catalog(MRAcatalog)
    logger.info("Cross-talk catalog loaded")
    return xtalk_coeff, xtalk_lookup_table, layers, nRes


@task
def create_web_dir(working_dir, output_dir):
    logger = get_run_logger()
    logger.info("Creating web directory")
    create_web_viewer(f"{working_dir}/{output_dir}")


@task
def check_env():
    logger = get_run_logger()
    if not os.environ.get('HOME_WAT_FILTERS'):
        logger.error("HOME_WAT_FILTERS is not set.")
        logger.info("Please download the latest version of cwb config "
                    "and set HOME_WAT_FILTERS to the path of folder XTALKS.")
        logger.info("Make sure you have installed git lfs before cloning the repository.")
        logger.info("For example:")
        logger.info("    git lfs install")
        logger.info("    git clone https://gitlab.com/gwburst/public/config_o3")
        logger.info("    export HOME_WAT_FILTERS=$(pwd)/config_o3/XTALKS")
        raise ValueError("HOME_WAT_FILTERS is not set.")
    return True


@task
def print_job_info(job_seg):
    logger = get_run_logger()
    job_id = job_seg.index
    logger.info(f"Job ID: {job_id}")
    logger.info(f"Start time: {job_seg.start_time}")
    logger.info(f"End time: {job_seg.end_time}")
    logger.info(f"Duration: {job_seg.end_time - job_seg.start_time}")


@task
def read_in_data(config, job_seg):
    data = None
    if job_seg.frames:
        data = read_from_job_segment(config, job_seg)
    if job_seg.injections:
        data = generate_injection(config, job_seg, data)
    return data


@task
def read_file_from_job_segment(config, job_seg, frame):
    single_frame = read_single_frame_from_job_segment(config, frame, job_seg)
    return single_frame


@task
def merge_frame_task(job_seg, data, seg_edge) -> list:
    merged_data = merge_frames(job_seg, data, seg_edge)

    return merged_data


@task
def data_conditioning_task(config, strain):
    print(f"Data conditioning for strain {strain}")
    data_regression = regression(config, strain)
    return whitening(config, data_regression)


@task
def coherence_task(config, conditioned_data, res):
    tf_maps, nRMS_list = zip(*conditioned_data)

    return coherence_single_res(res, config, tf_maps, nRMS_list)


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
    network = Network(config, tf_maps, nRMS_list)

    # perform supercluster
    super_fragment_clusters = supercluster_wrapper(config, network, fragment_clusters, tf_maps,
                                                   xtalk_coeff, xtalk_lookup_table, layers)

    # perform likelihood
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
def save_trigger(working_dir, config, job_seg, trigger_data):
    event, cluster, event_skymap_statistics = trigger_data

    if cluster.cluster_status != -1:
        return 0

    print(f"Saving trigger {event.hash_id}")
    extra_info = {}

    trigger_folder = f"{working_dir}/{config.outputDir}/trigger_{job_seg.index}_{event.stop[0]}_{event.hash_id}"
    print(f"Creating trigger folder: {trigger_folder}")
    if not os.path.exists(trigger_folder):
        os.makedirs(trigger_folder)
    else:
        print(f"Trigger folder {trigger_folder} already exists, skip")

    save_dataclass_to_json(event, f"{trigger_folder}/event.json")
    save_dataclass_to_json(cluster, f"{trigger_folder}/cluster.json")
    save_dataclass_to_json(event_skymap_statistics, f"{trigger_folder}/skymap_statistics.json")
