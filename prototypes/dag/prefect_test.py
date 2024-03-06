import shutil
import os
from prefect import flow, task, get_run_logger, unmapped
from prefect_dask.task_runners import DaskTaskRunner
from prefect_dask import get_dask_client
from dask.distributed import Client, LocalCluster

from pycwb.modules.super_cluster.super_cluster import supercluster_wrapper
from pycwb.modules.xtalk.monster import load_catalog

# import dask
# dask.config.set({"multiprocessing.context": "fork"})

import pycbc
from pycwb.modules.coherence.coherence import coherence_single_res
from pycwb.modules.superlag import generate_slags
from pycwb.types.network import Network
from pycwb.modules.reconstruction import get_network_MRA_wave
from pycwb.modules.read_data import read_from_job_segment, generate_injection, merge_frames, read_single_frame_from_job_segment
from pycwb.modules.data_conditioning import data_conditioning, regression, whitening
from pycwb.modules.coherence import coherence
from pycwb.modules.likelihood import likelihood
from pycwb.modules.job_segment import create_job_segment_from_config
from pycwb.modules.catalog import create_catalog, add_events_to_catalog
from pycwb.modules.web_viewer.create import create_web_viewer
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
    xtalk_coeff, xtalk_lookup_table, layers, nRes = load_catalog(MRAcatalog)
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

    # with get_dask_client() as client:
    #     merged_data = client.scatter(merged_data, hash=False)

    return merged_data


@task
def data_conditioning(config, strain):
    print(f"Data conditioning for strain {strain}")
    data_regression = regression(config, strain)
    return whitening(config, data_regression)


@task
def coherence(config, conditioned_data, res):
    tf_maps, nRMS_list = zip(*conditioned_data)

    # upper sample factor
    up_n = int(config.rateANA / 1024)
    if up_n < 1:
        up_n = 1

    return coherence_single_res(res, config, tf_maps, nRMS_list, up_n)


@task
def supercluster_and_likelihood_wrapper(config, fragment_clusters_multi_res, conditioned_data,
                                        xtalk_catalog):
    xtalk_coeff, xtalk_lookup_table, layers, nRes = xtalk_catalog
    fragment_clusters = [item for sublist in fragment_clusters_multi_res for item in sublist]
    tf_maps, nRMS_list = zip(*conditioned_data)
    network = Network(config, tf_maps, nRMS_list)
    super_fragment_clusters = supercluster_wrapper(config, network, fragment_clusters, tf_maps,
                                                   xtalk_coeff, xtalk_lookup_table, layers)
    events, clusters, skymap_statistics = likelihood(config, network, [super_fragment_clusters])

    events_data = zip(events, clusters, skymap_statistics)

    # with get_dask_client() as client:
    #     events_data = client.scatter(events_data)

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


@flow
def process_job_segment(working_dir, config, job_seg, xtalk_catalog):
    print_job_info(job_seg)
    data = read_file_from_job_segment.map(config, job_seg, job_seg.frames)
    data_merged = merge_frame_task.submit(job_seg, data, config.segEdge)
    conditioned_data = data_conditioning.map(config, data_merged)
    fragment_clusters_multi_res = coherence.map(config, unmapped(conditioned_data), range(config.nRES))

    triggers_data = supercluster_and_likelihood_wrapper.submit(config, fragment_clusters_multi_res,
                                                               conditioned_data, xtalk_catalog)

    save_trigger.map(working_dir, config, job_seg, triggers_data)



@flow
def pycwb_search(file_name, working_dir='.', overwrite=False, ):
    create_working_directory(working_dir)
    check_env()
    config = read_config(file_name)
    check_if_output_exists(working_dir, config.outputDir, overwrite)
    create_output_directory(working_dir, config.outputDir, config.logDir, file_name)

    job_segments = create_job_segment(config)
    create_catalog_file(working_dir, config, job_segments)
    create_web_dir(working_dir, config.outputDir)
    xtalk_catalog = load_xtalk_catalog.submit(config.MRAcatalog)
    # slags = job_generator(len(config.ifo), config.slagMin, config.slagMax, config.slagOff, config.slagSize)

    for job_seg in job_segments:
        print_job_info(job_seg)
        data = read_file_from_job_segment.map(config, job_seg, job_seg.frames)
        data_merged = merge_frame_task.submit(job_seg, data, config.segEdge)
        conditioned_data = data_conditioning.map(config, data_merged)
        fragment_clusters_multi_res = coherence.map(config, unmapped(conditioned_data), range(config.nRES))

        triggers_data = supercluster_and_likelihood_wrapper.submit(config, fragment_clusters_multi_res,
                                                                   conditioned_data, xtalk_catalog)

        save_trigger.map(working_dir, config, job_seg, triggers_data)


# Only initialize the Dask cluster in the worker processes
if __name__ != "__main__":
    cluster = LocalCluster(n_workers=4, processes=True, threads_per_worker=1)
    cluster.scale(4)
    client = Client(cluster)
    address = client.scheduler.address
    pycwb_search = pycwb_search.with_options(task_runner=DaskTaskRunner(address=address), log_prints=True, retries=0)


if __name__ == "__main__":
    pycwb_search.serve(name="PycWB-search")

# {
#     "file_name": "/Users/yumengxu/GWOSC/catalog/GWTC-1-confident/GW150914/pycWB/user_parameters_injection.yaml",
#     "working_dir": "/Users/yumengxu/GWOSC/catalog/GWTC-1-confident/GW150914/pycWB",
#     "overwrite": true
# }
