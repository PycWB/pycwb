import asyncio
import shutil

from gwpy.timeseries import TimeSeries
from prefect.task_runners import ConcurrentTaskRunner
from prefect.utilities.annotations import quote

from pycwb.modules.coherence.coherence import _coherence_single_res
from pycwb.modules.cwb_conversions import convert_to_wavearray, convert_wavearray_to_timeseries
from pycwb.modules.superlag import generate_slags
from pycwb.types.network import Network
from pycwb.modules.autoencoder import get_glitchness
from pycwb.modules.reconstruction import get_network_MRA_wave
from pycwb.modules.read_data import read_from_job_segment, generate_injection, read_from_gwf, check_and_resample
from pycwb.modules.data_conditioning import data_conditioning, regression, whitening
from pycwb.modules.coherence import coherence
from pycwb.modules.super_cluster import supercluster
from pycwb.modules.likelihood import likelihood
from pycwb.modules.job_segment import create_job_segment_from_config
from pycwb.modules.catalog import create_catalog, add_events_to_catalog
from pycwb.modules.plot.cluster_statistics import plot_statistics
from pycwb.modules.web_viewer.create import create_web_viewer
from pycwb.modules.plot_map.world_map import plot_world_map, plot_skymap_contour
from pycwb.modules.plot.waveform import plot_reconstructed_waveforms
from pycwb.utils.dataclass_object_io import save_dataclass_to_json
from pycwb.config import Config
from prefect import flow, task, get_run_logger
from prefect_dask.task_runners import DaskTaskRunner
import os
import time

from dask.distributed import Client


@task
def read_config(file_name):
    return Config(file_name)


@task
def job_generator(nifo, slag_min, slag_max, slag_off, slag_size):
    return generate_slags(nifo, slag_min, slag_max, slag_off, slag_size)


@task
def job_runner(config, slag):
    import time
    time.sleep(3)
    print(slag)
    return 0


@task
def create_working_directory(working_dir):
    logger = get_run_logger()

    working_dir = os.path.abspath(working_dir)
    if not os.path.exists(working_dir):
        logger.info(f"Creating working directory: {working_dir}")
        os.makedirs(working_dir)
    os.chdir(working_dir)


@task
def check_if_output_exists(output_dir, overwrite=False):
    logger = get_run_logger()
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
def create_catalog_file(config, job_segments):
    logger = get_run_logger()
    logger.info("Creating catalog file")
    create_catalog(f"{config.outputDir}/catalog.json", config, job_segments)


@task
def create_web_dir(config):
    logger = get_run_logger()
    logger.info("Creating web directory")
    create_web_viewer(config.outputDir)


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
def read_file_from_job_segment(config, frame, job_seg):
    # should read data with segment edge
    start = job_seg.start_time - config.segEdge
    end = job_seg.end_time + config.segEdge
    print(f"Reading data from {frame.path} from {start} to {end}")

    # for each frame, if the frame start time is later than the job segment start time, use the frame start time
    if frame.start_time > start:  start = frame.start_time

    # for each frame, if the frame end time is earlier than the job segment end time, use the frame end time
    if frame.end_time < end: end = frame.end_time

    i = config.ifo.index(frame.ifo)
    data = read_from_gwf(frame.path, config.channelNamesRaw[i], start=start, end=end)
    print(f'Read data: start={data.t0}, duration={data.duration}, rate={data.sample_rate}')
    if int(data.sample_rate.value) != int(config.inRate):
        sample_rate_old = data.sample_rate.value
        w = convert_to_wavearray(data)
        w.Resample(config.inRate)
        data = convert_wavearray_to_timeseries(w)
        # data = data.resample(config.inRate)
        print(f'Resample data from {sample_rate_old} to {config.inRate}')
    return check_and_resample(data, config, i)


@task
def merge_frame_files(config, job_seg, data) -> list:
    merged_data = []

    # split data by ifo for next step of merging
    ifo_frames = [[i for i, frame in enumerate(job_seg.frames) if frame.ifo == ifo] for ifo in config.ifo]

    for frames in ifo_frames:
        if len(frames) == 1:
            # if there is only one frame, no need to merge
            ifo_data = data[frames[0]]
        else:
            # merge gw frames
            ifo_data = TimeSeries.from_pycbc(data[frames[0]])
            # free memory
            data[frames[0]] = None

            for i in frames[1:]:
                # use append method from gwpy, raise error if there is a gap
                ifo_data.append(TimeSeries.from_pycbc(data[i]), gap='raise')
                # free memory
                data[i] = None

            # convert back to pycbc
            ifo_data = ifo_data.to_pycbc()

        # check if data range match with job segment
        if ifo_data.start_time != job_seg.start_time - config.segEdge or \
                ifo_data.end_time != job_seg.end_time + config.segEdge:
            print(f'Job segment {job_seg} not match with data {ifo_data}, '
                  f'the gwf data start at {ifo_data.start_time} and end at {ifo_data.end_time}')
            raise ValueError(f'Job segment {job_seg} not match with data {ifo_data}')

        print(f'data info: start={ifo_data.start_time}, duration={ifo_data.duration}, rate={ifo_data.sample_rate}')
        # append to final data
        merged_data.append(ifo_data)
    return merged_data


@task
def data_conditioning(config, strain):
    data_regression = regression(config, strain)
    return whitening(config, data_regression)


@task
def coherence(config, conditioned_data, res):
    tf_maps, nRMS_list = zip(*conditioned_data)

    # upper sample factor
    up_n = int(config.rateANA / 1024)
    if up_n < 1:
        up_n = 1

    return _coherence_single_res(res, config, tf_maps, nRMS_list, up_n)


@task
def create_network(config, conditioned_data):
    tf_maps, nRMS_list = zip(*conditioned_data)
    return Network(config, tf_maps, nRMS_list)


@task
def supercluster_wrapper(config, fragment_clusters_multi_res, network, conditioned_data):
    fragment_clusters = [item for sublist in fragment_clusters_multi_res for item in sublist]
    tf_maps, nRMS_list = zip(*conditioned_data)
    return supercluster(config, network, fragment_clusters, tf_maps)


@task
def likelihood_wrapper(config, network, fragment_clusters):
    return likelihood(config, network, fragment_clusters)


@task
def create_trigger_directory(config, job_seg, event):
    trigger_folder = f"{config.outputDir}/trigger_{job_seg.job_id}_{event.stop[0]}_{event.hash_id}"
    if not os.path.exists(trigger_folder):
        os.makedirs(trigger_folder)
    return trigger_folder


@task
def save_dataclass_to_json_wrapper(data, file_name):
    save_dataclass_to_json(data, file_name)


@flow(task_runner=DaskTaskRunner(cluster_kwargs={"n_workers": 6, "processes": True, "threads_per_worker": 1}),
      log_prints=True, retries=1)
def analysis(file_name, working_dir='.', overwrite=False, ):
    create_working_directory(working_dir)
    check_env()
    config = read_config(file_name)
    check_if_output_exists(config.outputDir, overwrite)
    create_output_directory(working_dir, config.outputDir, config.logDir, file_name)

    job_segments = create_job_segment(config)
    create_catalog_file(config, job_segments)
    create_web_dir(config)
    # slags = job_generator(len(config.ifo), config.slagMin, config.slagMax, config.slagOff, config.slagSize)

    for job_seg in job_segments:
        print_job_info(job_seg)
        data = [read_file_from_job_segment.submit(config, frame, job_seg) for frame in job_seg.frames]
        data_merged = merge_frame_files.submit(config, job_seg, data).result()
        conditioned_data = [data_conditioning.submit(config, strain) for strain in data_merged]
        fragment_clusters_multi_res = [coherence.submit(config, conditioned_data, res) for res in range(config.nRES)]
        network = create_network.submit(config, conditioned_data)
        fragment_clusters = supercluster_wrapper.submit(config, fragment_clusters_multi_res, network, conditioned_data)
        events, clusters, skymap_statistics = likelihood_wrapper.submit(config, network, fragment_clusters).result()
        for event, cluster, event_skymap_statistics in zip(events, clusters, skymap_statistics):
            if cluster.cluster_status != -1:
                continue
            extra_info = {}
            trigger_folder = create_trigger_directory.submit(config, job_seg, event)
            save_dataclass_to_json_wrapper.submit(event, f"{trigger_folder}/event.json")
            save_dataclass_to_json_wrapper.submit(cluster, f"{trigger_folder}/cluster.json")
            save_dataclass_to_json_wrapper.submit(event_skymap_statistics, f"{trigger_folder}/skymap_statistics.json")


if __name__ == "__main__":
    analysis('/Users/yumengxu/GWOSC/catalog/GWTC-1-confident/GW150914/pycWB/user_parameters_injection.yaml',
             working_dir="/Users/yumengxu/GWOSC/catalog/GWTC-1-confident/GW150914/pycWB",
             overwrite=True)
