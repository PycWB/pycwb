import logging
import os
import psutil
from pycwb.config import Config
from pycwb.modules.catalog import add_events_to_catalog
from pycwb.modules.super_cluster.super_cluster import supercluster_wrapper
from pycwb.modules.super_cluster.supercluster import supercluster
from pycwb.modules.xtalk.monster import load_catalog
from pycwb.modules.coherence.coherence import coherence
from pycwb.modules.read_data import generate_injection, generate_noise_for_job_seg, read_from_job_segment
from pycwb.modules.data_conditioning import data_conditioning
from pycwb.modules.likelihood import likelihood
from pycwb.types.job import WaveSegment
from pycwb.types.network import Network
from pycwb.modules.workflow_utils.job_setup import print_job_info
from pycwb.utils.dataclass_object_io import save_dataclass_to_json
from pycwb.workflow.subflow.postprocess_and_plots import plot_trigger_flow, reconstruct_waveforms_flow, plot_skymap_flow

logger = logging.getLogger(__name__)


def process_job_segment(working_dir: str, config: Config, job_seg: WaveSegment, compress_json: bool = True,
                        catalog_file: str = None):
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

    logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

    tf_maps, nRMS_list = data_conditioning(config, data)
    logger.info("Memory usage: %f.2 MB", psutil.Process().memory_info().rss / 1024 / 1024)

    

    return trigger_folders


def save_trigger(working_dir: str, trigger_dir: str, catalog_dir: str,
                 job_seg: WaveSegment, trigger_data: tuple | list,
                 save_sky_map: bool = True, index: bool = None, catalog_file: str = "catalog.json"):
    if index is None:
        event, cluster, event_skymap_statistics = trigger_data
    else:
        event, cluster, event_skymap_statistics = trigger_data[index]

    if catalog_file is None:
        catalog_file = "catalog.json"

    if cluster.cluster_status != -1:
        return 0

    print(f"Saving trigger {event.hash_id}")

    trigger_folder = f"{working_dir}/{trigger_dir}/trigger_{job_seg.index}_{event.stop[0]}_{event.hash_id}"
    print(f"Creating trigger folder: {trigger_folder}")
    if not os.path.exists(trigger_folder):
        os.makedirs(trigger_folder)
    else:
        print(f"Trigger folder {trigger_folder} already exists, skip")

    print(f"Saving trigger data")
    save_dataclass_to_json(event, f"{trigger_folder}/event.json")
    save_dataclass_to_json(cluster, f"{trigger_folder}/cluster.json")
    if save_sky_map:
        save_dataclass_to_json(event_skymap_statistics, f"{trigger_folder}/skymap_statistics.json")

    print(f"Adding event to catalog")
    # if catalog_file is in full absolute path, use it directly
    if not catalog_file.startswith("/"):
        catalog_file = f"{working_dir}/{catalog_dir}/{catalog_file}"
    add_events_to_catalog(catalog_file, event.summary())

    return trigger_folder


# def process_job_segment_dask(working_dir, config, job_seg, plot=False, compress_json=True, client=None):
#     print_job_info(job_seg)
#
#     if not job_seg.frames and not job_seg.noise and not job_seg.injections:
#         raise ValueError("No data to process")
#
#     if job_seg.frames:
#         frame_data = client.map(read_single_frame_from_job_segment,
#                                 [config] * len(job_seg.frames),
#                                 [job_seg] * len(job_seg.frames),
#                                 job_seg.frames)
#         data = client.submit(merge_frames, job_seg, frame_data, config.segEdge)
#     else:
#         data = None
#
#     if job_seg.noise:
#         data = client.submit(generate_noise_for_job_seg, job_seg, config.inRate, data=data)
#     if job_seg.injections:
#         data = client.submit(generate_injection, config, job_seg, data)
#
#     xtalk_catalog = client.submit(load_catalog, config.MRAcatalog)
#     conditioned_data = client.submit(data_conditioning, config, data)
#     fragment_clusters_multi_res = client.map(coherence_single_res_wrapper, list(range(config.nRES)),
#                                              [config] * config.nRES, [conditioned_data] * config.nRES)
#
#     trigger_folders = client.submit(supercluster_and_likelihood, working_dir, config, job_seg,
#                                  fragment_clusters_multi_res, conditioned_data, xtalk_catalog)
#
#     return client.gather(trigger_folders)
