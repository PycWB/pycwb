import os

from pycwb.modules.catalog import add_events_to_catalog
from pycwb.modules.likelihood import likelihood
from pycwb.modules.super_cluster.super_cluster import supercluster_wrapper
from pycwb.types.job import WaveSegment
from pycwb.types.network import Network
from pycwb.utils.dataclass_object_io import save_dataclass_to_json


def supercluster_and_likelihood(working_dir, config, job_seg, fragment_clusters_multi_res, tf_maps, nRMS_list,
                                     xtalk_catalog):
    # cross-talk catalog
    xtalk_coeff, xtalk_lookup_table, layers, nRes = xtalk_catalog

    # flatten the fragment clusters from different resolutions
    fragment_clusters = [item for sublist in fragment_clusters_multi_res for item in sublist]

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

    # save triggers
    trigger_folders = []
    for trigger in events_data:
        trigger_folders.append(
            save_trigger(working_dir, config.trigger_dir, config.catalog_dir, job_seg, trigger,
                         save_sky_map=config.save_sky_map)
        )
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
    add_events_to_catalog(catalog_file, event.summary(job_seg.index, f"{event.stop[0]}_{event.hash_id}"))

    return trigger_folder