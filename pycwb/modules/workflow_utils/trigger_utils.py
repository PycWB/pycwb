import logging
import os

from pycwb.types import BaseCatalog
from pycwb.modules.catalog import Catalog
from pycwb.types.job import WaveSegment
from pycwb.types.trigger import Trigger
from pycwb.utils.dataclass_object_io import save_dataclass_to_json

logger = logging.getLogger(__name__)


def create_single_trigger_folder(working_dir: str, trigger_dir: str, job_seg: WaveSegment, event: tuple) -> str:
    """
    Create a trigger folder for the given event and job segment.

    Parameters
    ----------
    working_dir : str
        The working directory for the run
    trigger_dir : str
        The directory to save the triggers
    job_seg : WaveSegment
        The job segment to process
    event : tuple
        The event data

    Returns
    -------
    str
        The path to the trigger folder
    """
    trigger_folder = (
        f"{working_dir}/{trigger_dir}/"
        f"trigger_{job_seg.index}_{job_seg.trial_idx}_{event[0].stop[0]}_{event[0].hash_id}"
    )

    ## Do not create the folder here, let the save function create it to prevent too many folders
    # if not os.path.exists(trigger_folder):
    #     os.makedirs(trigger_folder)
    # else:
    #     logger.info(f"Trigger folder {trigger_folder} already exists, skip")

    return trigger_folder


def save_trigger(trigger_folder: str, trigger_data: tuple | list,
                 save_cluster: bool = True, save_sky_map: bool = True,
                 index: bool = None):
    if index is None:
        event, cluster, event_skymap_statistics = trigger_data
    else:
        event, cluster, event_skymap_statistics = trigger_data[index]

    # Save the event to the trigger folder
    if save_cluster or save_sky_map:
        if not os.path.exists(trigger_folder):
            os.makedirs(trigger_folder)

        logger.info(f"Saving trigger {event.hash_id}")

        # save_dataclass_to_json(event, f"{trigger_folder}/event.json")
        if save_cluster:
            save_dataclass_to_json(cluster, f"{trigger_folder}/cluster.json")
        if save_sky_map:
            save_dataclass_to_json(event_skymap_statistics, f"{trigger_folder}/skymap_statistics.json")

    return trigger_folder


def add_event_to_catalog(working_dir: str, catalog_dir: str, trigger_data: tuple | list,
                         catalog_file: str = Catalog.DEFAULT_FILENAME, index: int = None):
    """
    Convert an event to a :class:`~pycwb.types.trigger.Trigger` and append it
    to the Arrow/Parquet catalog.

    Parameters
    ----------
    working_dir : str
        The working directory for the run
    catalog_dir : str
        The directory to save the catalog
    trigger_data : tuple | list
        The event data tuple ``(event, cluster, sky_stats)``
    catalog_file : str
        The catalog file to save the triggers (default ``catalog.parquet``)
    index : int
        The index of the event in the list of events

    Returns
    -------
    str
        The path to the catalog file
    """
    if catalog_file is None:
        catalog_file = Catalog.DEFAULT_FILENAME

    if index is None:
        event, _, _ = trigger_data
    else:
        event, _, _ = trigger_data[index]

    logger.info(f"Adding event {event.hash_id} to catalog")
    if not catalog_file.startswith("/"):
        catalog_file = f"{working_dir}/{catalog_dir}/{catalog_file}"

    trigger = Trigger.from_event(event)
    Catalog.open(catalog_file).add_triggers(trigger)
    logger.info(f"Event {event.hash_id} added to catalog {catalog_file}")

    return catalog_file
