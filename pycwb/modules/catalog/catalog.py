import os
import logging
from filelock import SoftFileLock
import orjson
import pycwb
from pycwb.config import Config
from pycwb.types.job import WaveSegment
from pycwb.types.network_event import Event

logger = logging.getLogger(__name__)


def create_catalog(filename: str, config: Config, jobs: list[WaveSegment]) -> None:
    """
    Create a catalog file with config, version, jobs

    A soft lock is used (default filelock does not work on CIT)

    Parameters
    ----------
    filename : str
        filename of the catalog
    config : pycwb.config.Config
        config object
    jobs : list of pycwb.types.job.WaveSegment
        list of jobs

    Returns
    -------
    None
    """
    output = {
        "config": config.__dict__,
        "version": pycwb.__version__,
        "jobs": jobs,
        "events": []
    }

    with SoftFileLock(filename + ".lock", timeout=10):
        with open(filename, 'wb') as f:
            f.write(orjson.dumps(output, option=orjson.OPT_SERIALIZE_NUMPY))


def add_events_to_catalog(filename: str, events: list[Event]) -> None:
    """
    Add events to catalog

    A soft lock is used (default filelock does not work on CIT)

    Parameters
    ----------
    filename : str
        filename of the catalog
    events : list of pycwb.types.network_event.Event
        list of events

    Returns
    -------
    None
    """
    if not isinstance(events, list):
        events = [events]

    if os.path.exists(filename):
        with SoftFileLock(filename + ".lock", timeout=10):
            # read the json file
            with open(filename, 'rb+') as f:
                catalog = orjson.loads(f.read())
                # append events
                catalog["events"].extend(events)
                # write the json file
                f.seek(0)
                f.write(orjson.dumps(catalog, option=orjson.OPT_SERIALIZE_NUMPY))
    else:
        logger.warning("Catalog file does not exist. Event will not be saved to catalog.")

