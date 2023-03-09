from filelock import SoftFileLock
import json
import pyburst


def create_catalog(filename, config, jobs):
    """
    Create a catalog file with config, version, jobs

    A soft lock is used (default filelock does not work on CIT)

    :param filename: filename of the catalog
    :type filename: str
    :param config: config object
    :type config: pyburst.config.Config
    :param jobs: list of jobs
    :type jobs: list[pyburst.module.job_segment.Job]
    :return: None
    """
    output = {
        "config": config.__dict__,
        "version": pyburst.__version__,
        "jobs": [job.to_dict() for job in jobs],
        "events": []
    }

    with SoftFileLock(filename + ".lock", timeout=10):
        with open(filename, 'w') as f:
            json.dump(output, f, default=vars)


def add_events_to_catalog(filename, events):
    """
    Add events to catalog

    A soft lock is used (default filelock does not work on CIT)

    :param filename: filename of the catalog to update
    :type filename: str
    :param events: list of events
    :type events: list[pyburst.module.netevent.Event]
    :return: None
    """
    with SoftFileLock(filename + ".lock", timeout=10):
        # read the json file
        with open(filename, 'r+') as f:
            catalog = json.load(f)
            # append events
            catalog["events"].extend(events)
            # write the json file
            f.seek(0)
            json.dump(catalog, f)

