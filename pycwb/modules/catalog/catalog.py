from filelock import SoftFileLock
import json
from pycwb import __version__


def create_catalog(filename, config, jobs):
    output = {
        "config": config.__dict__,
        "version": __version__,
        "jobs": [job.to_dict() for job in jobs],
        "events": []
    }

    with SoftFileLock(filename + ".lock", timeout=10):
        with open(filename, 'w') as f:
            json.dump(output, f, default=vars)


def add_events_to_catalog(filename, events):
    with SoftFileLock(filename + ".lock", timeout=10):
        # read the json file
        with open(filename, 'r+') as f:
            catalog = json.load(f)
            # append events
            catalog["events"].extend(events)
            # write the json file
            f.seek(0)
            json.dump(catalog, f)

