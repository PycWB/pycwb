from filelock import Timeout, FileLock
import json
from pycwb import __version__


def create_catalog(filename, config):
    output = {
        "config": json.dumps(config.__dict__),
        "version": __version__,
        "events": []
    }

    with FileLock(filename + ".lock", timeout=10):
        with open(filename, 'w') as f:
            json.dump(output, f)


def add_events_to_catalog(filename, events):
    with FileLock(filename + ".lock", timeout=10):
        # read the json file
        with open(filename, 'r+') as f:
            catalog = json.load(f)
            # append events
            catalog["events"].extend(events)
            # write the json file
            f.seek(0)
            json.dump(catalog, f)

