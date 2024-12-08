import orjson
import os

def read_catalog(catalog_file):
    with open(catalog_file, 'r') as f:
        catalog = orjson.loads(f.read())
    return catalog

def read_event(event_file):
    with open(event_file, 'r') as f:
        events = orjson.loads(f.read())
    return events

def events_filter(events, conditions):
    print(f"number of events before filtering: {len(events)}")
    filter_string = " and ".join(conditions)
    print(f"Performing filter: {filter_string}")
    filtered_events = [event for event in events if eval(filter_string, {"__builtins__": None}, event)]
    print(f"number of events after filtering: {len(filtered_events)}")
    return filtered_events

def read_triggers(work_dir, run_dir, filters, file='catalog/catalog.json',**kwargs):
    print(f"Reading results from {os.path.join(work_dir, run_dir, file)}")
    catalog = read_catalog(os.path.join(work_dir, run_dir, file))

    events = []
    for event in catalog['events']:
        event_file = os.path.join(work_dir, run_dir, f"trigger/trigger_{event['job_id']}_{event['id']}/event.json")
        events.append(read_event(event_file))

    if filters:
        events = events_filter(events, filters)
    return events


def read_live_time(work_dir, run_dir, file='catalog/catalog.json',**kwargs):
    print(f"Reading live time from {os.path.join(work_dir, run_dir, file)}")
    catalog = read_catalog(os.path.join(work_dir, run_dir, file))
    jobs = catalog['jobs']
    durations = [job['end_time'] - job['start_time'] for job in jobs]
    return sum(durations)