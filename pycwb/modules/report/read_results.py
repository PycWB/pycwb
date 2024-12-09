from concurrent.futures import ThreadPoolExecutor
import orjson
import os
import numpy as np


def read_catalog(catalog_file):
    with open(catalog_file, 'rb') as f:
        catalog = orjson.loads(f.read())

    # remove redundant events (unique event['id']) in catalog
    n_events_before = len(catalog['events'])
    catalog['events'] = list({f"{event['job_id']}_{event['id']}": event for event in catalog['events']}.values())
    n_events_after = len(catalog['events'])
    if n_events_before != n_events_after:
        print(f"Removed {n_events_before - n_events_after} duplicated events")
    return catalog

def read_event(event_file):
    with open(event_file, 'rb') as f:
        events = orjson.loads(f.read())
    return events

def list_dict_filter(data, conditions, name='event'):
    print(f"number of {name} before filtering: {len(data)}")
    filter_string = " and ".join(conditions)
    print(f"Performing filter: {filter_string}")
    filtered_events = [d for d in data if eval(filter_string, {"__builtins__": None}, d)]
    print(f"number of {name} after filtering: {len(filtered_events)}")
    return filtered_events

def read_triggers(work_dir, run_dir, filters=None, file='catalog/catalog.json',**kwargs):
    print(f"Reading results from {os.path.join(work_dir, run_dir, file)}")
    catalog = read_catalog(os.path.join(work_dir, run_dir, file))

    events = []
    n_events = len(catalog['events'])
    # for i, event in enumerate(catalog['events']):
    #     if i % 100 == 0:
    #         print(f"Reading event {i}/{n_events}", end='\r')
    #     event_file = os.path.join(work_dir, run_dir, f"trigger/trigger_{event['job_id']}_{event['id']}/event.json")
    #     events.append(read_event(event_file))

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(read_event, os.path.join(work_dir, run_dir, f"trigger/trigger_{event['job_id']}_{event['id']}/event.json"))
                   for event in catalog['events']]
        for i, future in enumerate(futures):
            if i % 100 == 0:
                print(f"Reading event {i}/{n_events}", end='\r')
            events.append(future.result())
    if filters:
        events = list_dict_filter(events, filters, name='triggers')
    return events


def read_live_time(work_dir, run_dir, filters=None, lag_filter=None, file='catalog/catalog.json',**kwargs):
    print(f"Reading live time from {os.path.join(work_dir, run_dir, file)}")
    catalog = read_catalog(os.path.join(work_dir, run_dir, file))

    # TODO: this is for the simplest case only
    # combine the filter for lag and slag
    lags = np.arange(catalog['config']['lagSize'])
    if lag_filter:
        lags = [lag for lag in lags if eval(lag_filter, {"__builtins__": None}, {'lag': lag})]
    jobs = catalog['jobs']

    # FIXME: fix the live time calculation
    if filters:
        jobs = list_dict_filter(jobs, filters, name='jobs')

    durations = [(job['end_time'] - job['start_time']) * len(lags) for job in jobs]
    return sum(durations)