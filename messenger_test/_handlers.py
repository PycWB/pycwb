import time


##########################
# Fake functions for each step
##########################
def online_periodic_trigger(event, trigger):
    now = int(time.time())
    interval = 5
    job_id = f"{event['data']['job_id']}_{now}"
    print(f"------ Starting job {job_id}, next job will be in {interval} second -----")
    trigger({
        "key": "RETRIEVE_DATA",
        "data": {
            "start": now,
            "end": now + interval,
            "interval": interval,
            "job_id": job_id
        },
        "cwb": event['cwb']})


def retrieve_data(event, trigger):
    time.sleep(0.5)
    print(f"new data retrieved: from {event['data']['start']} to {event['data']['end']} "
          f"Data length is {event['data']['interval']} second")
    # wrap some gwpy retrieve data functions
    trigger({"key": "DATA_RETRIEVED",
             "data": "a.hdf5",
             "cwb": event['cwb']})


def cwb_init(event, trigger):
    cwb = event['cwb']
    job_id = event['data']['job_id']
    cwb.cwb_inet2G(job_id, event['data']['user_parameters'], 'INIT')

    trigger({"key": "CWB_INITED",
             "data": {
                 "job_id": job_id,
                 "file_label": f"_"
             },
             "cwb": cwb})


def cwb_read_data(event, trigger):
    cwb = event['cwb']
    job_id = event['data']['job_id']

    # cwb.cwb_inet2G(job_id, event['data']['user_parameters'], 'FULL')
    trigger({"key": "DATA_RETRIEVED",
             "data": {
                 "job_id": job_id
             },
             "cwb": cwb})


def coherence(event, trigger):
    time.sleep(0.5)
    # wrap cwb functions
    print(f"pixelating {event['data']}")
    trigger({"key": "COHERENCE_DONE", "data": "cluster.hdf5",
             "cwb": event['cwb']})


def cluster(event, trigger):
    time.sleep(0.5)
    # wrap cwb functions
    print(f"clustering {event['data']}")
    trigger({"key": "CLUSTERED", "data": "cluster1.hdf5",
             "cwb": event['cwb']})


def cluster2(event, trigger):
    time.sleep(0.5)
    # some user defined clustering algorithm
    print(f"clustering {event['data']} with another methods")
    trigger({"key": "CLUSTERED", "data": "cluster2.hdf5",
             "cwb": event['cwb']})


def likelihood(event, trigger):
    time.sleep(0.5)
    # wrap cwb functions
    print(f"calculate likelihood from {event['data']}")
    trigger({"key": "LIKELIHOOD_CALCULATED", "data": "likelihood.hdf5",
             "cwb": event['cwb']})


def plot(event, trigger):
    time.sleep(0.5)
    print(f"generating plot for {event['data']}")
