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
    job_id = event['data']['job_id']
    with open(f"data/strain_{job_id}.txt", "w") as my_file:
        my_file.write("Hello world \n")


def coherence(event, trigger):
    time.sleep(0.5)
    # wrap cwb functions
    print(f"coherence {event['data']}")
    job_id = event['data']['job_id']
    with open(f"data/coherence_{job_id}.txt", "w") as my_file:
        my_file.write("Hello world \n")


def cluster(event, trigger):
    time.sleep(0.5)
    # wrap cwb functions
    print(f"clustering {event['data']}")
    job_id = event['data']['job_id']
    with open(f"data/cluster_{job_id}.txt", "w") as my_file:
        my_file.write("Hello world \n")


def cluster2(event, trigger):
    job_id = event['data']['job_id']

    # some user defined clustering algorithm
    print(f"clustering {event['data']} with another methods")
    data = "Hello world \n"

    # manually trigger a plot handler
    trigger({"key": "PLOT_MODULE",
             "data": {
                 "job_id": job_id,
                 "data": data
             }})

    # write file
    with open(f"data/cluster_{job_id}_2.txt", "w") as my_file:
        my_file.write(data)


def likelihood(event, trigger):
    time.sleep(0.5)
    # wrap cwb functions
    print(f"calculate likelihood from {event['data']}")
    job_id = event['data']['job_id']
    with open(f"data/likelihood_{job_id}_2.txt", "w") as my_file:
        my_file.write("Hello world \n")


def plot(event, trigger):
    time.sleep(0.5)
    print(f"generating plot for {event['data']}")
