import asyncio
import time
from pycWB import pycWB
import os


##########################
# middlewares
##########################
def cwb_config(event):
    cwb_user_config = event['data']['config']
    working_dir = event['data']['working_dir']
    os.chdir(working_dir)
    print(f"Current working dir {os.getcwd()}")
    print(f"Loading cwb config: {cwb_user_config}")
    cwb = pycWB(cwb_user_config)  # load envs and create dirs
    print(f"cwb initialized")
    event['cwb'] = cwb  # inject cwb instance
    return event


##########################
# Fake functions for each step
##########################
def periodic_trigger(event, trigger):
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


##########################
# Event configurations
##########################
config = {
    "middlewares": [
        {
            "key": "CWB_CONFIG",
            "inject": ["CWB_2G", "ONLINE"],
            "handler": "cwb_config"
        }
    ],
    "events": [
        {
            "event": "ONLINE",
            "handler": "periodic_trigger",
            "periodic": {
                "interval": 5,
                # "repeat": 20
            },
            "trigger": ["RETRIEVE_DATA"]
        },
        {
            "event": "CWB_2G",
            "handler": "cwb_init",
            "trigger": ["CWB_INITED"]
        },
        {
            "event": "CWB_INITED",
            "handler": "cwb_read_data",
            "trigger": ["DATA_RETRIEVED"]
        },
        {
            "event": "RETRIEVE_DATA",
            "handler": "retrieve_data",
            "trigger": ["DATA_RETRIEVED"]
        },
        {
            "event": "DATA_RETRIEVED",
            "handler": "coherence",
            "trigger": ["COHERENCE_DONE"]
        },
        {
            "event": "COHERENCE_DONE",
            "handler": "cluster",
            "trigger": ["CLUSTERED"]
        },
        {
            "event": "COHERENCE_DONE",
            "handler": "cluster2",
            "trigger": ["CLUSTERED"]
        },
        {
            "event": "CLUSTERED",
            "handler": "likelihood",
            "trigger": ["LIKELIHOOD_CALCULATED"]
        },
        {
            "event": "LIKELIHOOD_CALCULATED",
            "handler": "plot"
        },
        {
            "event": "CLUSTERED",
            "handler": "plot"
        },
        {
            "event": "PIXELATED",
            "handler": "plot"
        }
    ]
}

##########################
# Entry point event
##########################
starter = {"key": "CWB_2G", "data": {
    "job_id": 1,
    "working_dir": "/Users/yumengxu/Project/Physics/cwb/MultiStages2G_yaml",
    "config": "/Users/yumengxu/Project/Physics/cwb/MultiStages2G_yaml/config.ini",
    "user_parameters": "/Users/yumengxu/Project/Physics/cwb/MultiStages2G_yaml/user_parameters.yaml"
}}


class Messenger:
    def __init__(self, config, starter):
        self.events = asyncio.Queue()
        self.tasks = []
        self.registered = []
        self.middlewares = []
        self.load_config(config)

        self.trigger(starter)

        pass

    def load_config(self, config):
        self.registered = config['events']
        self.middlewares = config['middlewares']

    async def supervisor(self):
        # TODO: task monitor
        # TODO: file monitor
        await self.dispatcher()

    async def dispatcher(self):
        while True:
            event = await self.events.get()
            # print("event:", event)
            if event['key'] == 'EXIT':
                break

            # check if event handler exist
            if not event['key'] in [r['event'] for r in self.registered]:
                print("Event not exist, key error")

            # check middle ware
            middlewares = []
            for middleware in self.middlewares:
                if event['key'] in middleware['inject']:
                    middlewares.append(middleware)

            # get all modules registered for this event
            funcs = [r for r in self.registered if r['event'] == event['key']]

            if len(funcs) == 0:
                print(f"No handler for key {event['key']}")

            # trigger event
            for f in funcs:
                if "periodic" in f.keys():
                    task = asyncio.create_task(
                        self.periodic_task_wrapper(f["periodic"]["interval"],
                                                   globals()[f['handler']],
                                                   event,
                                                   middlewares))
                else:
                    task = asyncio.create_task(
                        self.task_wrapper(globals()[f['handler']],
                                          event,
                                          middlewares))
                self.tasks.append(task)

    async def task_wrapper(self, func, event, middlewares):
        # execute middleware
        for mw in middlewares:
            event = globals()[mw['handler']](event)
        try:
            func(event, self.trigger)
        except Exception as e:
            print('Error: ', e)
            self.events.task_done()
        # self.events.task_done()

    async def periodic_task_wrapper(self, dt, func, event, middlewares):
        # execute middleware
        for mw in middlewares:
            event = globals()[mw['handler']](event)

        while True:
            func(event, self.trigger)
            await asyncio.sleep(dt)

    def aggregator(self, event):
        pass

    def trigger(self, event):
        self.events.put_nowait(event)

    def cleaner(self):
        for task in self.tasks:
            task.cancel()


if __name__ == '__main__':
    messenger = Messenger(config, starter)
    try:
        asyncio.run(messenger.supervisor())
    except KeyboardInterrupt as e:
        print("Caught keyboard interrupt. Canceling tasks...")
        messenger.cleaner()
