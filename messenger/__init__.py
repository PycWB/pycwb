import asyncio
import time


def periodic_trigger(event, trigger):
    now = int(time.time())
    interval = 5
    print(f"------ Waiting for {interval} seconds -----")
    trigger({
        "key": "RETRIEVE_DATA",
        "data": {
            "start": now,
            "end": now + interval,
            "interval": interval
        }})


# Here are the functions
def retrieve_data(event, trigger):
    time.sleep(0.5)
    print(f"new data retrieved: from {event['data']['start']} to {event['data']['end']} "
          f"Data lens is {event['data']['interval']} second")
    # wrap some gwpy retrieve data functions
    trigger({"key": "DATA_RETRIEVED", "data": "a.hdf5"})


def pixelate(event, trigger):
    time.sleep(0.5)
    # wrap cwb functions
    print(f"pixelating {event['data']}")
    trigger({"key": "PIXELATED", "data": "cluster.hdf5"})


def cluster(event, trigger):
    time.sleep(0.5)
    # wrap cwb functions
    print(f"clustering {event['data']}")
    trigger({"key": "CLUSTERED", "data": "cluster1.hdf5"})


def cluster2(event, trigger):
    time.sleep(0.5)
    # some user defined clustering algorithm
    print(f"clustering {event['data']} with another methods")
    trigger({"key": "CLUSTERED", "data": "cluster2.hdf5"})


def likelihood(event, trigger):
    time.sleep(0.5)
    # wrap cwb functions
    print(f"calculate likelihood from {event['data']}")
    trigger({"key": "LIKELIHOOD_CALCULATED", "data": "likelihood.hdf5"})


def plot(event, trigger):
    time.sleep(0.5)
    print(f"generating plot for {event['data']}")


config = {
    "events": [
        {
            "event": "START",
            "function": "periodic_trigger",
            "periodic": {
                "interval": 5,
                # "repeat": 20
            }
        },
        {
            "event": "RETRIEVE_DATA",
            "function": "retrieve_data"
        },
        {
            "event": "DATA_RETRIEVED",
            "function": "pixelate"
        },
        {
            "event": "PIXELATED",
            "function": "cluster"
        },
        {
            "event": "PIXELATED",
            "function": "cluster2"
        },
        {
            "event": "CLUSTERED",
            "function": "likelihood"
        },
        {
            "event": "LIKELIHOOD_CALCULATED",
            "function": "plot"
        },
        {
            "event": "CLUSTERED",
            "function": "plot"
        },
        {
            "event": "PIXELATED",
            "function": "plot"
        }
    ]
}


class Messenger:
    def __init__(self, config):
        self.events = asyncio.Queue()
        self.tasks = []
        self.registered = []
        self.load_config(config)

        self.trigger({"key": "START"})

        pass

    def load_config(self, config):
        self.registered = config['events']

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

            if not event['key'] in [r['event'] for r in self.registered]:
                print("key error")

            funcs = [r for r in self.registered if r['event'] == event['key']]

            if len(funcs) == 0:
                print(f"No function for key {event['key']}")

            for f in funcs:
                if "periodic" in f.keys():
                    task = asyncio.create_task(
                        self.periodic_task_wrapper(f["periodic"]["interval"],
                                                   globals()[f['function']],
                                                   event))
                else:
                    task = asyncio.create_task(
                        self.task_wrapper(globals()[f['function']],
                                          event))
                self.tasks.append(task)

    async def task_wrapper(self, func, event):
        func(event, self.trigger)
        # self.events.task_done()

    async def periodic_task_wrapper(self, dt, func, event):
        while True:
            func(event, self.trigger)
            await asyncio.sleep(dt)

    def trigger(self, event):
        self.events.put_nowait(event)

    def cleaner(self):
        for task in self.tasks:
            task.cancel()


if __name__ == '__main__':
    messenger = Messenger(config)
    try:
        asyncio.run(messenger.supervisor())
    except KeyboardInterrupt as e:
        print("Caught keyboard interrupt. Canceling tasks...")
        messenger.cleaner()
