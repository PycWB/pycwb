import asyncio
import importlib
from watchfiles import awatch
import os
import re


def import_helper(name):
    p, m = name.rsplit('.', 1)

    mod = importlib.import_module(p)
    met = getattr(mod, m)
    return met


class Messenger:
    def __init__(self, config, starter):
        self.events = asyncio.Queue()
        self.tasks = []
        self.registered = []
        self.middlewares = []
        self.watched_dir = []
        self.file_watchers = []
        self.load_config(config)

        self.trigger(starter)

        pass

    def load_config(self, config):
        # Register event handlers
        self.registered = config['events']
        for f in self.registered:
            f['handler'] = import_helper(f['handler'])

        # Register middlewares
        if 'middlewares' in config:
            self.middlewares = config['middlewares']
            for f in self.middlewares:
                f['handler'] = import_helper(f['handler'])

        # Register watched files
        if 'watched_dir' in config:
            self.watched_dir = config['watched_dir']

        if 'file_watcher' in config:
            self.file_watchers = config['file_watcher']

    def run(self):
        try:
            asyncio.run(self.supervisor())
        except KeyboardInterrupt as e:
            print("Caught keyboard interrupt. Canceling tasks...")
            self.cleaner()

    async def supervisor(self):
        # TODO: task monitor
        # file monitor
        filewatcher = asyncio.create_task(self.filewatcher())
        # dispatcher
        dispatcher = asyncio.create_task(self.dispatcher())

        # wait for all job to be finished
        await filewatcher
        await dispatcher

    async def dispatcher(self):
        try:
            while True:
                # get event from event queue
                event = await self.events.get()
                print("event:", event)
                if event['key'] == 'EXIT':
                    break

                # check if event key exist
                if not event['key'] in [r['event'] for r in self.registered]:
                    print("Event not exist, key error")

                # check and fetch middleware
                middlewares = []
                for middleware in self.middlewares:
                    if event['key'] in middleware['inject']:
                        middlewares.append(middleware)

                # get all handlers registered for this event
                funcs = [r for r in self.registered if r['event'] == event['key']]

                # throw error is no handler for this event
                if len(funcs) == 0:
                    print(f"No handler for key {event['key']}")

                # trigger event
                for f in funcs:
                    # periodic trigger
                    if "periodic" in f.keys():
                        task = asyncio.create_task(
                            self.periodic_task_wrapper(f["periodic"]["interval"],
                                                       f['handler'],
                                                       event,
                                                       middlewares))
                    # normal trigger
                    else:
                        task = asyncio.create_task(
                            self.task_wrapper(f['handler'],
                                              event,
                                              middlewares))
                    self.tasks.append(task)
        except Exception as e:
            # TODO: handle error
            print('Dispatcher Error: ', e)
            self.events.task_done()

    async def filewatcher(self):
        try:
            async for changes in awatch(*self.watched_dir):
                for change in changes:
                    if change[0] <= 2:
                        # print(change)
                        filename = os.path.basename(change[1])
                        for watcher in self.file_watchers:
                            # print(filename, watcher)
                            if re.match(watcher['name'], filename):
                                self.trigger({
                                    "key": watcher['trigger']['event'],
                                    "data": {
                                        "file": change[1],
                                        "job_id": "_".join(filename.split('_')[1:])
                                    }
                                })
        except Exception as e:
            # TODO: handle error
            print('Filewatcher Error: ', e)
            self.events.task_done()

    async def task_wrapper(self, func, event, middlewares):

        try:
            # execute middleware
            for mw in middlewares:
                event = mw['handler'](event)

            # execute the handler
            func(event, self.trigger)
        except Exception as e:
            # TODO: handle error
            print('Task Error: ', e)
            self.events.task_done()

    async def periodic_task_wrapper(self, dt, func, event, middlewares):
        try:
            # execute middleware
            for mw in middlewares:
                event = mw['handler'](event)

            # execute periodically
            while True:
                # execute the handler
                func(event, self.trigger)
                # wait for dt seconds
                await asyncio.sleep(dt)
        except Exception as e:
            # TODO: handle error
            print('Error: ', e)
            self.events.task_done()

    def aggregator(self, event):
        # TODO: wait for multiple event to execute the handler
        pass

    def trigger(self, event):
        self.events.put_nowait(event)

    def cleaner(self):
        """
        Clean all the task
        :return:
        """
        for task in self.tasks:
            task.cancel()
