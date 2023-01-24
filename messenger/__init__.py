import asyncio
import importlib
from watchfiles import awatch
import os
import re
import logging


def import_helper(name):
    p, m = name.rsplit('.', 1)

    mod = importlib.import_module(p)
    met = getattr(mod, m)
    return met


def logger_init(log_file: str = None, log_level: str = 'INFO'):
    """
    Initialize logger
    :param log_file:
    :param log_level:
    :return:
    """
    # create logger
    logger = logging.getLogger('messenger')
    logger.setLevel(log_level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    # add file handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class Messenger:
    def __init__(self, config: dict):
        """
        Initialize the messenger
        :param config: config file in dict format
        """
        self.events = asyncio.Queue()
        self.tasks = []
        self.registered = []
        self.middlewares = []
        self.watched_dir = []
        self.file_watchers = []
        self.load_config(config)
        self.logger = logger_init()

        pass

    def load_config(self, config: dict):
        """
        Load config from config file
        :param config: config file in dict format
        """
        # load middleware
        if 'middleware' in config:
            for middleware in config['middleware']:
                self.middlewares.append({
                    "handler": import_helper(middleware['handler']),
                    "inject": middleware['inject']
                })

        # load file watcher
        if 'watched_dir' in config:
            self.watched_dir = config['watched_dir']

        if 'filewatcher' in config:
            for watcher in config['filewatcher']:
                self.file_watchers.append({
                    "name": watcher['name'],
                    "trigger": watcher['trigger']
                })
                # self.watched_dir.append(watcher['dir'])

        # load event handler
        if 'events' in config:
            for event in config['events']:
                self.registered.append({
                    "event": event['event'],
                    "handler": import_helper(event['handler'])
                })
        else:
            self.logger.error("No event handler registered")

        # load periodic event handler
        if 'periodic' in config:
            for event in config['periodic']:
                self.registered.append({
                    "event": event['event'],
                    "periodic": {
                        "interval": event['interval']
                    },
                    "handler": import_helper(event['handler'])
                })

    def run(self, starter):
        """
        Run the messenger
        :param starter:
        :return:
        """
        try:
            self.trigger(starter)
            asyncio.run(self.supervisor())
        except KeyboardInterrupt as e:
            self.logger.error("Keyboard interrupt, exit")
            self.cleaner()

    async def supervisor(self):
        """
        Supervisor to run the event loop
        """
        # TODO: task monitor
        # file monitor
        filewatcher = asyncio.create_task(self.filewatcher())
        # dispatcher
        dispatcher = asyncio.create_task(self.dispatcher())

        # wait for all job to be finished
        await filewatcher
        await dispatcher

    async def dispatcher(self):
        """
        Dispatch the event to the handler
        :return:
        """
        try:
            while True:
                # get event from event queue
                event = await self.events.get()
                self.logger.info("Event: {}".format(event))
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
            # log the error
            self.logger.error(e)
            self.logger.error("Error in dispatcher, exit")
            self.events.task_done()

    async def filewatcher(self):
        """
        Watch for file changes
        :return:
        """
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
            self.logger.error(e)
            self.logger.error("Error in filewatcher, exit")
            self.events.task_done()

    async def wait(self):
        """
        Wait for all task to be finished
        :return:
        """
        await self.events.join()

    async def exit(self):
        """
        Exit the program
        :return:
        """
        self.trigger({"key": "EXIT"})
        self.cleaner()

    async def task_wrapper(self, func, event, middlewares):

        try:
            # execute middleware
            for mw in middlewares:
                event = mw['handler'](event)

            # execute the handler
            func(event, self.trigger)
        except Exception as e:
            # TODO: handle error
            self.logger.error(e)
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
            self.logger.error(e)
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
