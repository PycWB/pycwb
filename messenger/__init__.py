import asyncio
import importlib


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
        self.load_config(config)

        self.trigger(starter)

        pass

    def load_config(self, config):
        self.registered = config['events']
        for f in self.registered:
            f['handler'] = import_helper(f['handler'])
        self.middlewares = config['middlewares']
        for f in self.middlewares:
            f['handler'] = import_helper(f['handler'])

    def run(self):
        try:
            asyncio.run(self.supervisor())
        except KeyboardInterrupt as e:
            print("Caught keyboard interrupt. Canceling tasks...")
            self.cleaner()

    async def supervisor(self):
        # TODO: task monitor
        # TODO: file monitor
        await self.dispatcher()

    async def dispatcher(self):
        while True:
            event = await self.events.get()
            print("event:", event)
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
                                                   f['handler'],
                                                   event,
                                                   middlewares))
                else:
                    task = asyncio.create_task(
                        self.task_wrapper(f['handler'],
                                          event,
                                          middlewares))
                self.tasks.append(task)

    async def task_wrapper(self, func, event, middlewares):

        try:
            # execute middleware
            for mw in middlewares:
                event = mw['handler'](event)

            func(event, self.trigger)
        except Exception as e:
            print('Error: ', e)
            self.events.task_done()
        # self.events.task_done()

    async def periodic_task_wrapper(self, dt, func, event, middlewares):
        try:
            # execute middleware
            for mw in middlewares:
                event = mw['handler'](event)

            while True:
                func(event, self.trigger)
                await asyncio.sleep(dt)
        except Exception as e:
            print('Error: ', e)
            self.events.task_done()

    def aggregator(self, event):
        pass

    def trigger(self, event):
        self.events.put_nowait(event)

    def cleaner(self):
        for task in self.tasks:
            task.cancel()
