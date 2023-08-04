import asyncio
from aiohttp import web
import logging
logger = logging.getLogger(__name__)


async def get_modules(request):
    return web.Response(text="Hello, world")


async def get_module_search_path(request):
    """
    Get the module search path
    :param request:
    :return:
    """
    return web.Response(text="Hello, world")


async def add_module_search_path(request):
    return web.Response(text="Hello, world")


async def remove_module_search_path(request):
    return web.Response(text="Hello, world")


async def trigger_event(request):
    return web.Response(text="Hello, world")


async def get_events(request):
    return web.Response(text="Hello, world")


async def update_events(request):
    return web.Response(text="Hello, world")


async def root_handler(request):
    return web.HTTPFound('/index.html')


async def run_server(web_app_dir="../messenger-web-interface/dist", host="127.0.0.1", port=8080):
    """
    Run the server
    :param web_app_dir: path to the web app, default to ../messenger-web-interface/dist
    :param host: host, default localhost
    :param port: port, default 8080
    :return:
    """

    app = web.Application()
    app.add_routes([
        web.get('/', root_handler),
    ])

    app.add_routes([web.static('/', web_app_dir, show_index=True)])

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    logger.info("Server started at http://{}:{}".format(host, port))
    logger.info("Web app directory: {}".format(web_app_dir))
    await site.start()


if __name__ == '__main__':
    logger.info = print

    async def main():
        await run_server()
        await asyncio.Event().wait()


    asyncio.run(main())
