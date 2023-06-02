import aiofiles
import logging

logger = logging.getLogger(__name__)


async def write_file(filename, data, mode='w'):
    """
    write data to file

    Parameters
    ----------
    filename : str
        file name
    data : str | bytes
        data to write
    mode : str
        file open mode
    """
    try:
        # save event to file
        async with aiofiles.open(filename, mode) as f:
            await f.write(data)
    except Exception as e:
        logger.error(e)
