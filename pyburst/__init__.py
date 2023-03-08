import logging, sys
from .config import user_parameters
from ._version import __version__

logger = logging.getLogger(__name__)


def logger_init(log_file: str = None, log_level: str = 'INFO'):
    """
    Initialize logger
    :param log_file:
    :param log_level:
    :return:
    """
    # create logger
    if log_file:
        logging.basicConfig(filename=log_file, level=log_level,
                            format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(stream=sys.stdout, level=log_level,
                        format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s', datefmt='%y-%m-%d %H:%M:%S')
