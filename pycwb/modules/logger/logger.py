import logging, sys

logger = logging.getLogger(__name__)


def logger_init(log_file: str = None, log_level: str = 'INFO'):
    """Initialize logger with format %(asctime)s - %(funcName)s - %(levelname)s - %(message)s

    :param log_file: log file path
    :type log_file: str
    :param log_level: log level, defaults to 'INFO'
    :type log_level: str, optional
    :return: None
    """
    # create logger
    if log_file:
        logging.basicConfig(filename=log_file, level=log_level,
                            format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(stream=sys.stdout, level=log_level,
                            format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s', datefmt='%y-%m-%d %H:%M:%S')

    logger.info("Logging initialized")
    logger.info("Logging level: " + log_level)
    logger.info("Logging file: " + str(log_file))