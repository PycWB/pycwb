import logging, sys
import traceback

logger = logging.getLogger(__name__)


def logger_init(log_file: str = None, log_level: str = 'INFO', silent: bool = False, worker_prefix: str = None):
    """Initialize logger with format %(asctime)s - %(funcName)s - %(levelname)s - %(message)s

    :param log_file: log file path
    :type log_file: str
    :param log_level: log level, defaults to 'INFO'
    :type log_level: str, optional
    :return: None
    """
    # create logger
    if worker_prefix:
        format_str = f'[{worker_prefix}]' + '%(asctime)s - %(funcName)s - %(levelname)s - %(message)s'
    else:
        format_str = '%(asctime)s - %(funcName)s - %(levelname)s - %(message)s'


    if log_file:
        logging.basicConfig(filename=log_file, level=log_level, force=True,
                            format=format_str, datefmt='%y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(stream=sys.stdout, level=log_level, force=True,
                            format=format_str, datefmt='%y-%m-%d %H:%M:%S')

    if not silent:
        logger.info("Logging initialized")
        logger.info("Logging level: " + log_level)
        logger.info("Logging file: " + str(log_file))


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        if buf.rstrip():
            # Extract the current stack frame to find the caller's function name
            fn = traceback.extract_stack()[-3][2]
            # Now we append the function name at the beginning of the log message
            self.logger.log(self.log_level, f"[{fn}] {buf.rstrip()}")

    def flush(self):
        pass


def log_prints():
    # Replace stdout with logging
    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    # Optionally, replace stderr as well
    sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)
