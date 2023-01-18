import configparser
import os
import logging, sys
logger = logging.getLogger(__name__)

from . import cwb_interface
from .cwb_interface import cwb_root_logon
from . import user_parameters
from .config.constants import CWB_STAGE
from .config import CWBConfig


class pycWB:
    def __init__(self, config_file, create_dirs=True, log_file=None, log_level='INFO'):
        # setup logger
        logger_init(log_file, log_level)

        # load config
        self.config = CWBConfig(config_file)

        # setup ROOT
        self.ROOT, self.gROOT = cwb_root_logon(self.config)

        # setup project directories
        if create_dirs:
            self.setup_project_dirs(self.config.working_dir)

    def cwb_inet2G(self, run_id, f_name, j_stage, u_name="", eced=False, inet_option=None):
        _, ext = os.path.splitext(f_name)

        logger.info(f"Loading user parameters from {f_name}")
        file_name = ""
        if ext.lower() == '.c':
            file_name = f_name
        elif ext.lower() == '.yaml':
            self.user_params_with_yaml(f_name)
        else:
            logger.error(f"Unknown file extension {ext}")
            return 1

        # check analysis (from cwb_eparameters.C)
        self.ROOT.CheckAnalysis()

        cwb_interface.cwb_inet2G(self.ROOT, self.gROOT, self.config, run_id, CWB_STAGE[j_stage],
                                 inet_option=inet_option, file_name=file_name)

    @staticmethod
    def load_config(config_file):
        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation(),
            inline_comment_prefixes='#')
        config.optionxform = str
        config.read(config_file)
        logger.info(f"Loaded config from {config_file}")
        return config

    def cwb_load_macro(self, file_name):
        self.gROOT.LoadMacro(self.config.cwb_macros + "/" + file_name)
        logger.info(f"Loaded macro from {file_name}")

    def user_params_with_yaml(self, file_name):
        user_parameters.load_yaml(self.gROOT, file_name)
        logger.info(f"Loaded user parameters from {file_name}")

    def setup_project_dirs(self, working_dir=os.getcwd()):
        if not os.path.exists('plugins'): os.symlink(f"{self.config.cwb_install}/etc/cwb/plugins", 'plugins')
        for dir in ['input', 'data', 'tmp/public_html/reports', 'tmp/condor', 'tmp/node', 'report/dump']:
            os.makedirs(f'{working_dir}/{dir}', exist_ok=True)
        logger.info(f"Created project directories in {working_dir}")


def logger_init(log_file: str = None, log_level: str = 'INFO'):
    """
    Initialize logger
    :param log_file:
    :param log_level:
    :return:
    """
    # create logger
    logging.basicConfig(stream=sys.stdout, level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
