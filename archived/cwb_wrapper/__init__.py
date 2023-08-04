import configparser
import os
import logging, sys
from . import cwb_interface
from .cwb_interface import cwb_root_logon
from .config import user_parameters
from .constants.cwb_dict import CWB_STAGE
from .config import CWBConfig

logger = logging.getLogger(__name__)


class pycWB:
    def __init__(self, config_file, create_dirs=True, log_file=None, log_level='INFO', minimal_load=False):
        # setup logger
        logger_init(log_file, log_level)

        # load config
        self.config = CWBConfig(config_file)
        self.config.export_to_envs()

        # setup ROOT
        if minimal_load:
            import ROOT
            ROOT.gSystem.Load(f"{self.config.cwb_install}/lib/cwb")
        else:
            self.ROOT, self.gROOT = cwb_root_logon(self.config)

        # setup project directories
        if create_dirs:
            self.setup_project_dirs(self.config.working_dir)

    def cwb_inet2G(self, run_id, f_name, j_stage, u_name="", eced=False, inet_option=None):
        file_name = self.init_cfg(f_name)

        return cwb_interface.cwb_inet2G(self.config, run_id, CWB_STAGE[j_stage],
                                        inet_option=inet_option, file_name=file_name)

    def cwb_xnet(self, run_id, f_name, j_stage, inet_option=None):
        file_name = self.init_cfg(f_name)

        return cwb_interface.cwb_xnet(self.config, run_id, CWB_STAGE[j_stage],
                                      inet_option=inet_option, file_name=file_name)

    def cwb_xnet_new(self, run_id, f_name, j_stage, inet_option=None):
        file_name = self.init_cfg(f_name)

        return cwb_interface.cwb_xnet_new(self.config, run_id, CWB_STAGE[j_stage],
                                          inet_option=inet_option, file_name=file_name)

    def init_cfg(self, f_name):
        _, ext = os.path.splitext(f_name)

        logger.info(f"Loading user parameters from {f_name}")
        file_name = ""
        if ext.lower() == '.c':
            file_name = f_name
        elif ext.lower() == '.yaml':
            user_parameters.load_yaml(f_name, load_to_root=True)
        else:
            logger.error(f"Unknown file extension {ext}")
            raise ValueError(f"Unknown file extension {ext}")

        # check analysis (from cwb_eparameters.C)
        self.ROOT.CheckAnalysis()
        return file_name

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
    if log_file:
        logging.basicConfig(filename=log_file, level=log_level,
                            format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(stream=sys.stdout, level=log_level,
                            format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s', datefmt='%y-%m-%d %H:%M:%S')
