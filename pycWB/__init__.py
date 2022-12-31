import os

from . import cwb_interface
from .cwb_interface import root_setup
from . import user_parameters
from .config.constants import CWB_STAGE


class pycWB:
    def __init__(self, config_file, create_dirs=True):
        ROOT, gROOT, gSystem, gStyle, config = root_setup.init(config_file)
        self.ROOT = ROOT
        self.gROOT = gROOT
        self.gSystem = gSystem
        self.gStyle = gStyle
        self.config = config

        if create_dirs:
            self.setup_project_dirs()

    def cwb_inet2G(self, run_id, f_name, j_stage, u_name="", eced=False, inet_option=None):
        _, ext = os.path.splitext(f_name)
        if ext.lower() == '.c':
            pass
        elif ext.lower() == '.yaml':
            self.user_params_with_yaml(f_name)
            print(f"cwb_inet2G: loaded {f_name}")

        cwb_interface.cwb_inet2G(self.ROOT, self.gROOT, self.config, run_id, CWB_STAGE[j_stage], inet_option=inet_option)

    def cwb_load_macro(self, file_name):
        self.gROOT.LoadMacro(self.config['MACROS']['CWB_MACROS'] + "/" + file_name)

    def user_params_with_yaml(self, file_name):
        # Todo: check file to be yaml
        user_parameters.load_yaml(self.gROOT, file_name)

    def setup_project_dirs(self, working_dir=os.getcwd()):
        if not os.path.exists('plugins'): os.symlink(f"{self.config['CWB']['CWB_INSTALL']}/etc/cwb/plugins", 'plugins')
        for dir in ['input', 'data', 'tmp/public_html/reports', 'tmp/condor', 'tmp/node', 'report/dump']:
            os.makedirs(f'{working_dir}/{dir}', exist_ok=True)
