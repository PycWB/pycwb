from . import setup
from . import cwb
from . import user_parameters
import os, glob


class pycWB:
    def __init__(self, config_file, create_dirs=True):
        ROOT, gROOT, gSystem, gStyle, config = setup.init(config_file)
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
            f_name = self.config["DUMB"]["C_DUMB"]
        cwb.cwb_inet2G(self.ROOT, self.gROOT, self.config, run_id, f_name, j_stage, u_name, eced, inet_option)

    def cwb_load_macro(self, file_name):
        self.gROOT.LoadMacro(self.config['MACROS']['CWB_MACROS'] + "/" + file_name)

    def user_params_with_yaml(self, file_name):
        # Todo: check file to be yaml
        user_parameters.load_yaml(self.gROOT, file_name)

    def setup_project_dirs(self, working_dir=os.getcwd()):
        if not os.path.exists('plugins'): os.symlink(f"{self.config['CWB']['CWB_INSTALL']}/etc/cwb/plugins", 'plugins')
        for dir in ['input', 'data', 'tmp/public_html/reports', 'tmp/condor', 'tmp/node', 'report/dump']:
            os.makedirs(f'{working_dir}/{dir}', exist_ok=True)

    @staticmethod
    def setup_sim_data(detectors, working_dir=os.getcwd()):
        for det in detectors:
            with open(f"{working_dir}/input/{det}.frames", 'w') as t:
                t.write(glob.glob(f"{working_dir}/frames/*/{det}*.gwf")[0])

        with open(f'{working_dir}/input/inspiral.in', 'w') as t:
            t.write("931158200    931158600\n")
