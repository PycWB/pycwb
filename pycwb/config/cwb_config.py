from configparser import ConfigParser, ExtendedInterpolation
import logging
import pycwb
from os import path, environ

logger = logging.getLogger(__name__)


class CWBConfig:
    def __init__(self, config_file):
        # check if config file exists
        if not path.exists(config_file):
            raise FileNotFoundError(f"Config file {config_file} does not exist.")
        # check the file type of config_file
        if not config_file.endswith('.ini'):
            raise ValueError(f"Config file {config_file} is not an .ini file.")

        config = self.load_ini(config_file)
        self.config = config
        self.cwb_install = config['CWB']['CWB_INSTALL']
        self.cwb_source = config['CWB']['CWB_SOURCE']
        if 'CWB_MACROS' in config['CWB']:
            self.cwb_macros = config['CWB']['CWB_MACROS']
        else:
            self.cwb_macros = f"{self.cwb_install}/etc/cwb/macros"  # macro path

        if 'CWB_PARAMETERS_FILE' in config['CWB']:
            self.cwb_parameters_file = config['CWB']['CWB_PARAMETERS_FILE']
        else:
            self.cwb_parameters_file = f"{self.cwb_install}/etc/cwb/macros/cwb2G_parameters.C"  # parameters file path

        if 'CWB_ANALYSIS' in config['CWB']:
            self.cwb_analysis = config['CWB']['CWB_ANALYSIS']
        else:
            self.cwb_analysis = '2G'

        self.use_ebbh = config.getboolean('LIB', '_USE_EBBH')
        self.use_root6 = config.getboolean('LIB', '_USE_ROOT6')
        self.use_lal = config.getboolean('LIB', '_USE_LAL')
        self.use_healpix = config.getboolean('LIB', '_USE_HEALPIX')
        self.use_icc = config.getboolean('LIB', '_USE_ICC')
        self.home_cvode = config['LIB']['HOME_CVODE'] if 'HOME_CVODE' in config['LIB'] else None

        self.working_dir = config['PROJECT']['WORK_DIR']

    @staticmethod
    def load_ini(config_file):
        config = ConfigParser(
            interpolation=ExtendedInterpolation(),
            inline_comment_prefixes='#')
        config.optionxform = str
        config.read(config_file)
        logger.info(f"Loaded config from {config_file}")
        return config

    def export_to_envs(self):
        config = self.config

        pycwb_path = path.dirname(path.abspath(pycwb.__file__))

        envs = {
            'LALINSPINJ_EXEC': 'lalapps_inspinj'
        }

        if self.use_lal:
            envs['_USE_LAL'] = "1"
        if self.use_healpix:
            envs['_USE_HEALPIX'] = "1"
        if self.use_root6:
            envs['_USE_ROOT6'] = "1"
        if self.use_ebbh:
            envs['_USE_EBBH'] = "1"

        for section in ['CWB', 'TOOLS', 'PROJECT']:
            for key in config[section].keys():
                envs[key] = config[section][key]

        envs['CWB_SCRIPTS'] = f"{self.cwb_install}/etc/cwb/scripts"
        # TODO: remove this by copying html files to installation directory
        envs['HOME_WAT'] = self.cwb_source
        envs['HOME_FRDISPLAY'] = f"{self.cwb_install}/bin"
        envs['HOME_CWB'] = f"{self.cwb_install}/etc/cwb"

        envs['CWB_ROOTLOGON_FILE'] = f"{pycwb_path}/vendor/dumb.c"
        envs['CWB_MACROS'] = self.cwb_macros
        envs['CWB_NETC_FILE'] = f"{self.cwb_macros}/cwb_net.C"
        envs['CWB_ANALYSIS'] = self.cwb_analysis
        envs['CWB_PARAMETERS_FILE'] = self.cwb_parameters_file
        envs['CWB_PARMS_FILES'] = self.cwb_parameters_file
        # envs['CWB_PARMS_FILES'] = f"{envs['CWB_ROOTLOGON_FILE']} {envs['CWB_PARAMETERS_FILE']} " \
        #                           f"{envs['CWB_UPARAMETERS_FILE']} {envs['CWB_EPARAMETERS_FILE']}"

        # TODO: figure out these parameter files
        envs['CWB_EMPARAMETERS_FILE'] = ""
        envs['CWB_PPARAMETERS_FILE'] = f"{self.cwb_macros}/cwb_pparameters.C"
        envs['CWB_EPARAMETERS_FILE'] = f"{self.cwb_macros}/cwb_eparameters.C"
        envs['CWB_PPARMS_FILES'] = envs['CWB_PPARAMETERS_FILE']
        # envs['CWB_PPARMS_FILES'] = f"{envs['CWB_ROOTLOGON_FILE']} {envs['CWB_PARAMETERS_FILE']} " \
        #                            f"{envs['CWB_UPARAMETERS_FILE']} {envs['CWB_EMPARAMETERS_FILE']} " \
        #                            f"{envs['CWB_EPARAMETERS_FILE']} {envs['CWB_PPARAMETERS_FILE']} " \
        #                            f"{envs['CWB_UPPARAMETERS_FILE']} {envs['CWB_EPPARAMETERS_FILE']}"
        envs['LD_LIBRARY_PATH'] = f"{self.cwb_install}/lib"
        for key in envs.keys():
            environ[key] = envs[key]

        logger.info("Environment variables loaded")
        return envs
