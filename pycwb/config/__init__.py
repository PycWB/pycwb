from configparser import ConfigParser, ExtendedInterpolation
import logging
from os.path import exists

logger = logging.getLogger(__name__)


class CWBConfig:
    def __init__(self, config_file):
        # check if config file exists
        if not exists(config_file):
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
