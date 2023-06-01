"""
Class to store all user parameters, load parameters from yaml file and check parameters. The supported parameters and their
default values are defined in the provided schema. By default, the schema is the schema defined in
pycwb.constants.user_parameters_schema.
"""
import os.path
import logging

from ..types.wdm_xtalk import WDMXTalkCatalog
from ..types.data_quality_file import DQFile
from ..utils.network import max_delay
from ..utils.yaml_helper import load_yaml
from ..constants import user_parameters_schema

logger = logging.getLogger(__name__)


class Config:
    """
    Class to store user parameters

    Parameters
    ----------
    file_name : str
        Path to the yaml file containing the user parameters
    schema : dict
        Schema to validate the user parameters, default is the schema defined in pycwb.constants.user_parameters_schema
    """

    def __init__(self, file_name, schema=None):
        if schema is None:
            schema = user_parameters_schema

        self.outputDir = None
        self.logDir = None
        self.nproc = None
        self.cfg_gamma = None
        self.gamma = None
        self.fResample = None
        self.rateANA = None
        self.levelR = None
        self.inRate = None
        self.nRES = None
        self.l_low = None
        self.l_high = None
        self.l_white = None
        self.fLow = None
        self.fHigh = None
        self.whiteWindow = None
        self.filter_dir = None
        self.wdmXTalk = None
        self.MRAcatalog = None
        self.TDRate = None
        self.lagStep = None
        self.lagBuffer = None
        self.lagMode = None
        self.max_delay = None
        self.dq_files = []
        self.injection = {}
        self.WDM_beta_order = None
        self.WDM_precision = None
        self.WDM_level = []

        params = load_yaml(file_name, schema)

        for key in params:
            setattr(self, key, params[key])

        self.add_derived_key()
        self.check_file(self.MRAcatalog)
        if self.MRAcatalog:
            self.check_MRA_catalog()
        self.check_lagStep()

    def add_derived_key(self):
        """
        Add derived key to the user parameters, this method is called after loading the user parameters
        """

        self.gamma = self.cfg_gamma
        self.search = self.cfg_search

        # calculate analysis data rate
        if self.fResample > 0:
            self.rateANA = self.fResample >> self.levelR
        else:
            self.rateANA = self.inRate >> self.levelR

        self.nRES = self.l_high - self.l_low + 1

        # load WAT filter directory and set MRAcatalog
        if not self.filter_dir:
            self.filter_dir = os.environ['HOME_WAT_FILTERS']

        self.MRAcatalog = f"{self.filter_dir}/{self.wdmXTalk}"

        # calculate TDRate
        if self.fResample > 0:
            self.TDRate = (self.fResample >> self.levelR) * self.upTDF
        else:
            self.TDRate = (self.inRate >> self.levelR) * self.upTDF
        self.TDRate = float(self.TDRate)

        # derive number of IFOs and DQFs
        self.nIFO = len(self.ifo)
        self.nDQF = len(self.DQF)

        # convert DQF to object
        for dqf in self.DQF:
            self.dq_files.append(DQFile(dqf[0], dqf[1], dqf[2], dqf[3], dqf[4], dqf[5]))

        self.max_delay = max_delay(self.ifo)

        self.WDM_level = [int(self.l_high + self.l_low - i) for i in range(self.l_low, self.l_high + 1)]

    def get_lag_buffer(self):
        """
        Get lag buffer from configuration and update lag mode
        """
        if self.lagMode == "r":
            with open(self.lagFile, "r") as f:
                self.lagBuffer = f.read()
            self.lagMode = 's'
        else:
            self.lagBuffer = self.lagFile
            self.lagMode = 'w'

    @staticmethod
    def check_file(file_name):
        """
        Helper function to check if file exists

        Parameters
        ----------
        file_name : str
            Path to the file

        Raises
        ------
        FileNotFoundError
        """
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f"File {file_name} does not exist")

    def check_lagStep(self):
        """
        Check if lagStep compatible with WDM parity

        this condition is necessary to avoid mixing between odd
        and even pixels when circular buffer is used for lag shift
        The MRAcatalog distinguish odd and even pixels
        """
        rate_min = self.rateANA >> self.l_high
        dt_max = 1. / rate_min
        if rate_min % 1:
            logger.error("rate min=%s (Hz) is not integer", rate_min)
            raise ValueError("rate min=%s (Hz) is not integer", rate_min)
        if int(self.lagStep * rate_min + 0.001) & 1:
            logger.error("lagStep=%s (sec) is not a multple of 2*max_time_resolution=%s (sec)", self.lagStep,
                         2 * dt_max)
            raise ValueError("lagStep=%s (sec) is not a multple of 2*max_time_resolution=%s (sec)", self.lagStep,
                             2 * dt_max)
        if int(self.segEdge * rate_min + 0.001) & 1:
            logger.error("segEdge=%s (sec) is not a multple of 2*max_time_resolution=%s (sec)", self.segEdge,
                         2 * dt_max)
            raise ValueError("segEdge=%s (sec) is not a multple of 2*max_time_resolution=%s (sec)", self.segEdge,
                             2 * dt_max)
        if int(self.segMLS * rate_min + 0.001) & 1:
            logger.error("segMLS=%s (sec) is not a multple of 2*max_time_resolution=%s (sec)", self.segMLS,
                         2 * dt_max)
            raise ValueError("segMLS=%s (sec) is not a multple of 2*max_time_resolution=%s (sec)", self.segMLS,
                             2 * dt_max)

    def check_MRA_catalog(self):
        """
        Check if MRAcatalog exists

        Raises
        ------
        FileNotFoundError: if MRAcatalog does not exist
        """
        logger.info("Checking MRA catalog")
        wdm_MRA = WDMXTalkCatalog(self.MRAcatalog)

        # check layers
        wdm_MRA.check_layers_with_MRAcatalog(self.l_low, self.l_high, self.nRES)

        # update beta order and precision
        if wdm_MRA.tag != 0:
            logger.info(f"MRA catalog has tag {wdm_MRA.tag}, updating beta order and precision from MRA catalog")
            self.WDM_beta_order, self.WDM_precision = int(wdm_MRA.beta_order), int(wdm_MRA.precision)

