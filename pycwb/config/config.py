from .user_parameters import load_yaml
import os.path
import logging

logger = logging.getLogger(__name__)


class DQFile:
    def __init__(self, ifo, file, dq_cat, shift, invert, c4):
        self.ifo = ifo
        self.file = file
        self.dq_cat = dq_cat
        self.shift = shift
        self.invert = invert
        self.c4 = c4


class Config:
    def __init__(self, file_name):
        self.outputDir = None
        self.logDir = None
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
        self.dq_files = []

        params = load_yaml(file_name, load_to_root=False)

        for key in params:
            setattr(self, key, params[key])

        self.add_derived_key()
        self.check_file(self.MRAcatalog)
        self.check_lagStep()

    def add_derived_key(self):
        """
        Add derived key to the user parameters
        :param params:
        :return:
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

        # derive number of IFOs and DQFs
        self.nIFO = len(self.ifo)
        self.nDQF = len(self.DQF)

        # convert DQF to object
        for dqf in self.DQF:
            self.dq_files.append(DQFile(dqf[0], dqf[1], dqf[2], dqf[3], dqf[4], dqf[5]))

    @staticmethod
    def check_file(file_name):
        """
        Check if file exists
        :param file_name:
        :return:
        """
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f"File {file_name} does not exist")

    def check_lagStep(self):
        """
        Check if lagStep compatible with WDM parity

        his condition is necessary to avoid mixing between odd
        and even pixels when circular buffer is used for lag shift
        The MRAcatalog distinguish odd and even pixels

        :param config:
        :param net:
        :return:
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
