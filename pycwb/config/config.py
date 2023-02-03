from .user_parameters import load_yaml
import os.path


class Config:
    def __init__(self, file_name):
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

        params = load_yaml(file_name, load_to_root=False)

        for key in params:
            setattr(self, key, params[key])

        self.add_derived_key()
        self.check_file(self.MRAcatalog)

    def add_derived_key(self):
        """
        Add derived key to the user parameters
        :param params:
        :return:
        """
        self.gamma = self.cfg_gamma

        # calculate analysis data rate
        if self.fResample > 0:
            self.rateANA = self.fResample >> self.levelR
        else:
            self.rateANA = self.inRate >> self.levelR

        self.nRES = self.l_high - self.l_low + 1

        if not self.filter_dir:
            self.filter_dir = os.environ['HOME_WAT_FILTERS']

        self.MRAcatalog = f"{self.filter_dir}/{self.wdmXTalk}"

    @staticmethod
    def check_file(file_name):
        """
        Check if file exists
        :param file_name:
        :return:
        """
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f"File {file_name} does not exist")
