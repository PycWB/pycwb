import os

import ROOT
import logging

logger = logging.getLogger(__name__)


class WDMXTalkCatalog:
    def __init__(self, catalog=None):
        if not catalog:
            self.catalog = ROOT.monster()
        elif isinstance(catalog, str):
            self.catalog = ROOT.monster()
            self.load_MRA(catalog)
        elif isinstance(catalog, ROOT.monster):
            self.catalog = catalog
        else:
            raise ValueError("catalog must be a string or a ROOT.monster object")

    def load_MRA(self, file):
        logger.info("Loading catalog of WDM cross-talk coefficients: %s", file)
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File {file} does not exist")
        else:
            self.catalog.read(file)

    @property
    def tag(self):
        return self.catalog.tag

    @property
    def beta_order(self):
        return self.catalog.BetaOrder

    @property
    def precision(self):
        return self.catalog.precision
    
    @property
    def n_res(self):
        return self.catalog.nRes
    
    @property
    def layers(self):
        return self.catalog.layers

    def get_wdmMRA(self):
        return ROOT.monster(self.catalog)

    def check_layers_with_MRAcatalog(self, l_low, l_high, n_res):
        """
        check if analysis layers are contained in the MRAcatalog.
        The layers are the number of layers along the frequency axis rateANA/(rateANA>>level)

        :param l_low: low level of the decomposition
        :type l_low: int
        :param l_high: high level of the decomposition
        :type l_high: int
        :param n_res: number of resolutions
        :type n_res: int
        :return: None
        """
        check_layers = 0
        for i in range(l_low, l_high + 1):
            # the decomposition level
            level = l_high + l_low - i
            # number of layers along the frequency axis rateANA/(rateANA>>level)
            layers = 2 ** level if level > 0 else 0
            for j in range(self.n_res):
                if layers == self.layers[j]:
                    check_layers += 1

        if check_layers != n_res:
            logger.error("analysis layers do not match the MRA catalog")
            logger.error("analysis layers : ")
            for level in range(l_high, l_low - 1, -1):
                layers = 1 << level if level > 0 else 0
                logger.error("level : %s layers : %s", level, layers)

            logger.error("MRA catalog layers : ")
            for i in range(self.n_res):
                logger.error("layers : %s", self.layers[i])
            raise ValueError("analysis layers do not match the MRA catalog")


