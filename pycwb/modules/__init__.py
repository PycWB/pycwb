import ROOT
import logging
from .cwb_2g import cwb_2g
logger = logging.getLogger(__name__)

if not hasattr(ROOT, "WDM"):
    ROOT.gSystem.Load("wavelet")
    logger.info("Loading wavelet library")