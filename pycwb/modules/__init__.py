import ROOT
import logging

logger = logging.getLogger(__name__)

if not hasattr(ROOT, "WDM"):
    ROOT.gSystem.Load("wavelet")
    logger.info("WDM not found, loading wavelet library")