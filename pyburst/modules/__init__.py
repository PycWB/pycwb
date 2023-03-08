import ROOT
import os, logging
import pyburst
from .cwb_2g import cwb_2g
logger = logging.getLogger(__name__)

if not hasattr(ROOT, "WDM"):
    try:
        pyburst_path = os.path.dirname(pyburst.__file__)
        logger.info("Loading wavelet library from " + f"{pyburst_path}/vendor/lib/wavelet")
        ROOT.gSystem.Load(f"{pyburst_path}/vendor/lib/wavelet")
    except:
        logger.error("Cannot find wavelet library in pyburst, trying to load from system")
        try:
            ROOT.gSystem.Load("wavelet")
        except:
            logger.error("Cannot load wavelet library")