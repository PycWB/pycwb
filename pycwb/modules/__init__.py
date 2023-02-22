import ROOT
import os, logging
import pycwb
from .cwb_2g import cwb_2g
logger = logging.getLogger(__name__)

if not hasattr(ROOT, "WDM"):
    logger.info("Loading wavelet library")
    try:
        pycwb_path = os.path.dirname(pycwb.__file__)
        ROOT.gSystem.Load(f"{pycwb_path}/vendor/lib/wavelet")
    except:
        logger.error("Cannot find wavelet library in pycwb, trying to load from system")
        try:
            ROOT.gSystem.Load("wavelet")
        except:
            logger.error("Cannot load wavelet library")