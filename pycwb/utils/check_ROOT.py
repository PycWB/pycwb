import logging, os, pycwb, site

logger = logging.getLogger(__name__)


def check_and_load_wavelet(ROOT):
    """
    Check if wavelet library is loaded, if not, try to load it from PyBurst

    :param ROOT: ROOT object
    :return:
    """
    if not hasattr(ROOT, "WDM"):
        logger.info("Loading wavelet library")
    try:
        pycwb_path = site.getsitepackages()[0]
        wavelet_path = f"{pycwb_path}/lib/wavelet"
        logger.info(f"Trying to load wavelet library from {wavelet_path}")
        ROOT.gInterpreter.AddIncludePath(f"{pycwb_path}/include")
        ROOT.gSystem.Load(wavelet_path)
    except:
        logger.error("Cannot find wavelet library in PyBurst, trying to load from system")
        try:
            ROOT.gSystem.Load("wavelet")
        except:
            logger.error("Cannot find wavelet library")
            raise Exception("Cannot find wavelet library")
