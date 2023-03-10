import logging, os, pyburst

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
        pyburst_path = os.path.dirname(pyburst.__file__)
        wavelet_path = f"{pyburst_path}/vendor/lib/wavelet"
        logger.info(f"Trying to load wavelet library from {wavelet_path}")
        ROOT.gSystem.Load(wavelet_path)
    except:
        logger.error("Cannot find wavelet library in PyBurst, trying to load from system")
        try:
            ROOT.gSystem.Load("wavelet")
        except:
            logger.error("Cannot find wavelet library")
            raise Exception("Cannot find wavelet library")
