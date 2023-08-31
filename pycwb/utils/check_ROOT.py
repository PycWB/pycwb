import logging
import site

logger = logging.getLogger(__name__)


def check_and_load_wavelet(ROOT):
    """
    Check if wavelet library is loaded, if not, try to load it from PyBurst

    :param ROOT: ROOT object
    :return:
    """
    if not hasattr(ROOT, "WDM"):
        print("Loading wavelet library")
    try:
        site_packages = site.getsitepackages()[0]
        wavelet_path = f"{site_packages}/lib/wavelet"
        print(f"Trying to load wavelet library from {wavelet_path}")
        ROOT.gInterpreter.AddIncludePath(f"{site_packages}/include")
        ROOT.gSystem.Load(wavelet_path)
    except:
        print("Cannot find wavelet library in PyBurst, trying to load from system")
        try:
            ROOT.gSystem.Load("wavelet")
        except:
            print("Cannot find wavelet library")
            raise Exception("Cannot find wavelet library")
