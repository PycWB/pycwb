import logging
import os
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
        wavelet_path = f"{site_packages}/lib/wavelet.so"
        # if wavelet_path does not exist, try the additional path
        if not os.path.exists(wavelet_path):
            # if site_packages ends with dist-packages, replace it with site-packages for a second try
            if site_packages.endswith("dist-packages"):
                wavelet_path_additional = wavelet_path.replace("dist-packages", "site-packages")
                if not os.path.exists(wavelet_path_additional):
                    raise Exception(f"Cannot find wavelet library in {wavelet_path} or {wavelet_path_additional}")
                else:
                    wavelet_path = wavelet_path_additional
                    site_packages = site_packages.replace("dist-packages", "site-packages")
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
