import ROOT
import numpy as np


def convert_to_cwb_detector(detector):
    """
    Convert a ROOT detector object to a CWB detector object.

    Args:
        detector (ROOT.TDetector): The ROOT detector object to convert.

    Returns:
        dict: A dictionary representing the CWB detector.
    """
    ifo_params = ROOT.detectorParams()
    ifo_params.name = detector.name.encode('utf-8')
    ifo_params.latitude = np.rad2deg(detector.latitude)
    ifo_params.longitude = np.rad2deg(detector.longitude)
    ifo_params.elevation = detector.altitude
    ifo_params.AltX = detector.x_altitude
    ifo_params.AzX = np.rad2deg(detector.x_azimuth)
    ifo_params.AltY = detector.y_altitude
    ifo_params.AzY = np.rad2deg(detector.y_azimuth)

    ifo = ROOT.detector(ifo_params)

    return ifo