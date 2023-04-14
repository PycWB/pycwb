from pycbc.detector import Detector
import numpy as np


def spherical_to_cartesian(ra, dec):
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.array([x, y, z])


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    ra = np.arctan2(y, x)
    dec = np.arcsin(z / r)
    return ra, dec


def max_delay(ifos):
    # find all combinations of detectors
    ifo_pairs = [(ifos[i], ifos[j]) for i in range(len(ifos)) for j in range(i + 1, len(ifos))]

    # find the maximum delay between any two detectors
    max_delay = 0
    for ifo1, ifo2 in ifo_pairs:
        det1 = Detector(ifo1)
        det2 = Detector(ifo2)

        # find the orientation of the detectors
        v1 = np.array(det1.optimal_orientation(0))
        v2 = np.array(det2.optimal_orientation(0))
        v1_cart = spherical_to_cartesian(v1[0], v1[1])
        v2_cart = spherical_to_cartesian(v2[0], v2[1])
        v3_cart = v2_cart - v1_cart
        v3 = cartesian_to_spherical(v3_cart[0], v3_cart[1], v3_cart[2])

        delay = abs(det1.time_delay_from_detector(det2, v3[0], v3[1], 0))

        max_delay = max(max_delay, delay)

    return max_delay
