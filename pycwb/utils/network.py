import numpy as np

from pycwb.utils.geometry import cartesian_to_spherical, local_to_earth_centered, spherical_to_cartesian

__all__ = [
    "local_to_earth_centered",
    "spherical_to_cartesian",
    "cartesian_to_spherical",
    "max_delay",
]


def max_delay(ifos):
    """Maximum light-travel time between any detector pair (seconds).

    Uses the simple geometric baseline: ``|r_i - r_j| / c``, consistent
    with the cWB ``getDelay("MAX")`` approach.
    """
    from astropy import constants
    from pycwb.types.detector import Detector

    c = float(constants.c.value)
    max_d = 0.0
    for i in range(len(ifos)):
        for j in range(i + 1, len(ifos)):
            d1 = Detector(ifos[i])
            d2 = Detector(ifos[j])
            baseline = np.linalg.norm(
                d1.vertex_vec_earth_centered - d2.vertex_vec_earth_centered
            )
            max_d = max(max_d, baseline / c)
    return max_d
