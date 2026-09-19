"""Fixed H1/L1 geometry used by cWB 6.4.6.9 (wat/detector.cc).

These are versioned physical inputs, not a numerical approximation to LAL.
The default Detector model remains LAL. Other instruments have not been
validated for this release model and must not silently mix geometries.
"""

import numpy as np

VECTORS = {
    "H1": (
        (-2161414.928, -3834695.183, 4600350.224),
        (-0.223891216, 0.799830697, 0.556905359),
        (-0.913978490, 0.026095321, -0.404922650),
    ),
    "L1": (
        (-74276.04192, -5496283.721, 3224257.016),
        (-0.954574615, -0.141579994, -0.262187738),
        (0.297740169, -0.487910627, -0.820544948),
    ),
}


def apply_release_geometry(detector, model):
    if model != "cwb_6.4.6.9":
        raise ValueError(f"Unknown detector geometry model: {model}")
    if detector.name not in VECTORS:
        raise ValueError("cwb_6.4.6.9 geometry currently supports H1 and L1 only")
    from astropy.coordinates import EarthLocation
    from astropy import units

    r, x, y = (np.array(value, dtype=np.float64) for value in VECTORS[detector.name])
    detector.vertex_vec_earth_centered = r
    detector.x_vec_earth_centered = x
    detector.y_vec_earth_centered = y
    detector.x_response = -np.outer(x, x) / 2
    detector.y_response = -np.outer(y, y) / 2
    detector.response = detector.y_response - detector.x_response
    # Keep geographic metadata consistent with the selected vectors. Do not
    # reconstruct the response from these angles: that would renormalize the
    # release's literal rounded arm vectors.
    location = EarthLocation.from_geocentric(*r, unit=units.m)
    lon, lat = location.lon.rad, location.lat.rad
    detector.longitude, detector.latitude = float(lon), float(lat)
    detector.altitude = float(location.height.value)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    for label, arm in [("x", x), ("y", y)]:
        e, n, u = float(arm @ east), float(arm @ north), float(arm @ up)
        setattr(detector, label + "_azimuth", float(np.arctan2(e, n) % (2 * np.pi)))
        setattr(detector, label + "_altitude", float(np.arctan2(u, np.hypot(e, n))))
