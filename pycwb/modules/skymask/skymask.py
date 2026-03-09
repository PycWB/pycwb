import numpy as np

try:
    import healpy as hp
except Exception:  # pragma: no cover - optional dependency
    hp = None


def _angular_distance_deg(phi1_deg, theta1_deg, phi2_deg, theta2_deg):
    """Angular distance on sphere for lon/lat points in degrees."""
    lon1 = np.deg2rad(phi1_deg)
    lat1 = np.deg2rad(theta1_deg)
    lon2 = np.deg2rad(phi2_deg)
    lat2 = np.deg2rad(theta2_deg)

    dlon = lon2 - lon1
    sin_lat1 = np.sin(lat1)
    sin_lat2 = np.sin(lat2)
    cos_lat1 = np.cos(lat1)
    cos_lat2 = np.cos(lat2)
    cos_d = sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * np.cos(dlon)
    cos_d = np.clip(cos_d, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_d))


def _cwb_to_geo(phi_deg, theta_deg):
    """Convert cWB sky coordinates to geographic-like lon/lat in degrees."""
    phi_geo = np.where(phi_deg > 180.0, phi_deg - 360.0, phi_deg)
    theta_geo = 90.0 - theta_deg
    return phi_geo, theta_geo


def make_sky_mask(sky_mask, theta: float, phi: float, radius: float, ROOT_module=None):
    """
    Fill a sky mask with 1 inside a circular region and 0 outside.

    Parameters
    ----------
    sky_mask : ROOT skymap-like object
        Output sky mask object exposing cWB skymap API.
    theta : float
        Latitude/declination in degrees, range [-90, 90].
    phi : float
        Longitude/right-ascension in degrees, range [0, 360].
    radius : float
        Radius of circular region in degrees.
    ROOT_module : module, optional
        Kept for backward compatibility; not used by python-native implementation.
    """
    l = sky_mask.size()
    healpix = sky_mask.getOrder()

    if abs(theta) > 90 or (phi < 0 or phi > 360) or radius <= 0 or l <= 0:
        raise ValueError(
            "cwb::MakeSkyMask : wrong input parameters !!! "
            "if(fabs(theta)>90)   cout << theta << \" theta must be in the range [-90,90]\" << endl;"
            "if(phi<0 || phi>360) cout << phi << \" phi must be in the range [0,360]\" << endl;"
            "if(radius<=0)        cout << radius << \" radius must be > 0\" << endl;"
            "if(L<=0)             cout << L << \" SkyMask size must be > 0\" << endl;"
            "EXIT(1);"
        )

    if healpix:
        nside = 2 ** int(healpix)
        skyres = hp.nside2pixarea(nside, degrees=True) if hp is not None else (4.0 * np.pi * (180.0 / np.pi) ** 2) / (12 * nside * nside)
        if radius < np.sqrt(skyres):
            radius = np.sqrt(skyres)
    else:
        if radius < sky_mask.sms:
            radius = sky_mask.sms

    # HEALPix path: native healpy disk query
    if healpix and hp is not None:
        nside = 2 ** int(healpix)
        vec = hp.ang2vec(phi, theta, lonlat=True)
        selected = hp.query_disc(nside, vec, np.deg2rad(radius), inclusive=True)
        mask_values = np.zeros(l, dtype=np.int8)
        selected = selected[(selected >= 0) & (selected < l)]
        mask_values[selected] = 1
        for idx, value in enumerate(mask_values):
            sky_mask.set(idx, int(value))
        return

    # Generic path: evaluate using pixel centers exposed by skymap object
    pix_phi = np.empty(l, dtype=np.float64)
    pix_theta = np.empty(l, dtype=np.float64)
    for idx in range(l):
        pix_phi[idx] = sky_mask.getPhi(idx)
        pix_theta[idx] = sky_mask.getTheta(idx)

    pix_phi_geo, pix_theta_geo = _cwb_to_geo(pix_phi, pix_theta)
    d_omega = _angular_distance_deg(phi, theta, pix_phi_geo, pix_theta_geo)
    inside = d_omega <= radius
    for idx in range(l):
        sky_mask.set(idx, int(inside[idx]))
