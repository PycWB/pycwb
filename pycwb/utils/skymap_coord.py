import numpy as np


def convert_cwb_to_geo(phi_deg, theta_deg):
    # for single point
    if isinstance(phi_deg, float) and isinstance(theta_deg, float):
        phi_deg = phi_deg - 360 if phi_deg > 180 else phi_deg
        theta_deg = -(theta_deg - 90)

        return phi_deg, theta_deg

    phi_deg[phi_deg > 180] -= 360
    theta_deg = -(theta_deg - 90)

    return phi_deg, theta_deg