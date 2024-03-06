import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import healpy as hp
import logging

from pycwb.utils.skymap_coord import convert_cwb_to_geo

logger = logging.getLogger(__name__)


def plot_world_map(phi, theta, filename=None):
    from mpl_toolkits.basemap import Basemap

    plt.figure(figsize=(12, 8))
    phi -= 360 if phi > 180 else 0
    theta = -(theta - 90)

    map = Basemap(projection='cyl')

    map.drawcoastlines(linewidth=0.25)
    map.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0])
    map.drawmeridians(np.arange(map.lonmin, map.lonmax + 30, 30), labels=[0, 0, 0, 1])
    lons = [phi]
    lats = [theta]

    x, y = map(lons, lats)

    map.scatter(x, y, marker='*', color='blue', s=200)

    if filename:
        plt.savefig(filename)
        plt.clf()
    else:
        return plt


def plot_skymap_contour(skymap_statistic, key="nProbability", reconstructed_loc=None, detector_loc=None, filename=None,
                        resolution=2):
    mplstyle.use('fast')

    # get the [key] property from the skymap
    skymap = np.array(skymap_statistic[key])

    # create theta and phi, add one extra point and remove it to make sure
    # the last point is not overlapped with the first point
    theta = np.linspace(0, np.pi, 180 * resolution + 1)[:-1]
    phi = np.linspace(0, 2 * np.pi, 360 * resolution + 1)[:-1]

    # create mesh
    theta, phi = np.meshgrid(theta, phi)
    flatten_theta = theta.flatten()
    flatten_phi = phi.flatten()

    # convert to HEALPix indices and get the values for each point
    healpix_indices = hp.ang2pix(2 ** 7, flatten_theta, flatten_phi)
    if len(skymap) < len(healpix_indices):
        raise ValueError('The length of the skymap is smaller than the length of the healpix indices')

    values = skymap[healpix_indices]

    # reshape to 2D map
    values = np.reshape(values, theta.shape)

    # convert to degrees
    theta_deg = np.rad2deg(theta)
    phi_deg = np.rad2deg(phi)

    # transform coordinates
    phi_deg, theta_deg = convert_cwb_to_geo(phi_deg, theta_deg)

    # phi_deg[phi_deg < -180] += 360
    # phi_deg[phi_deg > 180] -= 360
    # theta_deg[theta_deg < -90] += 180
    # theta_deg[theta_deg > 90] -= 180
    # phi_deg += 180
    # theta_deg += 90

    # set the value of the point to nan to prevent plotting 0 value
    values[values == 0] = np.nan

    #################
    # plot histogram
    #################
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.hist2d(phi_deg.ravel(), theta_deg.ravel(), weights=values.ravel(), bins=(360 * resolution, 180 * resolution))

    if reconstructed_loc:
        rec_x, rec_y = convert_cwb_to_geo(reconstructed_loc[0], reconstructed_loc[1])
        plt.scatter(rec_x, rec_y, marker='*', color='red', label='reconstructed position')

    if detector_loc:
        det_x, det_y = convert_cwb_to_geo(detector_loc[0], detector_loc[1])
        plt.scatter(det_x, det_y, marker='.', color='blue', label='detector position')
    # plt the ticks at the edge of the map
    plt.xticks([-180, -90, 0, 90, 180])
    plt.yticks([-90, -45, 0, 45, 90])

    plt.colorbar()
    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    plt.title(f'{key} skymap')
    plt.legend(loc=1)
    if filename:
        plt.savefig(filename)
        plt.close()
        logger.info(f'Plot {key} skymap saved to {filename}')
    else:
        return plt
