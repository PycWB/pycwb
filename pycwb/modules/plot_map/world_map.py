from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp


def plot_world_map(phi, theta, filename=None):
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


def plot_contour(network, filename=None):
    L = network.net.nLikelihood.size()
    nLikelihood = [network.net.nLikelihood.get(l) for l in range(L)]

    # create a 20 x 10 map
    theta = np.linspace(0, np.pi, 10)
    phi = np.linspace(0, 2 * np.pi, 20)

    # create mesh
    theta, phi = np.meshgrid(theta, phi)
    flatten_theta = theta.flatten()
    flatten_phi = phi.flatten()

    # convert to HEALPix indices
    healpix_indices = [hp.ang2pix(2 ** 7, flatten_theta[i], flatten_phi[i]) for i in range(len(flatten_theta))]
    values = [nLikelihood[i] for i in healpix_indices]

    # reshape to 2D map
    values = np.reshape(values, theta.shape)

    # contour plot
    import matplotlib.pyplot as plt
    plt.contourf(phi, theta, values)

    if filename:
        plt.savefig(filename)
        plt.clf()
    else:
        return plt
