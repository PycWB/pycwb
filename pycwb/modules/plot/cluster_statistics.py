from gwpy.spectrogram import Spectrogram
import matplotlib.style as mplstyle
import logging

logger = logging.getLogger(__name__)


def plot_statistics(cluster, key='likelihood', filename=None):
    """Plot the statistics of the event

    :param cluster: cluster
    :type cluster: Cluster
    :param key: key of the statistics, defaults to 'likelihood'
    :type key: str, optional
    :param filename: path to save the plot, defaults to None
    :type filename: str, optional
    """
    # plot the statistics
    mplstyle.use('fast')

    merged_map, start, dt, df = cluster.get_sparse_map(key)

    plt = Spectrogram(merged_map, t0=start, dt=dt, f0=0, df=df).plot()
    plt.colorbar()
    # save to png
    if filename is not None:
        plt.savefig(filename)
        logger.info(f'Plot {key} saved to {filename}')
    plt.close()
