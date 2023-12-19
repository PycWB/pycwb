import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import logging
import matplotlib
logger = logging.getLogger(__name__)


def plot_reconstructed_waveforms(outputDir, reconstructed_waves, xlim):
    logger.info('Plotting reconstructed waveforms')
    mplstyle.use('fast')
    print(matplotlib.get_backend())

    for j, reconstructed_wave in enumerate(reconstructed_waves):
        plt.plot(reconstructed_wave.sample_times, reconstructed_wave.data)
        plt.xlim(xlim)
        plt.savefig(f'{outputDir}/reconstructed_wave_ifo_{j+1}.png')
        plt.close()
        # log the path to the reconstructed waveforms
        logger.info(f'Plot saved to {outputDir}/reconstructed_wave_ifo_{j+1}.png')
