import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import logging
import matplotlib
logger = logging.getLogger(__name__)


def plot_reconstructed_waveforms(outputDir, reconstructed_waves, xlim):
    mplstyle.use('fast')

    for j, reconstructed_wave in enumerate(reconstructed_waves):
        plt.plot(reconstructed_wave.sample_times, reconstructed_wave.data)
        plt.xlim(xlim)
        plt.savefig(f'{outputDir}/reconstructed_wave_ifo_{j+1}.png')
        plt.close()
        # log the path to the reconstructed waveforms
        logger.info(f'Plot reconstructed waveforms saved to {outputDir}/reconstructed_wave_ifo_{j+1}.png')
