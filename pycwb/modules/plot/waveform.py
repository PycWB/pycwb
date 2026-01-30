import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from pycbc.types import TimeSeries
from gwpy.plot.colors import GW_OBSERVATORY_COLORS
import logging
logger = logging.getLogger(__name__)

custom_rc = {    
    # Tick Lengths
    'xtick.major.size': 10,
    'xtick.minor.size': 5,
    'ytick.major.size': 10,
    'ytick.minor.size': 5,
    
    # Tick Direction
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    
    # Enable Minor Ticks by default
    'xtick.minor.visible': True,
    'ytick.minor.visible': True
}

def plot(wave: TimeSeries, ifo = None):
    # set offset for gps offset
    offset = round(float(wave.start_time), -3)

    with plt.rc_context(custom_rc):
        plt.plot(wave.sample_times - offset, wave.data, zorder=3, color=GW_OBSERVATORY_COLORS[ifo] if ifo else 'r', marker='.', markersize=0.6, linewidth=0.4)
        plt.xlabel(f'Time (sec): GPS OFFSET = {offset:.3f}', fontsize=10)
        plt.ylabel('magnitude', fontsize=10)
        plt.grid(linestyle = ':', zorder=1)
     
def plot_reconstructed_waveforms(outputDir, reconstructed_waves, xlim):
    mplstyle.use('fast')

    for j, reconstructed_wave in enumerate(reconstructed_waves):
        plt.plot(reconstructed_wave.sample_times, reconstructed_wave.data)
        plt.xlim(xlim)
        plt.savefig(f'{outputDir}/reconstructed_wave_ifo_{j+1}.png')
        plt.close()
        # log the path to the reconstructed waveforms
        logger.info(f'Plot reconstructed waveforms saved to {outputDir}/reconstructed_wave_ifo_{j+1}.png')
