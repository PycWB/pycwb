import matplotlib.pyplot as plt


def plot_reconstructed_waveforms(outputDir, reconstructed_waves, xlim):
    for j, reconstructed_wave in enumerate(reconstructed_waves):
        plt.plot(reconstructed_wave.sample_times, reconstructed_wave.data)
        plt.xlim(xlim)
        plt.savefig(f'{outputDir}/reconstructed_wave_ifo_{j+1}.png')
        plt.clf()