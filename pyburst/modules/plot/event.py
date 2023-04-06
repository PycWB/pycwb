import matplotlib.pyplot as plt
from .spectrogram import plot_spectrogram


def plot_event_on_spectrogram(strain, events, filename=None):
    plot = plot_spectrogram(strain, figsize=(24, 6), gwpy_plot=True)

    # plot boxes on the plot
    i = 0
    boxes = [[e.start[i], e.stop[i], e.low[i], e.high[i]] for e in events if len(e.start) > 0]

    for box in boxes:
        ax = plot.gca()
        ax.add_patch(plt.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], linewidth=0.5, fill=False,
                                   color='red'))

    # save to png
    if filename is not None:
        plot.savefig(filename)

    return plot