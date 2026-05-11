def plot_1d_histogram(x: np.ndarray | list, colors: str | list, label: str = None, ax = None, bins=20, xlim: tuple = None, x_edges=None):
    if not isinstance(x, list):
        x = [x]
    if not isinstance(colors, list):
        colors = [colors] * len(x)
    
    x_tot = np.concatenate(x)
    x_min, x_max = x_tot.min() - x_tot.std(), x_tot.max() + x_tot.std()
    x_edges = np.linspace(x_min, x_max, bins + 1) if x_edges is None else x_edges

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    for data, color in zip(x, colors):
        x_vals = data
        ax.hist(x_vals, bins=x_edges, color=color, alpha=0.5, label=label if label else None)

    ax.set_xlim(*xlim) if xlim else ax.set_xlim(x_min, x_max)

    return ax

def plot_2d_histogram(x: np.ndarray, y: np.ndarray, xbins: int = 100, ybins: int = 100, 
                      xscale: str = 'lin', yscale: str = 'lin', xlim: tuple = None, ylim: tuple = None,
                      cmap: str = 'viridis', xlabel: str = '', ylabel: str = ''):
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_bins = np.logspace(np.log10(x_min), np.log10(x_max), xbins) if xscale == 'log' else np.linspace(x_min, x_max, xbins)
    y_bins = np.logspace(np.log10(y_min), np.log10(y_max), ybins) if yscale == 'log' else np.linspace(y_min, y_max, ybins)
    
    h, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])
    h_masked = np.ma.masked_where(h == 0, h)

    fig, ax = plt.subplots(figsize=(8, 5))
    mesh = ax.pcolormesh(xedges, yedges, h_masked.T, norm=LogNorm(), cmap=cmap, shading='auto')
    plt.colorbar(mesh, ax=ax)

    if xscale=='log': plt.xscale('log')
    if yscale=='log': plt.yscale('log')

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlim(*xlim) if xlim else ax.set_xlim(x_min * 0.9, x_max * 1.1)
    ax.set_ylim(*ylim) if ylim else ax.set_ylim(y_min * 0.9, y_max * 1.1)
    ax.grid(ls='dotted')

    return fig, ax

