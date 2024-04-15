import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def hrss50_bar_plot(data_sets: list[tuple[dict[str, pd.DataFrame], str]],
                    wf_selections=None, output_dir='.', filename='hrss50_comparison.png'):
    if wf_selections is None:
        wf_names_plot = list(data_sets[0][0].keys())
    else:
        wf_names_plot = wf_selections

    color = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
             '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']

    bar_width = 0.8 / len(data_sets)
    opacity = 0.8

    fig, ax = plt.subplots(figsize=(14, 3.5))
    len_data_sets = len(data_sets)
    for i, (data_set, label) in enumerate(data_sets):
        indexes = []
        hrss50s = []
        hrss50_errs = []
        for wf_name in data_set.keys():
            if wf_name not in wf_names_plot:
                continue
            indexes.append(wf_names_plot.index(wf_name))
            hrss50s.append(data_set[wf_name][0])
            hrss50_errs.append(data_set[wf_name][1])
        bars = ax.bar(np.array(indexes) + i * bar_width, hrss50s, bar_width, alpha=opacity, color=color[i],
                      yerr=hrss50_errs, label=label,
                      error_kw=dict(lw=1, capsize=1.5, capthick=1, alpha=0.7))

    index = np.arange(len(wf_names_plot))
    ax.set_yscale('log')
    ax.set_ylabel('hrss50')
    ax.set_xticks(index + bar_width * (len_data_sets - 1) / 2)
    ax.set_xticklabels(wf_names_plot, rotation=45, ha='right')  # Rotate x-labels 45 degrees
    ax.legend(ncol=3)

    plt.tight_layout()

    plt.savefig(output_dir + '/' + filename, bbox_inches='tight', transparent=True)

