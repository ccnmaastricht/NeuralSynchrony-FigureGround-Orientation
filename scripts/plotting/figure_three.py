"""
This script creates all panels of the third figure of the paper. 
"""

import os
import tomllib
import matplotlib.pyplot as plt

import numpy as np
from src.plot_utils import colored_heatmap, convert_size

BASE_PATH = 'results/figures/figure_three'

os.makedirs(BASE_PATH, exist_ok=True)

with open('config/plotting/figure_three.toml', 'rb') as f:
    figure_parameters = tomllib.load(f)

figsize = convert_size(*figure_parameters['general']['figure_size'])

exploration_data = np.load(
    'results/simulation/parameter_space_exploration.npz')

# Panel A - parameter space exploration using Pearson correlation
filename = os.path.join(BASE_PATH, 'panel_a')
correlation_fits = exploration_data['correlation_fits']
correlation_fits = np.flipud(correlation_fits)

heatmap = colored_heatmap(correlation_fits,
                          figsize=figsize,
                          labels=figure_parameters['panels'][0]['labels'],
                          fontsizes=figure_parameters['general']['fontsizes'],
                          ticks=figure_parameters['general']['ticks'],
                          bounds=figure_parameters['general']['bounds'],
                          colormap=figure_parameters['general']['colormap'])

ax = heatmap.gca()
ax.plot(figure_parameters['general']['vertical_line'][0],
        figure_parameters['general']['vertical_line'][1], 'k--')
ax.plot(figure_parameters['general']['horizontal_line'][0],
        figure_parameters['general']['horizontal_line'][1], 'k--')
ax.plot(figure_parameters['general']['marker'][0],
        figure_parameters['general']['marker'][1], 'ko')

ax.set_xlim(figure_parameters['general']['ticks'][0][0],
            figure_parameters['general']['ticks'][0][-1])
ax.set_ylim(figure_parameters['general']['ticks'][1][-1],
            figure_parameters['general']['ticks'][1][0])

heatmap.tight_layout()
filename = f'{filename}.svg'
heatmap.savefig(filename)
plt.close(heatmap)

# Panel B - parameter space exploration using Jaccard index
filename = os.path.join(BASE_PATH, 'panel_b')
jaccard_fits = exploration_data['jaccard_fits']
jaccard_fits = np.flipud(jaccard_fits)

heatmap = colored_heatmap(jaccard_fits,
                          figsize=figsize,
                          labels=figure_parameters['panels'][1]['labels'],
                          fontsizes=figure_parameters['general']['fontsizes'],
                          ticks=figure_parameters['general']['ticks'],
                          bounds=figure_parameters['general']['bounds'],
                          colormap=figure_parameters['general']['colormap'])

ax = heatmap.gca()
ax.plot(figure_parameters['general']['vertical_line'][0],
        figure_parameters['general']['vertical_line'][1], 'k--')
ax.plot(figure_parameters['general']['horizontal_line'][0],
        figure_parameters['general']['horizontal_line'][1], 'k--')
ax.plot(figure_parameters['general']['marker'][0],
        figure_parameters['general']['marker'][1], 'ko')

ax.set_xlim(figure_parameters['general']['ticks'][0][0],
            figure_parameters['general']['ticks'][0][-1])
ax.set_ylim(figure_parameters['general']['ticks'][1][-1],
            figure_parameters['general']['ticks'][1][0])

heatmap.tight_layout()
filename = f'{filename}.svg'
heatmap.savefig(filename)
plt.close(heatmap)
