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

figsize = convert_size(*figure_parameters['figure_size'])
filename = os.path.join(BASE_PATH, 'figure_three')
exploration_data = np.load(
    'results/simulation/parameter_space_exploration.npz')
jaccard_fits = exploration_data['jaccard_fits']
jaccard_fits = np.flipud(jaccard_fits)

heatmap = colored_heatmap(jaccard_fits,
                          figsize=figsize,
                          labels=figure_parameters['labels'],
                          fontsizes=figure_parameters['fontsizes'],
                          ticks=figure_parameters['ticks'],
                          bounds=figure_parameters['bounds'],
                          colormap=figure_parameters['colormap'])

ax = heatmap.gca()
ax.plot(figure_parameters['vertical_line'][0],
        figure_parameters['vertical_line'][1], 'k--')
ax.plot(figure_parameters['horizontal_line'][0],
        figure_parameters['horizontal_line'][1], 'k--')
ax.plot(figure_parameters['marker'][0], figure_parameters['marker'][1], 'ko')

ax.set_xlim(figure_parameters['ticks'][0][0],
            figure_parameters['ticks'][0][-1])
ax.set_ylim(figure_parameters['ticks'][1][-1],
            figure_parameters['ticks'][1][0])

heatmap.tight_layout()
filename = f'{filename}.svg'
heatmap.savefig(filename)
plt.close(heatmap)
