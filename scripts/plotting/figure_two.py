"""
This script creates all panels of the second figure of the paper. 
"""

import os
import tomllib
import matplotlib.pyplot as plt

import numpy as np
from src.plot_utils import colored_heatmap, convert_size

BASE_PATH = 'results/figures/figure_two'

os.makedirs(BASE_PATH, exist_ok=True)

with open('config/plotting/figure_two.toml', 'rb') as f:
    figure_parameters = tomllib.load(f)

figsize = convert_size(*figure_parameters['general']['figure_size'])
cmap = figure_parameters['general']['colormap']

# Panel A - behavioural Arnold tongue (raw)
filename = os.path.join(BASE_PATH, 'panel_a')
behavioural_arnold_tongue = np.load(
    'results/empirical/session_1/average_bat.npy')

colored_heatmap(behavioural_arnold_tongue,
                figsize=figsize,
                labels=figure_parameters['panels'][0]['labels'],
                fontsizes=figure_parameters['general']['fontsizes'],
                ticks=figure_parameters['panels'][0]['ticks'],
                bounds=figure_parameters['panels'][0]['bounds'],
                colormap=cmap,
                filename=filename)

# Panel B - behavioural Arnold tongue (fitted)
filename = os.path.join(BASE_PATH, 'panel_b')
fitted_arnold_tongue = np.load(
    'results/empirical/session_1/continuous_bat.npy')
psychometric_parameters = np.load(
    'results/empirical/session_1/optimal_psychometric_parameters.npy')

heatmap = colored_heatmap(fitted_arnold_tongue,
                          figsize=figsize,
                          labels=figure_parameters['panels'][1]['labels'],
                          fontsizes=figure_parameters['general']['fontsizes'],
                          ticks=figure_parameters['panels'][1]['ticks'],
                          bounds=figure_parameters['panels'][1]['bounds'],
                          colormap=cmap)

# Add threshold line (75% correct)
contrast_heterogeneity = np.linspace(
    figure_parameters['panels'][1]['min_contrast_heterogeneity'],
    figure_parameters['panels'][1]['max_contrast_heterogeneity'],
    figure_parameters['panels'][1]['num_contrast_heterogeneity'])
grid_coarseness = -(psychometric_parameters[1] * contrast_heterogeneity +
                    psychometric_parameters[2]) / psychometric_parameters[0]

ax = heatmap.gca()
ax.plot(contrast_heterogeneity, grid_coarseness, 'k--')
xticks, yticks = figure_parameters['panels'][1]['ticks']
ax.set_xlim(xticks[0], xticks[-1])
ax.set_ylim(yticks[-1], yticks[0])

heatmap.tight_layout()
filename = f'{filename}.svg'
heatmap.savefig(filename)
plt.close(heatmap)

# Panel C - simulated Arnold tongue (low resolution)
filename = os.path.join(BASE_PATH, 'panel_c')
simulated_arnold_tongue = np.load(
    'results/simulation/highres_arnold_tongues.npy')[0].mean(axis=0)

colored_heatmap(simulated_arnold_tongue[::6, ::6],
                figsize=figsize,
                labels=figure_parameters['panels'][2]['labels'],
                fontsizes=figure_parameters['general']['fontsizes'],
                ticks=figure_parameters['panels'][2]['ticks'],
                bounds=figure_parameters['panels'][2]['bounds'],
                colormap=cmap,
                filename=filename)

# Panel D - simulated Arnold tongue (high resolution)
filename = os.path.join(BASE_PATH, 'panel_d')

colored_heatmap(simulated_arnold_tongue,
                figsize=figsize,
                labels=figure_parameters['panels'][3]['labels'],
                fontsizes=figure_parameters['general']['fontsizes'],
                ticks=figure_parameters['panels'][3]['ticks'],
                bounds=figure_parameters['panels'][3]['bounds'],
                colormap=cmap,
                filename=filename)
