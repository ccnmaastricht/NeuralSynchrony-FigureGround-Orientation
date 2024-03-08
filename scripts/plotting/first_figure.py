"""
This script creates all panels of the first figure of the paper. 
"""

import os
import tomllib

import numpy as np
from src.plot_utils import colored_heatmap, convert_size

BASE_PATH = 'figures/first_figure'

os.makedirs(BASE_PATH, exist_ok=True)

with open('config/plotting/first_figure.toml', 'rb') as f:
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

colored_heatmap(fitted_arnold_tongue,
                figsize=figsize,
                labels=figure_parameters['panels'][1]['labels'],
                fontsizes=figure_parameters['general']['fontsizes'],
                ticks=figure_parameters['panels'][1]['ticks'],
                bounds=figure_parameters['panels'][1]['bounds'],
                colormap=cmap,
                filename=filename)

# Panel C - parameter space
filename = os.path.join(BASE_PATH, 'panel_c')
exploration_data = np.load(
    'results/simulation/parameter_space_exploration.npz')
jaccard_fits = exploration_data['jaccard_fits']
jaccard_fits = np.flipud(jaccard_fits)

colored_heatmap(jaccard_fits,
                figsize=figsize,
                labels=figure_parameters['panels'][2]['labels'],
                fontsizes=figure_parameters['general']['fontsizes'],
                ticks=figure_parameters['panels'][2]['ticks'],
                bounds=figure_parameters['panels'][2]['bounds'],
                colormap=cmap,
                filename=filename)

# Panel D - simulated Arnold tongue
filename = os.path.join(BASE_PATH, 'panel_d')
simulated_arnold_tongue = np.load(
    'results/simulation/highres_arnold_tongues.npy')[0].mean(axis=0)

colored_heatmap(simulated_arnold_tongue,
                figsize=figsize,
                labels=figure_parameters['panels'][3]['labels'],
                fontsizes=figure_parameters['general']['fontsizes'],
                ticks=figure_parameters['panels'][3]['ticks'],
                bounds=figure_parameters['panels'][3]['bounds'],
                colormap=cmap,
                filename=filename)
