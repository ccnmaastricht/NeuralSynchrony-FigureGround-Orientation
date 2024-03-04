"""
This script creates the Arnold Tongue figure...
"""

import os

import numpy as np
from src.plot_utils import plot_arnold_tongue

# Define file paths
data_file_path = '../data/simulation_results/arnold_tongue.npy'
figure_file_path = '../figures/arnold_tongue.png'

# Ensure necessary directories exist
os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)

# Load Arnold Tongue results
arnold_tongue = np.load(data_file_path)

# Define plot parameters
xticks = np.round(np.linspace(0.01, 1, 5), 1)
yticks = np.round(np.linspace(1, 1.5, 5), 1)
ticks = (xticks, yticks)

labels = ('Arnold Tongue', 'Contrast heterogeneity', 'Grid coarseness',
          'Synchronization')
fontsizes = (6, 5, 5, 5)  # title, label, tick, cbar label sizes

# Specify figure size in inches
figsize = (88.9 / 25.4, 88.9 / 25.4)  # convert from millimeters to inches

# Plot the results
plot_arnold_tongue(arnold_tongue.mean(axis=0),
                   figsize,
                   labels,
                   fontsizes,
                   ticks,
                   show=False,
                   save=True,
                   filename=figure_file_path,
                   fraction=0.0454)
