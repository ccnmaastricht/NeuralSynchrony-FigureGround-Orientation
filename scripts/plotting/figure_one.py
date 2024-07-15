"""
This script creates stimulus configurations for the first figure of the paper.
"""

import os
import tomllib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.stimulus_generator import StimulusGenerator

# Set Seaborn style
sns.set(style='white')


def load_config(config_directories, config_files):
    config = {}
    for directory, config_file in zip(config_directories, config_files):
        file = os.path.join(directory, f'{config_file}.toml')
        with open(file, 'rb') as f:
            config[config_file.split('_')[0]] = tomllib.load(f)
    return config


def create_stimulus_images(config, figure_directory):
    # Update stimulus config with figure specific parameters
    config['stimulus']['stimulus_resolution'] = config['figure']['stimulus'][
        'stimulus_resolution']
    config['stimulus']['stimulus_num_pixels'] = config['figure']['stimulus'][
        'stimulus_num_pixels']
    config['stimulus']['annulus_resolution'] = config['figure']['stimulus'][
        'annulus_resolution']

    stimulus_generator = StimulusGenerator(config['stimulus'])

    scaling_factors = np.linspace(config['experiment']['min_grid_coarseness'],
                                  config['experiment']['max_grid_coarseness'],
                                  config['experiment']['num_grid_coarseness'])
    contrast_ranges = np.linspace(
        config['experiment']['min_contrast_heterogeneity'],
        config['experiment']['max_contrast_heterogeneity'],
        config['experiment']['num_contrast_heterogeneity'])

    os.makedirs(figure_directory, exist_ok=True)

    figure_size = config['figure']['general']['figure_size']

    for scaling_factor in scaling_factors:
        for contrast_range in contrast_ranges:
            stimulus = stimulus_generator.generate(
                scaling_factor,
                contrast_range,
                mean_contrast=config['experiment']['mean_contrast'])

            fig, ax = plt.subplots(figsize=figure_size)
            ax.imshow(stimulus, cmap='gray')
            ax.axis('off')

            file_name = f'scaling_{scaling_factor:.2f}_contrast_{contrast_range:.2f}.svg'
            file_path = os.path.join(figure_directory, file_name)
            plt.savefig(file_path,
                        format='svg',
                        bbox_inches='tight',
                        pad_inches=0)
            plt.close(fig)


if __name__ == "__main__":
    config_directories = [
        'config/analysis', 'config/simulation', 'config/plotting'
    ]
    config_files = ['experiment_actual', 'stimulus', 'figure_one']
    figure_directory = os.path.join('results', 'figures', 'figure_one',
                                    'panel_d')

    config = load_config(config_directories, config_files)
    create_stimulus_images(config, figure_directory)
