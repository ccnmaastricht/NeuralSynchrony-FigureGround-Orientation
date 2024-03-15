"""
This script creates all panels of the fourth figure of the paper. 
"""

import os
import tomllib
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from src.anl_utils import min_max_normalize, weighted_jaccard
from src.plot_utils import fit_barplot, convert_size

BASE_PATH = 'results/figures/figure_four'

os.makedirs(BASE_PATH, exist_ok=True)


def compute_noise_ceiling(individual_bats):
    """
    Compute the noise ceiling within a specific session using the leave-one-out CV approach also used for the model.

    Parameters
    ----------
    individual_bats : array_like
        The individual behavioural Arnold tongues.

    Returns
    -------
    float
        The noise ceiling using the correlation coefficient.
    float
        The noise ceiling using the weighted Jaccard similarity.
    """

    num_subjects = individual_bats.shape[0]
    corrcoefs = np.zeros(num_subjects)
    jaccards = np.zeros(num_subjects)

    for i in range(num_subjects):
        left_out_bat = individual_bats[i].flatten()
        left_out_bat = min_max_normalize(left_out_bat)
        average_bat = np.delete(individual_bats, i,
                                axis=0).mean(axis=0).flatten()
        average_bat = min_max_normalize(average_bat)

        corrcoefs[i] = np.corrcoef(left_out_bat, average_bat)[0, 1]
        jaccards[i] = weighted_jaccard(left_out_bat, average_bat)

    noise_ceiling_corrcoef = np.percentile(corrcoefs,
                                           25), np.percentile(corrcoefs, 75)
    noise_ceiling_jaccard = np.percentile(jaccards,
                                          25), np.percentile(jaccards, 75)

    return noise_ceiling_corrcoef, noise_ceiling_jaccard


def compute_corr_moments(correlation_fits):
    """
    Compute the mean and standard error on the correlation scale.

    Parameters
    ----------
    correlation_fits : array_like
        The correlation coefficients.

    Returns
    -------
    array_like
        The mean correlation fit.
    array_like
        The standard error on the correlation fit.
    """
    # Fisher-z transform the correlation coefficients
    z_corr_fits = np.arctanh(correlation_fits)

    # Compute the mean and standard error on the Fisher-z scale
    mean_z_corr_fit = np.mean(z_corr_fits, axis=0)
    sem_z_corr_fit = np.std(z_corr_fits, axis=0) / np.sqrt(
        z_corr_fits.shape[0])

    # Inverse Fisher-z transform the mean and confidence interval to get them back on the correlation scale
    mean_corr_fit = np.tanh(mean_z_corr_fit)
    sem_corr_fit = np.tanh(sem_z_corr_fit)

    return mean_corr_fit, sem_corr_fit


if __name__ == '__main__':

    # Load the figure parameters
    with open('config/plotting/figure_four.toml', 'rb') as f:
        figure_parameters = tomllib.load(f)

    # Convert the figure size to inches
    figsize = convert_size(*figure_parameters['general']['figure_size'])

    # Load the model fits
    with np.load('results/simulation/learning_simulation.npz') as model_fits:
        correlation_fits = model_fits['correlation_fits']
        jaccard_fits = model_fits['jaccard_fits']

    noise_ceiling_corrcoef = np.zeros((8, 2))
    noise_ceiling_jaccard = np.zeros((8, 2))

    for session in figure_parameters['general']['sessions']:
        individual_bats = np.load(
            f'results/empirical/session_{session}/individual_bats.npy')

        noise_ceiling_corrcoef[session - 1], noise_ceiling_jaccard[
            session - 1] = compute_noise_ceiling(individual_bats)

    # Panel A - correlation fits per session
    filename = os.path.join(BASE_PATH, 'panel_a')

    # Inverse Fisher-z transform the mean and confidence interval to get them back on the correlation scale
    mean_corr_fit, sem_corr_fit = compute_corr_moments(correlation_fits)

    # Create a bar plot
    fit_barplot(mean_corr_fit,
                sem_corr_fit,
                figure_parameters['general']['sessions'],
                noise_ceiling_corrcoef,
                figsize=figsize,
                labels=figure_parameters['panels'][0]['labels'],
                fontsizes=figure_parameters['general']['fontsizes'],
                capsize=figure_parameters['general']['capsize'],
                face_color=figure_parameters['general']['face_color'],
                filename=filename,
                dpi=300,
                filetype='svg')

    # Panel B - Jaccard fits per session
    filename = os.path.join(BASE_PATH, 'panel_b')

    # Compute the mean and standard error on the Jaccard scale
    mean_jaccard_fit = np.mean(jaccard_fits, axis=0)
    sem_jaccard_fit = np.std(jaccard_fits, axis=0) / np.sqrt(
        jaccard_fits.shape[0])

    # Create a bar plot
    fit_barplot(mean_jaccard_fit,
                sem_jaccard_fit,
                figure_parameters['general']['sessions'],
                noise_ceiling_jaccard,
                figsize=figsize,
                labels=figure_parameters['panels'][1]['labels'],
                fontsizes=figure_parameters['general']['fontsizes'],
                capsize=figure_parameters['general']['capsize'],
                face_color=figure_parameters['general']['face_color'],
                filename=filename,
                dpi=300,
                filetype='svg')
