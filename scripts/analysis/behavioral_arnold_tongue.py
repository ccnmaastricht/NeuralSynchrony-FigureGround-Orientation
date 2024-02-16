"""
This script computes the behavioral Arnold tongue of each subject as well as the average behavioral Arnold tongue.
Results are saved in results/empirical/ and correspond to section X of the paper.
"""
import os
import tomllib
import numpy as np
from scipy.optimize import curve_fit

from src.anl_utils import load_data, get_session_data, get_subject_data, psychometric_function

def load_configuration():
    """
    Load parameters for the in silico experiment.

    Returns
    -------
    experiment_parameters : dict
        The experiment parameters.
    """
    with open('config/analysis/experiment_extended.toml', 'rb') as f:
        experiment_parameters = tomllib.load(f)

    return experiment_parameters

def get_unique_counts(data):
    """
    Get the unique counts of the data.

    Parameters
    ----------
    data : pandas.DataFrame
            The data.

    Returns
    -------
    num_subjects : int
        The number of subjects.
    num_grid_coarseness : int
        The number of grid coarseness values.
    num_contrast_heterogeneity : int
        The number of contrast heterogeneity values.
    """
    num_subjects = data['SubjectID'].nunique()
    num_grid_coarseness = data['GridCoarseness'].nunique()
    num_contrast_heterogeneity = data['ContrastHeterogeneity'].nunique()
    return num_subjects, num_grid_coarseness, num_contrast_heterogeneity

def create_predictors(bounds_grid_coarseness, num_grid_coarseness, bounds_contrast_heterogeneity, num_contrast_heterogeneity):
    """
    Create the predictors for the Arnold tongue.

    Parameters
    ----------
    bounds_grid_coarseness : tuple
        The bounds of the grid coarseness.
    num_grid_coarseness : int
        The number of grid coarseness values.
    bounds_contrast_heterogeneity : tuple
        The bounds of the contrast heterogeneity.
    num_contrast_heterogeneity : int
        The number of contrast heterogeneity values.

    Returns
    -------
    predictors : array_like
        The predictors for the Arnold tongue.
    """
    grid_coarseness = np.linspace(bounds_grid_coarseness[0], bounds_grid_coarseness[1], num_grid_coarseness)
    contrast_heterogeneity = np.linspace(bounds_contrast_heterogeneity[0], bounds_contrast_heterogeneity[1], num_contrast_heterogeneity)
    contrast_heterogeneity, grid_coarseness = np.meshgrid(contrast_heterogeneity, grid_coarseness)
    predictors = np.vstack((grid_coarseness.flatten(), contrast_heterogeneity.flatten()))
    predictors = np.vstack((predictors, np.ones(len(predictors[0]))))

    return predictors

BASE_PATH = 'results/analysis/'      

if __name__ == '__main__':
    # Load experiment parameters
    experiment_parameters = load_configuration()
    bounds_grid_coarseness = (experiment_parameters['min_grid_coarseness'], experiment_parameters['max_grid_coarseness'])
    bounds_contrast_heterogeneity = (experiment_parameters['min_contrast_heterogeneity'], experiment_parameters['max_contrast_heterogeneity'])

    # Load data
    data_path = 'data/main.csv'
    try:
        data = load_data(data_path)
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        exit(1)

    for session in range(1, 10):

        # Get session data
        session_data = get_session_data(data, session)

        # Get unique counts
        num_subjects, num_grid_coarseness, num_contrast_heterogeneity = get_unique_counts(session_data)

        # Initialize array for individual Arnold tongues
        individual_arnold_tongues = np.zeros((num_subjects, num_grid_coarseness, num_contrast_heterogeneity))
        conditions = session_data['Condition'].unique()

        # Loop over subjects and conditions to fill individual Arnold tongues
        for subject in range(num_subjects):
            subject_data = get_subject_data(session_data, subject + 1)
            accuracy = np.zeros(len(conditions))
            for condition in conditions:
                condition_data = subject_data[subject_data['Condition'] == condition]
                accuracy[condition - 1] = condition_data['Correct'].mean()

            individual_arnold_tongues[subject] = accuracy.reshape((num_grid_coarseness, num_contrast_heterogeneity))

        # Compute average Arnold tongue
        average_arnold_tongue = individual_arnold_tongues.mean(axis=0)

        # Get bounds and create predictors
        predictors = create_predictors(bounds_grid_coarseness, num_grid_coarseness,
                                    bounds_contrast_heterogeneity, num_contrast_heterogeneity)

        # Initial guesses for parameters
        initial_params = [0.0, 0.0, 0.0]

        # Fit psychometric function to data
        popt, _ = curve_fit(psychometric_function, predictors, average_arnold_tongue.flatten(), p0=initial_params)

        # Create predictors for continuous Arnold tongue
        predictors = create_predictors(bounds_grid_coarseness, experiment_parameters['num_grid_coarseness'],
                                    bounds_contrast_heterogeneity, experiment_parameters['num_contrast_heterogeneity'])

        # Compute continuous Arnold tongue
        size = (experiment_parameters['num_grid_coarseness'],
                experiment_parameters['num_contrast_heterogeneity'])
        continuous_arnold_tongue = psychometric_function(predictors, *popt)

        continuous_arnold_tongue = continuous_arnold_tongue.reshape(size)

        # Save results
        directory = f'{BASE_PATH}session_{session}/'
        os.makedirs(directory, exist_ok=True)
        
        results_to_save = {
            'optimal_psychometric_parameters': popt,
            'average_bat': average_arnold_tongue,
            'continuous_bat': continuous_arnold_tongue,
            'individual_bats': individual_arnold_tongues
        }

        for filename, result in results_to_save.items():
            np.save(directory + filename + '.npy', result)