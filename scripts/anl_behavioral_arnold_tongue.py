"""
This script computes the behavioral Arnold tongue of each subject as well as the average behavioral Arnold tongue.
Results are saved in ../data/empirical_results/behavioral_arnold_tongue.npy and correspond to section X of the paper.
"""
import numpy as np
from scipy.optimize import curve_fit

from src.anl_utils import load_data, get_session_data, get_subject_data, psychometric_function

def get_bounds(data, variable):
    """
    Get the bounds of a variable.

    Parameters
    ----------
    data : pandas.DataFrame
        The behavioral data.
    variable : str
        The variable.

    Returns
    -------
    bounds : tuple
        The bounds of the variable.
    """
    bounds = (data[variable].min(), data[variable].max())
    return bounds

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


if __name__ == '__main__':
    # Load data
    data_path = 'data/empirical/main.csv'
    try:
        data = load_data(data_path)
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        exit(1)

    session_data = get_session_data(data, 1)

    # Get unique counts
    num_subjects = session_data['SubjectID'].nunique()
    num_grid_coarseness = session_data['GridCoarseness'].nunique()
    num_contrast_heterogeneity = session_data['ContrastHeterogeneity'].nunique()

    # Initialize array for individual Arnold tongues
    individual_arnold_tongues = np.zeros((num_subjects, num_grid_coarseness, num_contrast_heterogeneity))
    conditions = session_data['Condition'].unique()

    # Loop over subjects and conditions to fill individual Arnold tongues
    for subject in range(num_subjects):
        subject_data = get_subject_data(data, subject + 1)
        accuracy = np.zeros(len(conditions))
        for condition in conditions:
            condition_data = subject_data[subject_data['Condition'] == condition]
            accuracy[condition - 1] = condition_data['Correct'].mean()

        individual_arnold_tongues[subject] = accuracy.reshape((num_grid_coarseness, num_contrast_heterogeneity))

    # Compute average Arnold tongue
    average_arnold_tongue = individual_arnold_tongues.mean(axis=0)

    # Get bounds and create predictors
    bounds_grid_coarseness = get_bounds(session_data, 'GridCoarseness')
    bounds_contrast_heterogeneity = get_bounds(session_data, 'ContrastHeterogeneity')
    predictors = create_predictors(bounds_grid_coarseness, num_grid_coarseness,
                                   bounds_contrast_heterogeneity, num_contrast_heterogeneity)

    # Initial guesses for parameters
    initial_params = [0.0, 0.0, 0.0]

    # Fit psychometric function to data
    popt, _ = curve_fit(psychometric_function, predictors, average_arnold_tongue.flatten(), p0=initial_params)

    # Create predictors for continuous Arnold tongue
    predictors = create_predictors(bounds_grid_coarseness, 30,
                                   bounds_contrast_heterogeneity, 30)

    # Compute continuous Arnold tongue
    continuous_arnold_tongue = psychometric_function(predictors, *popt)
    continuous_arnold_tongue = continuous_arnold_tongue.reshape((30, 30))

    # Save results
    np.save('data/results/empirical/optimal_psychometric_parameters.npy', popt)
    np.save('data/results/empirical/average_bat.npy', average_arnold_tongue)
    np.save('data/results/empirical/continuous_bat.npy', continuous_arnold_tongue)
    np.save('data/results/empirical/individual_bats.npy', individual_arnold_tongues)