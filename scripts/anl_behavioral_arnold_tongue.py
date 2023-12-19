"""
This script computes the behavioral Arnold tongue of each subject as well as the average behavioral Arnold tongue.
Results are saved in ../data/empirical_results/behavioral_arnold_tongue.npy and correspond to section X of the paper.
"""
import numpy as np
from scipy.optimize import minimize

from src.anl_utils import load_data, get_session_data, get_subject_data, psychometric_function

def optimize_psychometric(data):
    """
    Optimize the parameters of the psychometric function.

    Parameters
    ----------
    data : pandas.DataFrame
        The behavioral data.

    Returns
    -------
    params : array_like
        The optimized parameters.
    """
    x = data['Contrast'].values
    y = data['Correct'].values

    params = minimize(psychometric_function, x0=[0, 0, 0, 0], args=(x, y), method='Nelder-Mead').x

    return params


if __name__ == '__main__':

    data_path = 'data/empirical/main.csv'

    data = load_data(data_path)
    session_data = get_session_data(data, 1)

    num_subjects = session_data['SubjectID'].nunique()
    num_grid_coarseness = session_data['GridCoarseness'].nunique()
    num_contrast_heterogeneity = session_data['ContrastHeterogeneity'].nunique()

    print(f'Number of subjects: {num_subjects}')
    print(f'Number of grid coarseness: {num_grid_coarseness}')
    print(f'Number of contrast heterogeneity: {num_contrast_heterogeneity}')

    individual_arnold_tongues = np.zeros((num_subjects, num_grid_coarseness, num_contrast_heterogeneity))
    conditions = session_data['Condition'].unique()
    for subject in range(num_subjects):
        subject_data = get_subject_data(data, subject + 1)
        accuracy = np.zeros(len(conditions))
        for condition in conditions:
            print(f'Condition: {condition}')
            condition_data = subject_data[subject_data['Condition'] == condition]
            accuracy[condition - 1] = condition_data['Correct'].mean()

        individual_arnold_tongues[subject] = accuracy.reshape((num_grid_coarseness, num_contrast_heterogeneity))

    average_arnold_tongue = individual_arnold_tongues.mean(axis=0)

    np.save('data/results/empirical/average_bat.npy', average_arnold_tongue)
    np.save('data/results/empirical/individual_bats.npy', individual_arnold_tongues)