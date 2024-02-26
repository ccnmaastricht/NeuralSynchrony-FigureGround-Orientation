"""
This script compute fold-specific paramters of a sychometric function linking model synchrony to probability of a correct response of subjects in the first session of the experiment.
Results are saved in results/arnold_tongue.npy and correspond to section X of the paper.
"""

import os
import tomllib
import numpy as np
from scipy.optimize import curve_fit

from src.anl_utils import psychometric_function
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

   

if __name__ == '__main__':

    # load the simulated Arnold tongues of session 1
    simulated_arnold_tongue = np.load('results/simulation/baseline_arnold_tongue.npy').mean(axis=0)
    sat_rows, sat_columns = simulated_arnold_tongue.shape

    # load behavioral Arnold tongues of session 1
    individual_arnold_tongues = np.load('results/analysis/session_1/individual_bats.npy')
    num_subjects, bat_rows, bat_columns = individual_arnold_tongues.shape

    total_conditions = bat_rows * bat_columns
    
    row_ratio = sat_rows // bat_rows
    col_ratio = sat_columns // bat_columns

    simulated_arnold_tongue = simulated_arnold_tongue[::row_ratio, ::col_ratio].flatten()
    predictors = np.ones((2, total_conditions))
    predictors[0] = simulated_arnold_tongue

    optimal_psychometric_crossval = np.zeros((num_subjects, 2))

    for subject in range(num_subjects):
        # remove subject from arnold tongue but do not overwrite
        fold_arnold_tongues = np.delete(individual_arnold_tongues, subject, axis=0)
        
        # Compute average Arnold tongue
        average_arnold_tongue = fold_arnold_tongues.mean(axis=0)

        # Initial guesses for parameters
        initial_params = np.zeros(2)

        # Fit psychometric function to data
        popt, _ = curve_fit(psychometric_function, predictors, average_arnold_tongue.flatten(), p0=initial_params)
        optimal_psychometric_crossval[subject] = popt
        print(f'Optimal parameters for subject {subject}: {popt}')
        
    np.save('results/analysis/session_1/optimal_psychometric_crossval.npy', optimal_psychometric_crossval)









