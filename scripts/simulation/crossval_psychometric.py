"""
This script compute fold-specific paramters of a sychometric function linking model synchrony to probability of a correct response of subjects in the first session of the experiment.
Results are saved in results/arnold_tongue.npy and correspond to section X of the paper.
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

   

if __name__ == '__main__':

    arnold_tongue = np.load('results/simulation/baseline_arnold_tongue.npy')
    behavioral_arnold_tongues = np.load('results/analysis/session_1/individual_bats.npy')
    print(arnold_tongue.shape)
    print(behavioral_arnold_tongues.shape)

    at_rows, at_cols = arnold_tongue.shape
    bat_rows, bat_cols = behavioral_arnold_tongues.shape[:2]

    row_factor = at_rows // bat_rows
    col_factor = at_cols // bat_cols

    arnold_tongue = arnold_tongue[::row_factor, ::col_factor]
    print(arnold_tongue.shape)
    print(arnold_tongue.flatten())









