import os
import tomllib
import pandas as pd
import numpy as np
from src.stat_utils import bootstrap

def load_configuration():
    """
    Load parameters for the in silico experiment.

    Returns
    -------
    experiment_parameters : dict
        The experiment parameters.
    """
    with open('config/bootstrap.toml', 'rb') as f:
        bootstrap_parameters = tomllib.load(f)

    return bootstrap_parameters

if __name__ == '__main__':
    # Load configuration
    bootstrap_parameters = load_configuration()

    # Load data
    experimental_data = pd.read_csv('data/main.csv')

    # Bootstrap
    num_repeats = bootstrap_parameters['num_repetitions']
    session_id = bootstrap_parameters['session_id']
    mean_difference_distribution = bootstrap(experimental_data, num_repeats, session_id)

    # Save results
    file = 'results/statistics/bootstrap.npy'
    os.makedirs('results/bootstrap', exist_ok=True)
    np.save(file, mean_difference_distribution)
    
    

