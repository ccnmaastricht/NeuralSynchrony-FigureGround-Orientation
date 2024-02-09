"""
Runs simulations for seven training sessions (sessions 2-8). The first session corresponds to results from the baseline_simulation.py script.
Results are saved in ../data/simulation_results/learning_simulation.npy and correspond to section X of the paper.


NOTES:
- save simulated Arnold tongues per session starting from session 2 based on aggregated parameters
- save results per fold and per session (k-fold cross-validation)
- decide whether k is 1 (i.e., leave-one-subject-out) or 4 (split-half cross-validation)
- think about parallelizing the simulations
"""

import os

import tomllib
import numpy as np
from scipy.optimize import curve_fit

from src.v1_model import V1Model
from src.stimulus_generator import StimulusGenerator
from src.sim_utils import get_num_blocks
from src.anl_utils import order_parameter, weighted_jaccard, min_max_normalize, psychometric_function, compute_coherence

from multiprocessing import Pool, Array, cpu_count
def load_configurations():
    """
    Load the model, stimulus, simulation, and experiment parameters.

    Returns
    -------
    model_parameters : dict
        The model parameters.
    stimulus_parameters : dict
        The stimulus parameters.
    simulation_parameters : dict
        The simulation parameters.
    experiment_parameters : dict
        The experiment parameters.
    cross_validation_parameters : dict
        The cross-validation parameters.
    """
    parameters = {}
    config_files = ['model', 'stimulus', 'simulation', 'experiment_actual', 'cross_validation']
    
    for config_file in config_files:
        with open(f'config/simulation/{config_file}.toml', 'rb') as f:
            parameters[config_file] = tomllib.load(f)

    return parameters['model'], parameters['stimulus'], parameters['simulation'], parameters['experiment_actual'], parameters['cross_validation']

def run_block(block):
    """
    Run a block of the Arnold tongue. This function is used for parallel processing.

    Parameters
    ----------
    block : int
        The block number.

    Returns
    -------
    None
    """
    global weightedcoherence, num_pairs, sync_index
    global grid_coarseness, contrast_heterogeneity
    global experiment_parameters, simulation_parameters
    global model, stimulus_generator
    
    for condition, (scaling_factor, contrast_range) in enumerate(zip(grid_coarseness, contrast_heterogeneity)):
        stimulus = stimulus_generator.generate(scaling_factor,
                                               contrast_range,
                                               experiment_parameters['mean_contrast'])
        model.compute_omega(stimulus.flatten())
        state_variables, _ = model.simulate(simulation_parameters)
        state_variables = state_variables[sync_index]
        coheherence = compute_coherence(state_variables)
        coheherence = np.mean(coheherence, axis=2)
        weighted_coherence += coheherence * probability[condition]

def simulation(sync_slice):
    
    # Initialize the coherence 
    coherence = np.zeros((num_blocks, num_pairs))
    coherence = Array('d', coherence.reshape(-1))

    

    # Run the experiment
    for batch in range(num_batches):
        print(f' Blocks {batch * num_cores} to {(batch + 1) * num_cores - 1}')
        with Pool(num_blocks) as p:
            p.map(run_block, range(batch * num_cores, (batch + 1) * num_cores))

    # Retrieve the results
    arnold_tongue = np.array(arnold_tongue).reshape(num_blocks, num_conditions)
    arnold_tongue = arnold_tongue.reshape(num_blocks, experiment_parameters['num_contrast_heterogeneity'],
                                                       experiment_parameters['num_grid_coarseness'])


if __name__ == '__main__':


    # slice_time_steps = num_time_steps - keep_time_steps
    sync_index = slice(slice_time_steps, None)

    num_pairs = model_parameters['num_populations']**2

