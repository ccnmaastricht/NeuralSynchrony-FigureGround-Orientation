"""
This script explores the decay rate (lambda) and maximum coupling (gamma) parameter space of the model
 and generates an Arnold Tongue for each combination of parameters, and compares the results to behavioral data.
Results are saved in ../data/simulation_results/parameter_space_exploration.npy and correspond to section X of the paper.
"""

import os

import json
import numpy as np

from src.v1_model import V1Model
from src.stimulus_generator import StimulusGenerator
from src.sim_utils import get_num_blocks
from src.anl_utils import order_parameter, weighted_jaccard, min_max_normalize

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
    exploration_parameters : dict
        The parameters for the parameter space exploration.
    """
    parameters = {}
    config_files = ['model_parameters', 'stimulus_parameters', 'simulation_parameters', 'experiment_parameters', 'exploration_parameters']
    
    for config_file in config_files:
        with open(f'config/{config_file}.json') as f:
            parameters[config_file] = json.load(f)

    return parameters['model_parameters'], parameters['stimulus_parameters'], parameters['simulation_parameters'], parameters['experiment_parameters'], parameters['exploration_parameters']

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
    global arnold_tongue, num_conditions, sync_index
    global grid_coarseness, contrast_heterogeneity
    global experiment_parameters, simulation_parameters
    global model, stimulus_generator
    
    for condition, (scaling_factor, contrast_range) in enumerate(zip(grid_coarseness, contrast_heterogeneity)):
        stimulus = stimulus_generator.generate(scaling_factor,
                                               contrast_range,
                                               experiment_parameters['mean_contrast'])
        model.compute_omega(stimulus.flatten())
        state_variables, _ = model.simulate(simulation_parameters)
        synchronization = np.abs(order_parameter(state_variables))
        index = block * num_conditions + condition
        arnold_tongue[index] = np.mean(synchronization[sync_index])


if __name__ == '__main__':
    # Load empirical (behavioral) Arnold tongue
    behavioral_arnold_tongue = np.load('data/results/empirical/continuous_bat.npy').flatten()
    behavioral_arnold_tongue = min_max_normalize(behavioral_arnold_tongue)

    # Load the parameters
    model_parameters, stimulus_parameters, simulation_parameters, experiment_parameters, exploration_parameters = load_configurations()

    # Initialize the stimulus generator
    stimulus_generator = StimulusGenerator(stimulus_parameters)

    # Set up the simulation and parallel processing
    simulation_parameters['num_time_steps'] = int(simulation_parameters['simulation_time'] / simulation_parameters['time_step'])
    simulation_parameters['initial_state'] = np.random.rand(model_parameters['num_populations']) * 2 * np.pi

    num_available_cores = cpu_count()
    num_cores = int(num_available_cores * simulation_parameters['proportion_used_cores'])

    print(f'Using {num_cores} of {num_available_cores} available cores.')

    # Set up the synchronization index
    sync_index = slice(simulation_parameters['num_time_steps'] // 2, None)
    
    # Set up single experiment
    num_conditions = experiment_parameters['num_contrast_heterogeneity'] * experiment_parameters['num_grid_coarseness']
    num_blocks = get_num_blocks(experiment_parameters['num_blocks'], num_cores)
    num_batches = num_blocks // num_cores
    
    contrast_heterogeneity = np.linspace(0.01, 1, experiment_parameters['num_contrast_heterogeneity'])
    grid_coarseness = np.linspace(1, 1.5, experiment_parameters['num_grid_coarseness'])

    contrast_heterogeneity = np.tile(contrast_heterogeneity, experiment_parameters['num_grid_coarseness'])
    grid_coarseness = np.repeat(grid_coarseness, experiment_parameters['num_contrast_heterogeneity'])

    # Set up the exploration
    decay_rates = np.linspace(exploration_parameters['decay_rate_min'],
                              exploration_parameters['decay_rate_max'],
                              exploration_parameters['num_decay'])
    max_couplings = np.linspace(exploration_parameters['max_coupling_min'],
                                exploration_parameters['max_coupling_max'],
                                exploration_parameters['num_coupling'])
    
    size = (exploration_parameters['num_decay'], exploration_parameters['num_coupling'])
    correlation_fits = np.zeros(size)
    jaccard_fits = np.zeros(size)

    # Initialize the Arnold tongue
    arnold_tongue = np.zeros((num_blocks, num_conditions))
    arnold_tongue = Array('d', arnold_tongue.reshape(-1))

    # Run the exploration
    for decay_counter, decay_rate in enumerate(decay_rates):
        for coupling_counter, max_coupling in enumerate(max_couplings):
            print(f'Running decay rate {decay_counter + 1} of {len(decay_rates)} and coupling {coupling_counter + 1} of {len(max_couplings)}')
            
            # Set the parameters
            model_parameters['decay_rate'] = decay_rate
            model_parameters['max_coupling'] = max_coupling

            # Initialize the model
            model = V1Model(model_parameters, stimulus_parameters)
    
            # Run the experiment
            for batch in range(num_batches):
                with Pool(num_blocks) as p:
                    p.map(run_block, range(batch * num_cores, (batch + 1) * num_cores))

            # Retrieve the results
            simulated_arnold_tongue = np.array(arnold_tongue)

            # Compute the fits
            correlation_fits[decay_counter, coupling_counter] = np.corrcoef(simulated_arnold_tongue, behavioral_arnold_tongue)[0, 1]
            jaccard_fits[decay_counter, coupling_counter] = weighted_jaccard(simulated_arnold_tongue, behavioral_arnold_tongue)
        
    
    # Save the results
    file = 'data/simulation_results/parameter_space_exploration'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    np.savez(file, correlation_fits=correlation_fits, jaccard_fits=jaccard_fits, decay_rates=decay_rates, max_couplings=max_couplings)