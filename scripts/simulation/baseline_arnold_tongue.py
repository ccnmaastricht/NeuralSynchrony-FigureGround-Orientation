"""
This script simulates the first session of the experiment and generate an Arnold Tongue.
Results are saved in results/arnold_tongue.npy and correspond to section X of the paper.
"""
import os

import tomllib
import numpy as np

from src.v1_model import V1Model
from src.stimulus_generator import StimulusGenerator
from src.sim_utils import get_num_blocks
from src.anl_utils import order_parameter

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
    """
    parameters = {}
    config_files = ['model', 'stimulus', 'simulation', 'experiment_extended']
    
    for config_file in config_files:
        with open(f'config/simulation/{config_file}.toml', 'rb') as f:
            parameters[config_file] = tomllib.load(f)

    return parameters['model'], parameters['stimulus'], parameters['simulation'], parameters['experiment_extended']


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

    # Load the parameters
    model_parameters, stimulus_parameters, simulation_parameters, experiment_parameters = load_configurations()


    # Initialize the model and stimulus generator
    model = V1Model(model_parameters, stimulus_parameters)
    stimulus_generator = StimulusGenerator(stimulus_parameters)

    # Set up the simulation and parallel processing
    simulation_parameters['num_time_steps'] = int(simulation_parameters['simulation_time'] / simulation_parameters['time_step'])
    simulation_parameters['initial_state'] = np.random.rand(model_parameters['num_populations']) * 2 * np.pi

    num_available_cores = cpu_count()
    num_cores = int(num_available_cores * simulation_parameters['proportion_used_cores'])

    print(f'Using {num_cores} of {num_available_cores} available cores.')
    
    # Set up the experiment
    num_conditions = experiment_parameters['num_contrast_heterogeneity'] * experiment_parameters['num_grid_coarseness']
    num_blocks = get_num_blocks(experiment_parameters['num_blocks'], num_cores)
    num_batches = num_blocks // num_cores

    print(f'Running {num_blocks} blocks in {num_batches} batches.')
    
    contrast_heterogeneity = np.linspace(0.01, 1, experiment_parameters['num_contrast_heterogeneity'])
    grid_coarseness = np.linspace(1, 1.5, experiment_parameters['num_grid_coarseness'])

    contrast_heterogeneity = np.tile(contrast_heterogeneity, experiment_parameters['num_grid_coarseness'])
    grid_coarseness = np.repeat(grid_coarseness, experiment_parameters['num_contrast_heterogeneity'])

    # Initialize the Arnold tongue
    arnold_tongue = np.zeros((num_blocks, num_conditions))
    arnold_tongue = Array('d', arnold_tongue.reshape(-1))

    sync_index = slice(simulation_parameters['num_time_steps'] // 2, None)

    # Run the experiment
    for batch in range(num_batches):
        print(f' Blocks {batch * num_cores} to {(batch + 1) * num_cores - 1}')
        with Pool(num_blocks) as p:
            p.map(run_block, range(batch * num_cores, (batch + 1) * num_cores))

    # Retrieve the results
    arnold_tongue = np.array(arnold_tongue).reshape(num_blocks, num_conditions)
    arnold_tongue = arnold_tongue.reshape(num_blocks, experiment_parameters['num_contrast_heterogeneity'],
                                                       experiment_parameters['num_grid_coarseness'])
    
    # Save the results
    file = 'results/simulation/baseline_arnold_tongue.npy'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    np.save(file, arnold_tongue)



