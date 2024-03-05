"""
This script simulates the first session of the experiment and generate an Arnold Tongue.
Results are saved in results/arnold_tongue.npy and correspond to section X of the paper.
"""
import os
import time
import tomllib
import numpy as np

from src.v1_model import V1Model
from src.stimulus_generator import StimulusGenerator
from src.sim_utils import get_num_blocks
from src.anl_utils import order_parameter

from multiprocessing import Pool, cpu_count, Array, Manager


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
    config_files = ['model', 'stimulus', 'simulation']

    for config_file in config_files:
        with open(f'config/simulation/{config_file}.toml', 'rb') as f:
            parameters[config_file] = tomllib.load(f)

    with open('config/analysis/experiment_extended.toml', 'rb') as f:
        parameters['experiment_extended'] = tomllib.load(f)

    return parameters['model'], parameters['stimulus'], parameters[
        'simulation'], parameters['experiment_extended']


def run_block(block, experiment_parameters, simulation_parameters,
              stimulus_conditions, simulation_classes, indexing):
    """
    Run a block of the Arnold tongue. This function is used for parallel processing.

    Parameters
    ----------
    block : int
        The block number.
    experiment_parameters : dict
        The experiment parameters.
    simulation_parameters : dict
        The simulation parameters.
    stimulus_conditions : tuple
        The stimulus conditions.
    simulation_classes : tuple
        The simulation classes.
    indexing : tuple
        The indexing for synchronization.
    """
    global arnold_tongue

    grid_coarseness, contrast_heterogeneity = stimulus_conditions
    model, stimulus_generator = simulation_classes
    sync_index, _ = indexing

    np.random.seed(simulation_parameters['random_seed'] + block)

    for condition, (scaling_factor, contrast_range) in enumerate(
            zip(grid_coarseness, contrast_heterogeneity)):
        stimulus = stimulus_generator.generate(
            scaling_factor, contrast_range,
            experiment_parameters['mean_contrast'])
        model.compute_omega(stimulus.flatten())
        state_variables, _ = model.simulate(simulation_parameters)
        synchronization = np.abs(order_parameter(state_variables))
        index = block * experiment_parameters['num_conditions'] + condition
        arnold_tongue[index] = np.mean(synchronization[sync_index])


def run_simulation(experiment_parameters, simulation_parameters,
                   stimulus_conditions, simulation_classes, indexing):
    """
    Run the simulation.

    Parameters
    ----------
    experiment_parameters : dict
        The experiment parameters.
    simulation_parameters : dict
        The simulation parameters.
    stimulus_conditions : tuple
        The stimulus conditions.
    simulation_classes : tuple
        The simulation classes.
    indexing : tuple
        The indexing for synchronization.
        
    Returns
    -------
    arnold_tongue : array_like
        The Arnold tongue.
    """

    global arnold_tongue

    # Initialize the Arnold tongue
    arnold_tongue = np.zeros((experiment_parameters['num_blocks'],
                              experiment_parameters['num_conditions']))
    arnold_tongue = Array('d', arnold_tongue.reshape(-1))

    # Run a batch of blocks in parallel
    for batch in range(simulation_parameters['num_batches']):
        with Pool(experiment_parameters['num_blocks']) as p:
            p.starmap(
                run_block,
                [(block, experiment_parameters, simulation_parameters,
                  stimulus_conditions, simulation_classes, indexing)
                 for block in range(batch * simulation_parameters['num_cores'],
                                    (batch + 1) *
                                    simulation_parameters['num_cores'])])

    # Collect simulation results
    arnold_tongue = np.array(arnold_tongue).reshape(
        experiment_parameters['num_blocks'],
        experiment_parameters['num_grid_coarseness'],
        experiment_parameters['num_contrast_heterogeneity'])

    return arnold_tongue


if __name__ == '__main__':

    # Load the parameters
    model_parameters, stimulus_parameters, simulation_parameters, experiment_parameters = load_configurations(
    )

    # Load learning rates
    #crossval_results = np.load('results/simulation/crossval_estimation.npz')
    #learning_rates = crossval_results['learning_rate_crossval']

    # Initialize the model and stimulus generator
    model = V1Model(model_parameters, stimulus_parameters)
    stimulus_generator = StimulusGenerator(stimulus_parameters)

    simulation_classes = (model, stimulus_generator)

    # Set up the simulation and parallel processing
    simulation_parameters['num_time_steps'] = int(
        simulation_parameters['simulation_time'] /
        simulation_parameters['time_step'])

    num_cores = min(cpu_count(), simulation_parameters['num_cores'])

    # Set up parallel processing
    num_blocks = get_num_blocks(experiment_parameters['num_blocks'], num_cores)
    num_batches = num_blocks // num_cores

    experiment_parameters.update({'num_blocks': num_blocks})
    simulation_parameters.update({
        'num_cores': num_cores,
        'num_batches': num_batches
    })

    # Set up the stimulus conditions
    contrast_heterogeneity = np.linspace(
        experiment_parameters['min_contrast_heterogeneity'],
        experiment_parameters['max_contrast_heterogeneity'],
        experiment_parameters['num_contrast_heterogeneity'])
    grid_coarseness = np.linspace(experiment_parameters['min_grid_coarseness'],
                                  experiment_parameters['max_grid_coarseness'],
                                  experiment_parameters['num_grid_coarseness'])

    contrast_heterogeneity = np.tile(
        contrast_heterogeneity, experiment_parameters['num_grid_coarseness'])
    grid_coarseness = np.repeat(
        grid_coarseness, experiment_parameters['num_contrast_heterogeneity'])

    stimulus_conditions = (grid_coarseness, contrast_heterogeneity)

    # Set indexing for synchronization
    sync_index = slice(simulation_parameters['num_time_steps'] // 2, None)
    indexing = (sync_index, None)

    # Run the simulation

    arnold_tongue = run_simulation(experiment_parameters,
                                   simulation_parameters, stimulus_conditions,
                                   simulation_classes, indexing)

    # Save the results
    file = 'results/simulation/baseline_arnold_tongue_test.npy'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    np.save(file, arnold_tongue)
