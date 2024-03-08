"""
This script explores the decay rate (lambda) and maximum coupling (gamma) parameter space of the model
 and generates an Arnold Tongue for each combination of parameters, and compares the results to behavioral data.
Results are saved in results/simulation/parameter_space_exploration.npy and correspond to section X of the paper.
"""

import os
import time
import tomllib
import numpy as np

from src.v1_model import V1Model
from src.stimulus_generator import StimulusGenerator
from src.sim_utils import setup_parallel_processing, generate_stimulus_conditions, generate_condition_space, generate_time_index, generate_exploration_space
from src.anl_utils import order_parameter, weighted_jaccard, min_max_normalize

from multiprocessing import Pool, Array


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
    config_files = ['model', 'stimulus', 'simulation', 'exploration']

    for config_file in config_files:
        with open(f'config/simulation/{config_file}.toml', 'rb') as f:
            parameters[config_file] = tomllib.load(f)

    with open(f'config/analysis/experiment_actual.toml', 'rb') as f:
        parameters['experiment_actual'] = tomllib.load(f)

    return parameters['model'], parameters['stimulus'], parameters[
        'simulation'], parameters['experiment_actual'], parameters[
            'exploration']


def run_block(block, experiment_parameters, simulation_parameters,
              stimulus_conditions, simulation_classes, sync_index):
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
                   stimulus_conditions, simulation_classes, sync_index):
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
    sync_index : slice
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
                  stimulus_conditions, simulation_classes, sync_index)
                 for block in range(batch * simulation_parameters['num_cores'],
                                    (batch + 1) *
                                    simulation_parameters['num_cores'])])

    # Collect simulation results
    arnold_tongue = np.array(arnold_tongue).reshape(
        experiment_parameters['num_blocks'],
        experiment_parameters['num_conditions'])

    return arnold_tongue


if __name__ == '__main__':
    # Load empirical (behavioral) Arnold tongue
    behavioral_arnold_tongue = np.load(
        'results/empirical/session_1/average_bat.npy').flatten()
    behavioral_arnold_tongue = min_max_normalize(behavioral_arnold_tongue)

    # Load the parameters
    model_parameters, stimulus_parameters, simulation_parameters, experiment_parameters, exploration_parameters = load_configurations(
    )

    # Initialize the stimulus generator
    stimulus_generator = StimulusGenerator(stimulus_parameters)

    # Set up parallel processing
    simulation_parameters, experiment_parameters = setup_parallel_processing(
        simulation_parameters, experiment_parameters)

    # Set up the stimulus conditions
    stimulus_conditions = generate_stimulus_conditions(experiment_parameters)

    # Set up the condition space
    condition_space = generate_condition_space(experiment_parameters)

    # Set up the synchronization index and timepoint
    sync_index, _ = generate_time_index(simulation_parameters)

    # Set up the exploration
    decay_rates, max_couplings = generate_exploration_space(
        exploration_parameters)

    size = (exploration_parameters['num_decay'],
            exploration_parameters['num_max_coupling'])
    correlation_fits = np.zeros(size)
    jaccard_fits = np.zeros(size)

    # Run the exploration
    for decay_counter, decay_rate in enumerate(decay_rates):
        for coupling_counter, max_coupling in enumerate(max_couplings):
            print(
                f'Running decay rate {decay_counter + 1} of {len(decay_rates)} and coupling {coupling_counter + 1} of {len(max_couplings)}'
            )

            # Set the parameters
            model_parameters['decay_rate'] = decay_rate
            model_parameters['max_coupling'] = max_coupling

            # Initialize the model
            model = V1Model(model_parameters, stimulus_parameters)

            # Initialize the simulation classes
            simulation_classes = (model, stimulus_generator)

            # Run the simulation
            simulated_arnold_tongue = run_simulation(experiment_parameters,
                                                     simulation_parameters,
                                                     stimulus_conditions,
                                                     simulation_classes,
                                                     sync_index)
            simulated_arnold_tongue = np.mean(simulated_arnold_tongue, axis=0)

            # Compute the fits
            correlation_fits[decay_counter, coupling_counter] = np.corrcoef(
                simulated_arnold_tongue, behavioral_arnold_tongue)[0, 1]
            jaccard_fits[decay_counter, coupling_counter] = weighted_jaccard(
                simulated_arnold_tongue, behavioral_arnold_tongue)

    # Save the results
    file = 'results/simulation/parameter_space_exploration'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    np.savez(file,
             correlation_fits=correlation_fits,
             jaccard_fits=jaccard_fits,
             decay_rates=decay_rates,
             max_couplings=max_couplings)
