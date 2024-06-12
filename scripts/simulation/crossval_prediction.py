"""
Runs simulations for seven training sessions (sessions 2-8). The first session corresponds to results from the baseline_simulation.py script.
Results are saved in ../data/simulation_results/learning_simulation.npy.
"""

BASE_PATH = 'results/empirical/'

import os

import tomllib
import numpy as np

from src.sim_utils import initialize_simulation_classes, setup_parallel_processing, generate_stimulus_conditions, generate_condition_space, generate_time_index
from src.anl_utils import order_parameter, weighted_jaccard, compute_size, compute_phase_difference, compute_weighted_locking, expand_matrix, min_max_normalize

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
    """
    parameters = {}
    config_files = ['model', 'stimulus', 'simulation', 'crossvalidation']

    for config_file in config_files:
        with open(f'config/simulation/{config_file}.toml', 'rb') as f:
            parameters[config_file] = tomllib.load(f)

    with open('config/analysis/experiment_actual.toml', 'rb') as f:
        parameters['experiment'] = tomllib.load(f)

    return parameters['model'], parameters['stimulus'], parameters[
        'simulation'], parameters['crossvalidation'], parameters['experiment']


def run_block(block, experiment_parameters, simulation_parameters, num_entries,
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
    global arnold_tongue, locking

    grid_coarseness, contrast_heterogeneity = stimulus_conditions
    model, stimulus_generator = simulation_classes
    sync_index, timepoints = indexing

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
        lower_index = index * num_entries
        upper_index = lower_index + num_entries
        phase_differences = [
            compute_phase_difference(state_variables[timepoint])
            for timepoint in timepoints
        ]
        locking[lower_index:upper_index] = np.abs(
            np.mean(phase_differences, axis=0))


def run_simulation(experiment_parameters, simulation_parameters, num_entries,
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

    global arnold_tongue, locking

    # Initialize the Arnold tongue
    arnold_tongue = np.zeros((experiment_parameters['num_blocks'],
                              experiment_parameters['num_conditions']))
    arnold_tongue = Array('d', arnold_tongue.reshape(-1))

    # Initialize the locking
    locking = np.zeros((experiment_parameters['num_blocks'],
                        experiment_parameters['num_conditions'], num_entries))
    locking = Array('d', locking.reshape(-1))

    # Run a batch of blocks in parallel
    for batch in range(simulation_parameters['num_batches']):
        with Pool(experiment_parameters['num_blocks']) as p:
            p.starmap(
                run_block,
                [(block, experiment_parameters, simulation_parameters,
                  num_entries, stimulus_conditions, simulation_classes,
                  indexing)
                 for block in range(batch * simulation_parameters['num_cores'],
                                    (batch + 1) *
                                    simulation_parameters['num_cores'])])

    # Collect simulation results
    arnold_tongue = np.array(arnold_tongue).reshape(
        experiment_parameters['num_blocks'],
        experiment_parameters['num_conditions'])

    locking = np.array(locking).reshape(
        experiment_parameters['num_blocks'],
        experiment_parameters['num_conditions'], num_entries)

    return arnold_tongue, locking


def run_learning(fold, learning_rate, num_sessions, experiment_parameters,
                 simulation_parameters, num_entries, stimulus_conditions,
                 condition_space, simulation_classes, indexing):
    """
    Run the learning simulation.

    Parameters
    ----------
    fold : int
        The fold number.
    learning_rate : float
        The learning rate.
    num_sessions : int
        The number of sessions.
    experiment_parameters : dict
        The experiment parameters.
    simulation_parameters : dict
        The simulation parameters.
    num_entries : int
        The number of entries.
    stimulus_conditions : tuple
        The stimulus conditions.
    condition_space : tuple
        The condition space.
    simulation_classes : tuple
        The simulation classes.
    indexing : tuple
        The indexing for synchronization.


    Returns
    -------
    correlation_fits : array_like
        The correlation fits.
    jaccard_fits : array_like
        The Jaccard fits.
    arnold_tongue_size : array_like
        The Arnold tongue size.
    """

    model, stimulus_generator = simulation_classes
    # Set the learning rate and generate the coupling
    model.generate_coupling()
    model.effective_learning_rate = learning_rate

    # Load the optimal psychometric curve
    data = np.load('results/simulation/crossval_estimation.npz')
    optimal_psychometric_fold = data['optimal_psychometric_crossval'][fold]

    # Initialize the fits
    correlation_fits = np.zeros(num_sessions)
    jaccard_fits = np.zeros(num_sessions)
    arnold_tongue_size = np.zeros(num_sessions)

    # Initialize the diagonal
    diagonal = np.ones(model.num_populations)

    # Run the learning simulation
    for session in range(num_sessions):
        # Load the left-out Arnold tongue and normalize it
        file = os.path.join(BASE_PATH, f'session_{session + 1}',
                            'individual_bats.npy')
        left_out_arnold_tongue = np.load(file)[fold].flatten()
        left_out_arnold_tongue = min_max_normalize(left_out_arnold_tongue)

        simulation_classes = (model, stimulus_generator)

        # Run the simulation
        arnold_tongue, locking = run_simulation(experiment_parameters,
                                                simulation_parameters,
                                                num_entries,
                                                stimulus_conditions,
                                                simulation_classes, indexing)

        # Compute the weighted locking and update the coupling
        weighted_locking = compute_weighted_locking(
            experiment_parameters['num_conditions'],
            experiment_parameters['num_blocks'], num_entries, arnold_tongue,
            locking, optimal_psychometric_fold)
        weighted_locking = expand_matrix(weighted_locking, diagonal)
        model.update_coupling(weighted_locking)

        # Compute the fits
        simulated_arnold_tongue = arnold_tongue.mean(axis=0)
        correlation_fits[session] = np.corrcoef(simulated_arnold_tongue,
                                                left_out_arnold_tongue)[0, 1]
        jaccard_fits[session] = weighted_jaccard(simulated_arnold_tongue,
                                                 left_out_arnold_tongue)

        # Compute the Arnold tongue size
        simulated_arnold_tongue = simulated_arnold_tongue.reshape(
            experiment_parameters['num_grid_coarseness'],
            experiment_parameters['num_contrast_heterogeneity'])
        arnold_tongue_size[session] = compute_size(simulated_arnold_tongue,
                                                   condition_space[0],
                                                   condition_space[1])

    return correlation_fits, jaccard_fits, arnold_tongue_size


if __name__ == '__main__':

    # Load parameters
    model_parameters, stimulus_parameters, simulation_parameters, crossval_parameters, experiment_parameters = load_configurations(
    )

    # Derive additional parameters
    num_entries = model_parameters['num_populations'] * (
        model_parameters['num_populations'] - 1) // 2
    num_sessions = experiment_parameters['num_training_sessions']
    num_folds = experiment_parameters['num_subjects']

    # Initialize the model and stimulus generator
    simulation_classes = initialize_simulation_classes(model_parameters,
                                                       stimulus_parameters)

    # Set up parallel processing
    simulation_parameters, experiment_parameters = setup_parallel_processing(
        simulation_parameters, experiment_parameters)

    # Set up the stimulus conditions
    stimulus_conditions = generate_stimulus_conditions(experiment_parameters)

    # Set up the condition space
    condition_space = generate_condition_space(experiment_parameters)

    # Set up the synchronization index and timepoint
    indexing = generate_time_index(simulation_parameters)

    # Load fold-specific effective learning rates
    crossval_results = np.load('results/simulation/crossval_estimation.npz')
    effective_learning_rates = crossval_results['learning_rate_crossval']

    correlation_fits = np.zeros((num_folds, num_sessions))
    jaccard_fits = np.zeros((num_folds, num_sessions))
    arnold_tongue_size = np.zeros((num_folds, num_sessions))

    # Run the learning simulation
    for fold in range(num_folds):
        correlation_fits[fold], jaccard_fits[fold], arnold_tongue_size[
            fold] = run_learning(fold, effective_learning_rates[fold],
                                 num_sessions, experiment_parameters,
                                 simulation_parameters, num_entries,
                                 stimulus_conditions, condition_space,
                                 simulation_classes, indexing)

    # Save the results
    results_file = 'results/simulation/learning_simulation.npz'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    np.savez(results_file,
             correlation_fits=correlation_fits,
             jaccard_fits=jaccard_fits,
             arnold_tongue_size=arnold_tongue_size)
