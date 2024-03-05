"""
Runs simulations for seven training sessions (sessions 2-8). The first session corresponds to results from the baseline_simulation.py script.
Results are saved in ../data/simulation_results/learning_simulation.npy and correspond to section X of the paper.


NOTES:
- save simulated Arnold tongues per session starting from session 2 based on aggregated parameters
- save results per fold and per session (k-fold cross-validation)
- decide whether k is 1 (i.e., leave-one-subject-out) or 4 (split-half cross-validation)
- think about parallelizing the simulations
"""

BASE_PATH = 'results/analysis/'

import os

import tomllib
import numpy as np
from scipy.optimize import curve_fit

from src.v1_model import V1Model
from src.stimulus_generator import StimulusGenerator
from src.sim_utils import get_num_blocks
from src.anl_utils import order_parameter, weighted_jaccard, compute_size, compute_coherence, compute_weighted_coherence, expand_matrix, min_max_normalize

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
    config_files = ['model', 'stimulus', 'simulation', 'crossvalidation']

    for config_file in config_files:
        with open(f'config/simulation/{config_file}.toml', 'rb') as f:
            parameters[config_file] = tomllib.load(f)

    with open('config/analysis/experiment_actual.toml', 'rb') as f:
        parameters['experiment'] = tomllib.load(f)

    return parameters['model'], parameters['stimulus'], parameters[
        'simulation'], parameters['crossvalidation'], parameters['experiment']


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

    # TO DO: COPY AND ADJUST FUNCTION FROM HIGH_RESOLUTION_SIMULATIONS.PY

    global arnold_tongue, coherence
    global num_conditions, num_entries
    global sync_index, timepoints
    global grid_coarseness, contrast_heterogeneity
    global experiment_parameters, simulation_parameters
    global model, stimulus_generator

    np.random.seed(simulation_parameters['random_seed'] + block)

    for condition, (scaling_factor, contrast_range) in enumerate(
            zip(grid_coarseness, contrast_heterogeneity)):
        stimulus = stimulus_generator.generate(
            scaling_factor, contrast_range,
            experiment_parameters['mean_contrast'])
        model.compute_omega(stimulus.flatten())
        state_variables, _ = model.simulate(simulation_parameters)
        synchronization = np.abs(order_parameter(state_variables))
        index = block * num_conditions + condition
        arnold_tongue[index] = np.mean(synchronization[sync_index])
        lower_index = index * num_entries
        upper_index = lower_index + num_entries
        coherence_placeholder = [
            compute_coherence(state_variables[timepoint])
            for timepoint in timepoints
        ]
        coherence[lower_index:upper_index] = np.mean(coherence_placeholder,
                                                     axis=0)


def run_simulation(counts_tuple):
    """
    Run the simulation.

    Returns
    -------
    arnold_tongue : array_like
        The Arnold tongue.
    coherence : array_like
        The coherence.
    """

    # TO DO: COPY AND ADJUST FUNCTION FROM HIGH_RESOLUTION_SIMULATIONS.PY

    global arnold_tongue, coherence

    num_blocks, num_conditions, num_entries = counts_tuple

    # Initialize the Arnold tongue
    arnold_tongue = np.zeros((num_blocks, num_conditions))
    arnold_tongue = Array('d', arnold_tongue.reshape(-1))

    # Initialize the coherence
    coherence = np.zeros((num_blocks, num_conditions, num_entries))
    coherence = Array('d', coherence.reshape(-1))

    # Run a batch of blocks in parallel
    for batch in range(num_batches):
        with Pool(num_blocks) as p:
            p.map(run_block, range(batch * num_cores, (batch + 1) * num_cores))

    # Collect simulation results
    arnold_tongue = np.array(arnold_tongue).reshape(num_blocks, num_conditions)
    coherence = np.array(coherence).reshape(num_blocks, num_conditions,
                                            num_entries)

    return arnold_tongue, coherence


def run_learning(fold, learning_rate, num_sessions, counts_tuple,
                 condition_space, measurements):
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
    counts_tuple : tuple
        The counts tuple.
    condition_space : tuple
        The condition space.
    measurements : tuple
        The measurements.

    Returns
    -------
    correlation_fits : array_like
        The correlation fits.
    jaccard_fits : array_like
        The Jaccard fits.
    arnold_tongue_size : array_like
        The Arnold tongue size.
    """
    # TO DO: REMOVE GLOBAL VARIABLES AND MOVE TO SIM_UTILS.PY

    global model

    # Unpack the measurements
    arnold_tongue, coherence = measurements

    # Get the number of sessions by popping the first element from the counts tuple
    num_sessions = counts_tuple[0]
    counts_tuple = counts_tuple[1:]

    # Set the learning rate and generate the coupling
    model.effective_learning_rate = learning_rate
    model.generate_coupling()

    # Load the optimal psychometric curve
    file = os.path.join(BASE_PATH, f'session_1',
                        'optimal_psychometric_crossval.npy')
    optimal_psychometric_fold = np.load(file)[fold]

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

        # Run the simulation
        arnold_tongue, coherence = run_simulation(counts_tuple)

        # Compute the weighted coherence and update the coupling
        measurements = (arnold_tongue, coherence)
        weighted_coherence = compute_weighted_coherence(
            counts_tuple, measurements, optimal_psychometric_fold)
        weighted_coherence = expand_matrix(weighted_coherence, diagonal)
        model.update_coupling(weighted_coherence)

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

    # Load the model, stimulus, simulation, and experiment parameters
    model_parameters, stimulus_parameters, simulation_parameters, crossval_parameters, experiment_parameters = load_configurations(
    )
    num_entries = model_parameters['num_populations'] * (
        model_parameters['num_populations'] - 1) // 2
    num_sessions = experiment_parameters['num_training_sessions']
    num_folds = experiment_parameters['num_subjects']

    # Initialize the model and stimulus generator
    model = V1Model(model_parameters, stimulus_parameters)
    stimulus_generator = StimulusGenerator(stimulus_parameters)

    # Set up the simulation and parallel processing
    simulation_parameters['num_time_steps'] = int(
        simulation_parameters['simulation_time'] /
        simulation_parameters['time_step'])

    num_available_cores = cpu_count()
    num_cores = simulation_parameters['num_cores']
    if num_cores > num_available_cores:
        num_cores = num_available_cores

    # Set up the experiment
    num_conditions = experiment_parameters[
        'num_contrast_heterogeneity'] * experiment_parameters[
            'num_grid_coarseness']
    num_blocks = get_num_blocks(experiment_parameters['num_blocks'], num_cores)
    num_batches = num_blocks // num_cores

    # Create a tuple of counts_tuple
    counts_tuple = (num_sessions, num_blocks, num_conditions, num_entries)

    # Set up the grid coarseness and contrast heterogeneity
    contrast_heterogeneity_orig = np.linspace(
        experiment_parameters['min_contrast_heterogeneity'],
        experiment_parameters['max_contrast_heterogeneity'],
        experiment_parameters['num_contrast_heterogeneity'])
    grid_coarseness_orig = np.linspace(
        experiment_parameters['min_grid_coarseness'],
        experiment_parameters['max_grid_coarseness'],
        experiment_parameters['num_grid_coarseness'])

    condition_space = (grid_coarseness_orig, contrast_heterogeneity_orig)

    contrast_heterogeneity = np.tile(
        contrast_heterogeneity_orig,
        experiment_parameters['num_grid_coarseness'])
    grid_coarseness = np.repeat(
        grid_coarseness_orig,
        experiment_parameters['num_contrast_heterogeneity'])

    # Set up the synchronization index and timepoints
    start = simulation_parameters['num_time_steps'] // 2
    sync_index = slice(start, None)
    timepoints = range(start, simulation_parameters['num_time_steps'])

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
                                 num_sessions, counts_tuple, condition_space)

    # Save the results
    results_file = 'results/simulation/learning_simulation.npz'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    np.savez(results_file,
             correlation_fits=correlation_fits,
             jaccard_fits=jaccard_fits,
             arnold_tongue_size=arnold_tongue_size)
