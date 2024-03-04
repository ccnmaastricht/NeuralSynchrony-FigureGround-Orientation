"""
This script compute fold-specific paramters of a sychometric function linking model synchrony to probability of a correct response of subjects in the first session of the experiment.
Results are saved in results/arnold_tongue.npy and correspond to section X of the paper.
"""

import os
import tomllib
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
from multiprocessing import Pool, Array, cpu_count

from src.v1_model import V1Model
from src.stimulus_generator import StimulusGenerator
from src.sim_utils import get_num_blocks
from src.anl_utils import order_parameter, compute_coherence, compute_weighted_coherence, psychometric_function, expand_matrix, weighted_jaccard


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


def coarse_to_fine(weighted_coherence, behavioral_arnold_tongue,
                   crossval_parameters, counts_tuple):
    """
    Estimate the effective learning rate through coarse-to-fine grid search.

    Parameters
    ----------
    weighted_coherence : array_like
        The weighted coherence.
    behavioral_arnold_tongue : array_like
        The behavioral Arnold tongue.
    crossval_parameters : dict
        The cross-validation parameters.
    counts_tuple : tuple
        The number of blocks, conditions, and entries.

    Returns
    -------
    float
        The effective learning rate.
    """

    global model

    diagonal = np.ones(model.num_populations)
    weighted_coherence = expand_matrix(weighted_coherence, diagonal)

    effective_learning_rates = np.linspace(
        crossval_parameters['effective_learning_rate_min'],
        crossval_parameters['effective_learning_rate_max'],
        crossval_parameters['num_effective_learning_rate'])

    for _ in tqdm(range(crossval_parameters['num_grids'])):
        weighted_jaccard_fits = simulation_grid(
            effective_learning_rates, weighted_coherence,
            behavioral_arnold_tongue,
            crossval_parameters['num_effective_learning_rate'], counts_tuple)
        lower_bound = np.argmax(weighted_jaccard_fits)
        upper_bound = lower_bound + 2
        best_index = lower_bound + 1
        best_learning_rate = effective_learning_rates[best_index]
        effective_learning_rates = np.linspace(
            lower_bound, upper_bound,
            crossval_parameters['num_effective_learning_rate'])

    return best_learning_rate


def simulation_grid(effective_learning_rates, weighted_coherence,
                    behavioral_arnold_tongue, num_effective_learning_rate,
                    counts_tuple):
    """
    Run a grid of simulations.

    Parameters
    ----------
    effective_learning_rates : array_like
        The effective learning rates.
    weighted_coherence : array_like
        The weighted coherence.
    behavioral_arnold_tongue : array_like
        The behavioral Arnold tongue.
    num_effective_learning_rate : int
        The number of effective learning rates.
    counts_tuple : tuple
        The number of blocks, conditions, and entries.

    Returns
    -------
    array_like
        The weighted Jaccard fits.
    """
    global arnold_tongue
    global model

    weighted_jaccard_fits = np.zeros(num_effective_learning_rate - 2)
    for i, effective_learning_rate in enumerate(
            effective_learning_rates[1:-1]):
        model.effective_learning_rate = effective_learning_rate
        model.generate_coupling()
        model.update_coupling(weighted_coherence)

        arnold_tongue, _ = run_simulation(counts_tuple)
        weighted_jaccard_fits[i] = weighted_jaccard(arnold_tongue.mean(axis=0),
                                                    behavioral_arnold_tongue)

    return weighted_jaccard_fits


if __name__ == '__main__':

    # Load the model, stimulus, simulation, and experiment parameters
    model_parameters, stimulus_parameters, simulation_parameters, crossval_parameters, experiment_parameters = load_configurations(
    )
    num_entries = model_parameters['num_populations'] * (
        model_parameters['num_populations'] - 1) // 2

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
    counts_tuple = (num_blocks, num_conditions, num_entries)

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

    # Set up the synchronization index and timepoints
    start = simulation_parameters['num_time_steps'] // 2
    sync_index = slice(start, None)
    timepoints = range(start, simulation_parameters['num_time_steps'])

    # Run the simulation
    arnold_tongue, coherence = run_simulation(counts_tuple)

    # load behavioral Arnold tongues of session 1
    session1_arnold_tongues = np.load(
        'results/analysis/session_1/individual_bats.npy')

    # load behavioral Arnold tongues of session 2
    session2_arnold_tongues = np.load(
        'results/analysis/session_2/individual_bats.npy')

    # create predictors for the psychometric function
    predictors = np.ones((2, num_conditions))
    predictors[0] = arnold_tongue.mean(axis=0)

    optimal_psychometric_crossval = np.zeros(
        (experiment_parameters['num_subjects'], 2))
    weighted_coherence_crossval = np.zeros(
        (experiment_parameters['num_subjects'], num_entries))
    learning_rate_crossval = np.zeros(experiment_parameters['num_subjects'])

    for subject in range(experiment_parameters['num_subjects']):

        # remove subject from behavioral Arnold tongues of session 1
        fold_arnold_tongues = np.delete(session1_arnold_tongues,
                                        subject,
                                        axis=0)

        # Compute average behavioral Arnold tongue of session 1
        average_arnold_tongue = fold_arnold_tongues.mean(axis=0)

        # Initial guesses for parameters
        initial_params = np.zeros(2)

        # Fit psychometric function to session 1 data
        popt, _ = curve_fit(psychometric_function,
                            predictors,
                            average_arnold_tongue.flatten(),
                            p0=initial_params)
        optimal_psychometric_crossval[subject] = popt

        # Estimate weighted coherence from session 1 data
        measurements = (arnold_tongue, coherence)
        weighted_coherence = compute_weighted_coherence(
            counts_tuple, measurements, popt)

        weighted_coherence_crossval[subject] = weighted_coherence

        # remove subject from behavioral Arnold tongues of session 2
        fold_arnold_tongues = np.delete(session2_arnold_tongues,
                                        subject,
                                        axis=0)

        # Compute average behavioral Arnold tongue of session 2
        average_arnold_tongue = fold_arnold_tongues.mean(axis=0)

        learning_rate_crossval[subject] = coarse_to_fine(
            weighted_coherence, average_arnold_tongue, crossval_parameters)

    # Save results
    np.savez('results/simulation/crossval_estimation.npz',
             optimal_psychometric_crossval=optimal_psychometric_crossval,
             weighted_coherence_crossval=weighted_coherence_crossval,
             learning_rate_crossval=learning_rate_crossval)
