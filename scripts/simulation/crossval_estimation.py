"""
This script compute fold-specific paramters of a sychometric function linking model synchrony to probability of a correct response of subjects in the first session of the experiment.
Results are saved in results/arnold_tongue.npy and correspond to section X of the paper.
"""

import os
import tomllib
import numpy as np
from scipy.optimize import curve_fit
from multiprocessing import Pool, Array

from src.sim_utils import initialize_simulation_classes, setup_parallel_processing, generate_stimulus_conditions, generate_time_index
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
    global arnold_tongue, coherence

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
        coherence_placeholder = [
            compute_coherence(state_variables[timepoint])
            for timepoint in timepoints
        ]
        coherence[lower_index:upper_index] = np.mean(coherence_placeholder,
                                                     axis=0)


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

    global arnold_tongue, coherence

    # Initialize the Arnold tongue
    arnold_tongue = np.zeros((experiment_parameters['num_blocks'],
                              experiment_parameters['num_conditions']))
    arnold_tongue = Array('d', arnold_tongue.reshape(-1))

    # Initialize the coherence
    coherence = np.zeros(
        (experiment_parameters['num_blocks'],
         experiment_parameters['num_conditions'], num_entries))
    coherence = Array('d', coherence.reshape(-1))

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

    coherence = np.array(coherence).reshape(
        experiment_parameters['num_blocks'],
        experiment_parameters['num_conditions'], num_entries)

    return arnold_tongue, coherence


def simulation_grid(num_effective_learning_rate, effective_learning_rates,
                    experiment_parameters, simulation_parameters, num_entries,
                    stimulus_conditions, simulation_classes, indexing,
                    weighted_coherence, behavioral_arnold_tongue):
    """
    Run a grid of simulations.

    Parameters
    ----------
    num_effective_learning_rate : int
        The number of effective learning rates.
    effective_learning_rates : array_like
        The effective learning rates.
    experiment_parameters : dict
        The experiment parameters.
    simulation_parameters : dict
        The simulation parameters.
    num_entries : int
        The number of entries.
    stimulus_conditions : tuple
        The stimulus conditions.
    simulation_classes : tuple
        The simulation classes.
    indexing : tuple
        The indexing for synchronization.

    Returns
    -------
    array_like
        The weighted Jaccard fits.
    """

    model, stimulus_generator = simulation_classes

    weighted_jaccard_fits = np.zeros(num_effective_learning_rate - 2)
    for i, effective_learning_rate in enumerate(
            effective_learning_rates[1:-1]):
        model.effective_learning_rate = effective_learning_rate
        model.generate_coupling()
        model.update_coupling(weighted_coherence)

        simulation_classes = (model, stimulus_generator)

        arnold_tongue, _ = run_simulation(experiment_parameters,
                                          simulation_parameters, num_entries,
                                          stimulus_conditions,
                                          simulation_classes, indexing)
        weighted_jaccard_fits[i] = weighted_jaccard(arnold_tongue.mean(axis=0),
                                                    behavioral_arnold_tongue)

    return weighted_jaccard_fits


def coarse_to_fine(crossval_parameters, weighted_coherence,
                   experiment_parameters, simulation_parameters, num_entries,
                   stimulus_conditions, simulation_classes, indexing,
                   behavioral_arnold_tongue):
    """
    Estimate the effective learning rate through coarse-to-fine grid search.

    Parameters
    ----------
    crossval_parameters : dict
        The cross-validation parameters.
    weighted_coherence : array_like
        The weighted coherence matrix.
    experiment_parameters : dict
        The experiment parameters.
    simulation_parameters : dict
        The simulation parameters.
    num_entries : int
        The number of entries.
    stimulus_conditions : tuple
        The stimulus conditions.
    simulation_classes : tuple
        The simulation classes.
    indexing : tuple
        The indexing for synchronization.
    behavioral_arnold_tongue : array_like
        The behavioral Arnold tongue.
    
    Returns
    -------
    float
        The effective learning rate.
    """

    # Expand the weighted coherence matrix
    model = simulation_classes[0]
    diagonal = np.ones(model.num_populations)
    weighted_coherence = expand_matrix(weighted_coherence, diagonal)

    # Set up the grid search (initialize the effective learning rates)
    effective_learning_rates = np.linspace(
        crossval_parameters['effective_learning_rate_min'],
        crossval_parameters['effective_learning_rate_max'],
        crossval_parameters['num_effective_learning_rate'])

    # Coarse-to-fine grid search
    for _ in range(crossval_parameters['num_grids']):
        # Run a grid of simulations
        weighted_jaccard_fits = simulation_grid(
            crossval_parameters['num_effective_learning_rate'],
            effective_learning_rates, experiment_parameters,
            simulation_parameters, num_entries, stimulus_conditions,
            simulation_classes, indexing, weighted_coherence,
            behavioral_arnold_tongue)

        # Find the best learning rate
        lower_bound = np.argmax(weighted_jaccard_fits)
        upper_bound = lower_bound + 2
        best_index = lower_bound + 1
        best_learning_rate = effective_learning_rates[best_index]

        # Update the grid
        effective_learning_rates = np.linspace(
            lower_bound, upper_bound,
            crossval_parameters['num_effective_learning_rate'])

    return best_learning_rate


if __name__ == '__main__':
    # Load the model, stimulus, simulation, and experiment parameters
    model_parameters, stimulus_parameters, simulation_parameters, crossval_parameters, experiment_parameters = load_configurations(
    )
    num_entries = model_parameters['num_populations'] * (
        model_parameters['num_populations'] - 1) // 2

    # Initialize the model and stimulus generator
    simulation_classes = initialize_simulation_classes(model_parameters,
                                                       stimulus_parameters)

    # Set up parallel processing
    simulation_parameters, experiment_parameters = setup_parallel_processing(
        simulation_parameters, experiment_parameters)

    # Set up the stimulus conditions
    stimulus_conditions = generate_stimulus_conditions(experiment_parameters)

    # Set up the synchronization index and timepoint
    indexing = generate_time_index(simulation_parameters)

    # Run simulation of session 1
    arnold_tongue, coherence = run_simulation(experiment_parameters,
                                              simulation_parameters,
                                              num_entries, stimulus_conditions,
                                              simulation_classes, indexing)

    # load behavioral Arnold tongues of session 1
    session1_arnold_tongues = np.load(
        'results/empirical/session_1/individual_bats.npy')

    # load behavioral Arnold tongues of session 2
    session2_arnold_tongues = np.load(
        'results/empirical/session_2/individual_bats.npy')

    # create predictors for the psychometric function
    predictors = np.ones((2, experiment_parameters['num_conditions']))
    predictors[0] = arnold_tongue.mean(axis=0)

    optimal_psychometric_crossval = np.zeros(
        (experiment_parameters['num_subjects'], 2))
    weighted_coherence_crossval = np.zeros(
        (experiment_parameters['num_subjects'], num_entries))
    learning_rate_crossval = np.zeros(experiment_parameters['num_subjects'])

    # Cross-validation
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
        optimized_parameters, _ = curve_fit(psychometric_function,
                                            predictors,
                                            average_arnold_tongue.flatten(),
                                            p0=initial_params)
        optimal_psychometric_crossval[subject] = optimized_parameters

        # Estimate weighted coherence from session 1 data
        weighted_coherence = compute_weighted_coherence(
            experiment_parameters['num_conditions'],
            experiment_parameters['num_blocks'], num_entries, arnold_tongue,
            coherence, optimized_parameters)

        weighted_coherence_crossval[subject] = weighted_coherence

        # remove subject from behavioral Arnold tongues of session 2
        fold_arnold_tongues = np.delete(session2_arnold_tongues,
                                        subject,
                                        axis=0)

        # Compute average behavioral Arnold tongue of session 2
        average_arnold_tongue = fold_arnold_tongues.mean(axis=0)

        learning_rate_crossval[subject] = coarse_to_fine(
            crossval_parameters, weighted_coherence, experiment_parameters,
            simulation_parameters, num_entries, stimulus_conditions,
            simulation_classes, indexing, average_arnold_tongue)

    # Save results
    np.savez('results/simulation/crossval_estimation.npz',
             optimal_psychometric_crossval=optimal_psychometric_crossval,
             weighted_coherence_crossval=weighted_coherence_crossval,
             learning_rate_crossval=learning_rate_crossval)
