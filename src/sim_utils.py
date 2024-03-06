import numpy as np

from multiprocessing import cpu_count

from src.v1_model import V1Model
from src.stimulus_generator import StimulusGenerator


def get_num_blocks(desired, num_cores):
    """
    Get the number of blocks for parallel processing.

    Parameters
    ----------
    desired : int
        The desired number of blocks.
    num_cores : int
        The number of cores.

    Returns
    -------
    int
        The number of blocks.
    """
    bounds = np.array(
        [np.floor(desired / num_cores),
         np.ceil(desired / num_cores)]) * num_cores
    index = np.argmin(np.abs(bounds - desired))
    return int(bounds[index])


def initialize_simulation_classes(model_parameters, stimulus_parameters):
    """
    Initialize the model and stimulus generator.

    Parameters
    ----------
    model_parameters : dict
        The parameters for the V1Model.
    stimulus_parameters : dict
        The parameters for the StimulusGenerator.

    Returns
    -------
    tuple
        A tuple containing the initialized V1Model and StimulusGenerator.
    """
    model = V1Model(model_parameters, stimulus_parameters)
    stimulus_generator = StimulusGenerator(stimulus_parameters)
    simulation_classes = (model, stimulus_generator)

    return simulation_classes


def setup_parallel_processing(simulation_parameters, experiment_parameters):
    """
    Set up parameters for parallel processing.

    Parameters
    ----------
    simulation_parameters : dict
        The parameters for the simulation.
    experiment_parameters : dict
        The parameters for the experiment.

    Returns
    -------
    tuple
        A tuple containing the updated simulation and experiment parameters.
    """
    simulation_parameters['num_time_steps'] = int(
        simulation_parameters['simulation_time'] /
        simulation_parameters['time_step'])

    num_cores = min(cpu_count(), simulation_parameters['num_cores'])

    num_blocks = get_num_blocks(experiment_parameters['num_blocks'], num_cores)
    num_batches = num_blocks // num_cores

    experiment_parameters.update({'num_blocks': num_blocks})
    simulation_parameters.update({
        'num_cores': num_cores,
        'num_batches': num_batches
    })

    return simulation_parameters, experiment_parameters


def generate_stimulus_conditions(experiment_parameters):
    """
    Generate stimulus conditions based on the experiment parameters.

    Parameters
    ----------
    experiment_parameters : dict
        The parameters for the experiment.

    Returns
    -------
    tuple
        A tuple containing the grid coarseness and contrast heterogeneity arrays.
    """
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

    return grid_coarseness, contrast_heterogeneity


def generate_condition_space(experiment_parameters):
    """
    Generate the condition space based on the experiment parameters.

    Parameters
    ----------
    experiment_parameters : dict
        The parameters for the experiment.

    Returns
    -------
    tuple
        A tuple containing the grid coarseness and contrast heterogeneity arrays.
    """
    contrast_heterogeneity = np.linspace(
        experiment_parameters['min_contrast_heterogeneity'],
        experiment_parameters['max_contrast_heterogeneity'],
        experiment_parameters['num_contrast_heterogeneity'])
    grid_coarseness = np.linspace(experiment_parameters['min_grid_coarseness'],
                                  experiment_parameters['max_grid_coarseness'],
                                  experiment_parameters['num_grid_coarseness'])

    return grid_coarseness, contrast_heterogeneity


def generate_time_index(simulation_parameters):
    """
    Generate indices for synchronization and timepoints based on the simulation parameters.

    Parameters
    ----------
    simulation_parameters : dict
        The parameters for the simulation.

    Returns
    -------
    tuple
        A tuple containing the sync_index and timepoints.
    """
    start = simulation_parameters['num_time_steps'] // 2
    sync_index = slice(start, None)

    start = simulation_parameters['num_time_steps'] - simulation_parameters[
        'num_timepoints']
    timepoints = range(start, simulation_parameters['num_time_steps'])

    return sync_index, timepoints
