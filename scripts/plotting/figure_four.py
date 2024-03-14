import os
import tomllib
import numpy as np

from src.anl_utils import compute_size, compute_growth_rate
from src.sim_utils import generate_condition_space

import seaborn as sns
import matplotlib.pyplot as plt
from src.plot_utils import comparative_lineplot


def compute_empirical_sizes(experiment_parameters, condition_space):
    """
    Get the empirical data.

    Returns
    -------
    bat_sizes : array_like
        The bat sizes.
    """

    base_path = 'results/empirical'

    bat_sizes = np.zeros((experiment_parameters['num_subjects'],
                          experiment_parameters['num_training_sessions']))

    for session in range(experiment_parameters['num_training_sessions']):
        session_path = os.path.join(base_path, f'session_{session + 1}')
        filename = os.path.join(session_path, 'individual_bats.npy')
        individual_bats = np.load(filename)
        for subject, bat in enumerate(individual_bats):
            bat_sizes[subject, session] = compute_size(bat, *condition_space)

    return bat_sizes


def load_data(config_path, simulation_path):
    # load configuration
    with open(config_path, 'rb') as f:
        experiment_parameters = tomllib.load(f)

    # Load model learning data
    data = np.load(simulation_path)
    model_sizes = data['arnold_tongue_size']

    # Get empirical learning data
    condition_space = generate_condition_space(experiment_parameters)
    empirical_sizes = compute_empirical_sizes(experiment_parameters,
                                              condition_space)

    return model_sizes, empirical_sizes


def main():
    # Load data
    config_path = 'config/analysis/experiment_actual.toml'
    model_path = 'results/simulation/learning_simulation.npz'
    model_sizes, empirical_sizes = load_data(config_path, model_path)

    # Compute growth rates
    model_growth = compute_growth_rate(model_sizes)
    empirical_growth = compute_growth_rate(empirical_sizes)

    # Plotting
    sessions = np.arange(1, model_sizes.shape[1] + 1)

    # Plot model sizes
    lower = np.percentile(model_sizes, 2.5, axis=0)
    upper = np.percentile(model_sizes, 97.5, axis=0)
    bounds_model = (lower, upper)

    lower = np.percentile(empirical_sizes, 2.5, axis=0)
    upper = np.percentile(empirical_sizes, 97.5, axis=0)
    bounds_empirical = (lower, upper)

    bounds = (bounds_model, bounds_empirical)

    linecolor = ('#785ef0', '#fe6100')

    y = (model_sizes.mean(axis=0), empirical_sizes.mean(axis=0))

    lineplot = comparative_lineplot(sessions,
                                    y,
                                    bounds,
                                    figsize=(10, 5),
                                    labels=('Model', 'Session',
                                            'Arnold tongue size'),
                                    fontsizes=(14, 12),
                                    line_color=linecolor)

    # save plot and close
    filename = 'results/figures/figure_four/model_sizes.svg'
    #lineplot.savefig(filename)

    plt.show()


if __name__ == "__main__":
    main()
