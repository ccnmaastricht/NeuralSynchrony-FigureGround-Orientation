import os
import tomllib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.anl_utils import compute_size
from src.sim_utils import generate_condition_space

import matplotlib.pyplot as plt


def compute_empirical_sizes(experiment_parameters, condition_space):
    """
    Get the empirical data.

    Returns
    -------
    data : pandas.DataFrame
        The empirical data.
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


if __name__ == '__main__':

    # load configuration
    with open('config/analysis/experiment_actual.toml', 'rb') as f:
        experiment_parameters = tomllib.load(f)

    # Load model learning data
    data = np.load('results/simulation/learning_simulation.npz')
    model_sizes = data['arnold_tongue_size']

    # Get empirical learning data
    condition_space = generate_condition_space(experiment_parameters)
    empirical_sizes = compute_empirical_sizes(experiment_parameters,
                                              condition_space)

    # Stack the arrays
    data = np.hstack((model_sizes.reshape(-1,
                                          1), empirical_sizes.reshape(-1, 1)))

    # Create subject and session indices
    n_subjects = experiment_parameters['num_subjects']
    n_sessions = experiment_parameters['num_training_sessions']
    subject_idx = np.repeat(np.arange(n_subjects), n_sessions)
    session_idx = np.tile(np.arange(n_sessions), n_subjects)

    # Create a DataFrame
    df = pd.DataFrame(data, columns=["model", "empirical"])
    df["fold"] = subject_idx
    df["session_id"] = session_idx

    # Define the model formula with fixed and random effects
    md = smf.mixedlm("empirical ~ model",
                     df,
                     groups=df["fold"],
                     re_formula="~0 + model",
                     vc_formula={"fold": "0 + C(fold)"})

    # Fit the model
    mdf = md.fit()

    print(mdf.summary())

    plt.plot(model_sizes.T, 'o')
    plt.show()
