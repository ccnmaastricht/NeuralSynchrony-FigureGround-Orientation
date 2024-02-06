import os
import tomllib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.anl_utils import get_session_data, get_subject_data


def load_configuration():
    """
    Load parameters for the model comparison procedure.

    Returns
    -------
    model_comparison_parameters : dict
        The model comparison parameters.
    """
    with open('config/analysis/model_comparison.toml', 'rb') as f:
        model_comparison_parameters = tomllib.load(f)

    return model_comparison_parameters

if __name__ == '__main__':
    # Load model comparison parameters
    model_comparison_parameters = load_configuration()

    transfer_session = model_comparison_parameters['transfer_session']
    num_training_sessions = model_comparison_parameters['num_training_sessions']
    num_predictors = model_comparison_parameters['num_predictors']

    # Load experimental data

    data_path = 'data/main.csv'

    # Load experimental data
    try:
        experimental_data = pd.read_csv(data_path)
    except FileNotFoundError:
            print(f"Data file not found: {data_path}")
            exit(1)

    transfer_data = get_session_data(experimental_data, transfer_session)

    likelihoods = np.ones(num_training_sessions, dtype=np.float128)

    # loop through the sessions
    for session in range(num_training_sessions):
        # load the training data
        train_data = get_session_data(experimental_data, session+1)

        # loop through the subjects
        for subject in train_data['SubjectID'].unique():
            # get the training and test data for the current subject
            subject_train = get_subject_data(train_data, subject)
            subject_test = get_subject_data(transfer_data, subject)

            # fit a logistic regression model
            model = smf.logit('Correct ~ GridCoarseness + ContrastHeterogeneity', subject_train)
            result = model.fit(disp=False)

            # calculate the likelihood of the test data given the model
            correct_answers = subject_test['Correct'].values
            predicted_probabilities = result.predict(subject_test)
            likelihoods[session] *= np.prod(np.power(predicted_probabilities, correct_answers) * np.power(1 - predicted_probabilities, 1 - correct_answers))

    # calculate the Akaike Information Criterion (AIC)
    AIC = 2 * num_predictors - 2 * np.log(likelihoods)

    # calculate the relative likelihood and weights of each model
    min_AIC = np.min(AIC)
    delta_AIC = AIC - min_AIC
    relative_likelihood = np.exp(-0.5 * delta_AIC)
    weights = relative_likelihood / np.sum(relative_likelihood)

    print(f'Weights: {weights}')

