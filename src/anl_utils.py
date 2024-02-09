import numpy as np
import pandas as pd

def load_data(path):
    """
    Load the behavioral data.

    Returns
    -------
    data : pandas.DataFrame
        The behavioral data.
    """
    data = pd.read_csv(path)
    return data

def get_session_data(data, session):
    """
    Get the data of a specific session.

    Parameters
    ----------
    data : pandas.DataFrame
        The behavioral data.
    session : int
        The session number.

    Returns
    -------
    session_data : pandas.DataFrame
        The data of the session.
    """
    session_data = data[data['SessionID'] == session]
    return session_data

def get_subject_data(data, subject):
    """
    Get the data of a specific subject.

    Parameters
    ----------
    data : pandas.DataFrame
        The behavioral data.
    subject : int
        The subject number.

    Returns
    -------
    subject_data : pandas.DataFrame
        The data of the subject.
    """
    subject_data = data[data['SubjectID'] == subject]
    return subject_data

def order_parameter(theta):
    """
    Compute the order parameter of a set of phases.

    Parameters
    ----------
    theta : array_like
        The phases.

    Returns
    ------- 
    float (complex)
        The order parameter.
    """

    return np.mean(np.exp(1j * theta), axis=1)

def compute_coherence(theta):
    """
    Compute the coherence of a set of phases.

    Parameters
    ----------
    theta : array_like
        The phases.

    Returns
    ------- 
    float
        The coherence.
    """
    theta = theta.T
    phase_difference = np.angle(np.exp(1j * (theta[:, None] - theta[None, :])))
    return np.cos(phase_difference)

def psychometric_function(predictors, *parameters, chance_level = 0.5):
    """
    Compute an n-dimensional psychometric function.

    Parameters
    ----------
    predictors : array_like
        The predictors.
    parameters : array_like
        The parameters.
    chance_level : float
        The chance level.

    Returns
    -------
    float
        The probability of a correct response.
    """
    logit = np.dot(parameters, predictors)
    probability = (1 - chance_level) / (1 + np.exp(-logit)) + chance_level
    return probability

def weighted_jaccard(X, Y):
    """
    Compute weighted Jaccard similarity between two sets of elements.

    Parameters
    ----------
    X : array_like
        The first set of elements.
    Y : array_like
        The second set of elements.

    Returns
    -------
    float
        The weighted Jaccard similarity.
    """
    minimum = np.minimum(X.flatten(), Y.flatten())
    maximum = np.maximum(X.flatten(), Y.flatten())

    numerator = np.sum(minimum)
    denominator = np.sum(maximum)

    return numerator / denominator

def min_max_normalize(X):
    """
    Normalize an array between 0 and 1.

    Parameters
    ----------
    X : array_like
        The array to normalize.

    Returns
    -------
    array_like
        The normalized array.
    """
    return (X - np.min(X)) / (np.max(X) - np.min(X))