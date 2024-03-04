import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.integrate import simps

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

def compute_coherence(theta, condense = True):
    """
    Compute the coherence of a set of phases.

    Parameters
    ----------
    theta : array_like
        The phases.
    condense : bool
        Whether to condense the matrix.

    Returns
    ------- 
    float
        The coherence.
    """
    theta = theta.T
    phase_difference = np.angle(np.exp(1j * (theta[:, None] - theta[None, :])))
    coherence = np.cos(phase_difference)
    if condense:
        coherence, _ = condense_matrix(coherence)
    return coherence 

def compute_size(arnold_tongue, grid_coarseness, contrast_heterogeneity):
    """
    Compute the size of the Arnold tongue as the volume under its surface.

    Parameters
    ----------
    arnold_tongue : array_like
        The Arnold tongue.
    grid_coarseness : array_like
        The grid coarseness.
    contrast_heterogeneity : array_like
        The contrast heterogeneity.

    Returns
    -------
    float
        The size of the Arnold tongue.
    """

    return simps(simps(arnold_tongue, contrast_heterogeneity), grid_coarseness)

    
def condense_matrix(matrix):
    """
    Condense a symmetric matrix.

    Parameters
    ----------
    coherence : array_like
        The symmetric matrix.

    Returns
    ------- 
    array_like
        The condensed matrix.
    """

    diagonal = np.diag(matrix)
    matrix = matrix - np.diag(diagonal)
    return squareform(matrix), diagonal

def expand_matrix(matrix, diagonal):
    """
    Expand a condensed matrix.

    Parameters
    ----------
    matrix : array_like
        The condensed matrix.
    diagonal : array_like
        The diagonal of the original matrix.

    Returns
    ------- 
    array_like
        The expanded matrix.
    """
    diagonal = np.diag(diagonal)
    return squareform(matrix) + diagonal

def welford_update(mean, count, new_value):
    """
    Update the mean and count of a set of values.

    Parameters
    ----------
    mean : float
        The current mean.
    count : int
        The current count.
    new_value : float
        The new value.

    Returns
    ------- 
    float
        The updated mean.
    int
        The updated count.
    """
    delta = new_value - mean
    mean += delta / count
    return mean

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

def compute_weighted_coherence(counts_tuple, measurements, optimal_psychometric_parameters):
    """
    Compute the weighted coherence.  

    Parameters
    ----------
    counts_tuple : tuple
        The number of blocks, conditions, and entries.
    measurements : array_like
        The Arnold tongue and coherence.
    optimal_psychometric_parameters : array_like
        The optimal psychometric parameters.

    Returns
    -------
    array_like
        The weighted coherence.
    """

    num_blocks, num_conditions, num_entries = counts_tuple
    arnold_tongue, coherence = measurements

    predictors = np.ones((2, num_conditions))
    weighted_coherence = np.zeros(num_entries)
    
    for block in range(num_blocks):
        predictors[0] = arnold_tongue[block]
        probability_correct = psychometric_function(predictors, *optimal_psychometric_parameters)
             
        probability_correct = np.tile(probability_correct, (num_entries, 1)).T
        weighted_coherence = welford_update(weighted_coherence, block + 1, (probability_correct * coherence[block]).mean(axis=0))

    return weighted_coherence