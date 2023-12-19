import numpy as np

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

def coherence(theta):
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
    phase_difference = np.angle(np.exp(1j * (theta[:, None] - theta[None, :])))
    return np.cos(phase_difference)

def psychometric_function(predictors, slopes, intercept, chance_level = 0.5):
    """
    Compute an n-dimensional psychometric function.

    Parameters
    ----------
    predictors : array_like
        The predictors.
    slope : array_like
        The slopes.
    intercept : float
        The intercept.
    chance_level : float
        The chance level.

    Returns
    -------
    float
        The probability of a correct response.
    """
    logit = slopes * predictors + intercept
    probability = (1 - chance_level) / (1 + np.exp(-logit)) + chance_level
    return probability