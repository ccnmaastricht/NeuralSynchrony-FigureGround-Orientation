import numpy as np


def gaussian(X, Y, x, y, sigma):
    '''
    Isotropic 2D Gaussian function.

    Parameters
    ----------
    X : array_like
        X coordinate space.
    Y : array_like
        Y coordinate space.
    x : float
        The x-coordinate of the Gaussian.
    y : float
        The y-coordinate of the Gaussian.
    sigma : float
        The standard deviation of the Gaussian.

    Returns
    ------- 
    float
        The value of the Gaussian at the given coordinates.
    '''
    return np.exp(-((X - x)**2 + (Y - y)**2) / (2 * sigma**2))

def threshold_linear(x, slope, intercept, offset):
    '''
    Threshold linear function.

    Parameters
    ----------
    x : float
        The input value.
    slope : float
        The slope of the linear function.
    intercept : float
        The intercept of the linear function.
    offset : float
        The offset of the linear function.

    Returns
    ------- 
    float
        The value of the function at the given input.
    '''
    return np.maximum(slope * x + intercept, offset)