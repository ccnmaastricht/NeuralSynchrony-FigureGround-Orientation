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

def inverse_complex_log_transform(X, Y, k=15.0, a=0.7, b=80, alpha=0.9):
    '''
    Inverse of the complex-logarithm transformation described in Schwartz
    (1980) - doi:10.1016/0042-6989(80)90090-5.

    Parameters
    ----------
    X : array_like
        X coordinate in visual field.
    Y : array_like
        Y coordinate in visual field.

    Returns
    ------- 
    X : array_like
        X coordinate in cortical space.
    Y : array_like
        Y coordinate in cortical space.
    '''

    eccentricity = np.abs(X + Y * 1j)
    polar_angle = np.angle(X + Y * 1j)

    Z = eccentricity * np.exp(1j * alpha * polar_angle)
    W = k * np.log((Z + a) / (Z + b)) - k * np.log(a / b)

    X = np.real(W)
    Y = np.imag(W)

    return X, Y

def pairwise_distance(X, Y):
    '''
    does not yet give the correct result!
    Compute the pairwise distance between all pairs of points.

    Parameters
    ----------
    X : array_like (1d array of all x-coordinates)
        X coordinate space.
    Y : array_like (1d array of all y-coordinates)
        Y coordinate space.

    Returns
    ------- 
    array_like
        The pairwise distance between all pairs of points.
    '''

    return np.sqrt((X[:, None] - X[None, :])**2 + (Y[:, None] - Y[None, :])**2)

def order_parameter(theta):
    '''
    Compute the order parameter of a set of phases.

    Parameters
    ----------
    theta : array_like
        The phases.

    Returns
    ------- 
    float (complex)
        The order parameter.
    '''

    return np.mean(np.exp(1j * theta), axis=1)

