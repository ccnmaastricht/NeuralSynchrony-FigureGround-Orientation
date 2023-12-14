import numpy as np


def gaussian(X, Y, x, y, sigma):
    """
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
    """
    return np.exp(-((X - x)**2 + (Y - y)**2) / (2 * sigma**2))

def threshold_linear(x, slope, intercept, offset):
    """
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
    """
    return np.maximum(slope * x + intercept, offset)

def inverse_complex_log_transform(X, Y, k=15.0, a=0.7, b=80, alpha=0.9):
    """
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
    """

    eccentricity = np.abs(X + Y * 1j)
    polar_angle = np.angle(X + Y * 1j)

    Z = eccentricity * np.exp(1j * alpha * polar_angle)
    W = k * np.log((Z + a) / (Z + b)) - k * np.log(a / b)

    X = np.real(W)
    Y = np.imag(W)

    return X, Y

def pairwise_distance(X, Y):
    """
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
    """

    return np.sqrt((X[:, None] - X[None, :])**2 + (Y[:, None] - Y[None, :])**2)

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


def create_annulus(diameter, frequency, resolution):
    """
    Create a Gabor annulus.

    Parameters
    ----------
    diameter : float
        The diameter of the annulus.
    frequency : float
        The spatial frequency of the radial modulation.
    resolution : int
        The resolution of the annulus.

    Returns
    -------
    array_like
        The annulus.
    """
    r = np.linspace(-diameter/2, diameter/2, resolution)
    X, Y = np.meshgrid(r, -r)
    radius = np.hypot(X, Y)
    mask = radius <= diameter/2
    annulus = 0.5 * np.cos(radius * frequency * 2 * np.pi + np.pi) * mask
    return annulus

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
    bounds = np.array([np.floor(desired / num_cores), np.ceil(desired / num_cores)]) * num_cores
    index = np.argmin(np.abs(bounds - desired))
    return int(bounds[index])

