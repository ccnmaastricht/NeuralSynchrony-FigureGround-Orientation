import numpy as np
from scipy import ndimage


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


def compute_angular_differences(preferred_orientations):
    """Compute the angular differences between neurons.

    Parameters
    ----------
    preferred_orientations : array_like
        Preferred orientations of the neurons.
    Returns
    -------
    array_like
        The angular differences between neurons.
    """
    angular_differences = preferred_orientations[:, np.
                                                 newaxis] - preferred_orientations[
                                                     np.newaxis, :]
    return np.angle(np.exp(2j * angular_differences)) / 2


def weighted_kuramoto(orientation_map, receptive_fields):
    """
    Compute the Kuramoto order parameter of the orientation map weighted by the receptive fields of each neural population.

    Parameters
    ----------
    orientation_map : np.ndarray
        The orientation map.

    receptive_fields : np.ndarray
        The receptive fields of all neural populations.

    Returns
    -------
    np.ndarray
        The weighted Kuramoto order parameter.
    """
    orientation_map = orientation_map.flatten()

    # Compute the complex representation of the orientation
    complex_orientation = np.exp(1j * orientation_map)

    # Compute the weighted Kuramoto order parameter
    order_parameter = receptive_fields.dot(complex_orientation)

    return np.angle(order_parameter)


def orientation_response(preferred_orientations,
                         input_orientations,
                         sigma=np.pi / 2):
    """
    Compute the orientation response using a Gaussian tuning curve.

    Parameters
    ----------
    preferred_orientatiosn : np.ndarray
                The preferred orientation of the unit.
        input_orientation : np.ndarray
            The input orientation for the unit.
        sigma : float, optional
            The tuning width of the orientation response. Default is pi/2.

        Returns
        -------
        response : np.ndarray
            The orientation response of the unit.
        """
    angular_distance = np.angle(
        np.exp((preferred_orientations - input_orientations) * 2j))
    return np.exp(-angular_distance**2 / (8 * sigma**2))


def convert_frequencies(parameters):
    """
    Convert the orientation frequency to cycles per pixel.
    """
    return parameters['orientation_frequency'] / parameters['patch_resolution']


def compute_sigma(frequency, bandwidth=1):
    """
    Compute the standard deviation of the Gaussian envelope for the Gabor filter bank.
    """
    bandwidth_factor = (2**bandwidth + 1) / (2**bandwidth - 1)
    return bandwidth_factor * np.sqrt(np.log(2)) / (frequency * np.pi)


def gabor_filter_bank(image, frequency, orientations, sigma=1.0):
    """
    Apply a Gabor filter bank to an image to extract orientation information.

    Parameters:
    - image: 2D numpy array representing the grayscale image.
    - frequency: Wavelength of the sinusoidal factor.
    - orientations: List of orientations (theta) in radians.
    - sigma: Standard deviation of the Gaussian envelope.

    Returns:
    - orientation_map: 2D numpy array with the estimated orientation at each pixel.
    - max_response: 2D numpy array with the maximum filter response at each pixel.
    """
    rows, cols = image.shape
    response_stack = np.zeros((len(orientations), rows, cols))

    for i, theta in enumerate(orientations):
        # Create Gabor filter kernel
        kernel = gabor_kernel(frequency, theta, sigma_x=sigma, sigma_y=sigma)
        # Apply filter to image
        filtered = ndimage.convolve(image, np.real(kernel), mode='reflect')
        # Compute magnitude of response
        response = np.abs(filtered)
        # Accumulate maximum response over frequencies
        response_stack[i] = np.maximum(response_stack[i], response)

    # Determine the orientation with the maximum response at each pixel
    orientation_indices = np.argmax(response_stack, axis=0)
    orientation_map = orientations[orientation_indices]

    return orientation_map


def gabor_kernel(frequency, theta, sigma_x, sigma_y):
    """
    Generate a Gabor filter kernel.

    Parameters:
    - frequency: Wavelength of the sinusoidal factor.
    - theta: Orientation angle in radians.
    - sigma_x, sigma_y: Standard deviations of the Gaussian envelope.

    Returns:
    - kernel: 2D numpy array representing the Gabor kernel.
    """
    # Define kernel size
    nstds = 3  # Number of standard deviations to include in the kernel size
    xmax = max(abs(nstds * sigma_x * np.cos(theta)),
               abs(nstds * sigma_y * np.sin(theta)))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)),
               abs(nstds * sigma_y * np.cos(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = np.ceil(max(1, ymax))
    x = np.linspace(-xmax, xmax, int(2 * xmax + 1))
    y = np.linspace(-ymax, ymax, int(2 * ymax + 1))
    x, y = np.meshgrid(x, y)

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # Gabor kernel
    kernel = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * \
             np.exp(1j * (2 * np.pi * frequency * x_theta))

    return kernel
