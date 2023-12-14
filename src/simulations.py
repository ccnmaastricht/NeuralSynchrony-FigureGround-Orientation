import numpy as np
from copy import copy
from scipy.integrate import odeint

from sim_utils import gaussian, threshold_linear, inverse_complex_log_transform, pairwise_distance, create_annulus

class V1Model:
    def __init__(self, model_parameters, stimulus_parameters):
        self.X = None
        self.Y = None
        self.omega = None
        self.decay_rate = model_parameters['decay_rate']
        self.max_coupling = model_parameters['max_coupling']
        self.num_populations = model_parameters['num_populations']
        self.contrast_slope = model_parameters['contrast_slope']
        self.contrast_intercept = model_parameters['contrast_intercept']
        
        self._generate_receptive_fields(stimulus_parameters)
        self._generate_couplinggenerate_coupling()

    def compute_omega(self, stimulus):
        """
        Compute intrinsic frequencies based on the stimulus.

        Parameters
        ----------
        stimulus : array_like
            The stimulus.
        """
        mean_luminance = np.mean(stimulus)
        normalized_luminance = (stimulus - mean_luminance)**2 / (mean_luminance**2)
        contrast = np.sqrt( np.matmul(self.receptive_fields, normalized_luminance) ) * 100
        frequency = self.contrast_slope * contrast + self.contrast_intercept
        self.omega = 2 * np.pi * frequency
        
    def update_coupling(self, coupling):
        """
        Update the coupling matrix.

        Parameters
        ----------
        coupling : array_like
            The new coupling matrix.
        """
        self.coupling = coupling

    def simulate(self, parameters):
        """
        Simulate the model for the given initial state and time.

        Parameters
        ----------
        parameters : dict
            The parameters for the simulation.
            - time_step : float
                The time step of the simulation.
            - simulation_time : float
                The total simulation time.
            - initial_state : array_like
                The initial state of the system.

        Returns
        -------
        time_vector : array_like
            The time points at which the state was evaluated.
        state : array_like
            The state of the system at each time point.
        """
        time_step = parameters['time_step']
        simulation_time = parameters['simulation_time']
        initial_state = parameters['initial_state']
        

        time_vector = np.arange(0, simulation_time, time_step)
        state = odeint(self._dynamics, initial_state, time_vector)
        return state, time_vector
    
    def _generate_receptive_fields(self, parameters):
        """
        Generate receptive fields.

        Parameters
        ----------
        parameters : dict
            The parameters of the stimulus.
            - stimulus_num_pixels : int
                The number of pixels in the stimulus.
            - stimulus_eccentricity : float
                The stimulus_eccentricity of the stimulus center.
            - stimulus_side_length : float
                The side length of the stimulus.
        """
        stimulus_num_pixels = parameters['stimulus_num_pixels']
        stimulus_eccentricity = parameters['stimulus_eccentricity']
        stimulus_side_length = parameters['stimulus_side_length']
        
        lower_bound = stimulus_eccentricity - stimulus_side_length / 2
        upper_bound = stimulus_eccentricity + stimulus_side_length / 2
        r = np.linspace(lower_bound, upper_bound, int(np.sqrt(self.num_populations)))
        X, Y = np.meshgrid(r, r)
        self.X = X.flatten()
        self.Y = Y[::-1].flatten()
 
        r = np.linspace(lower_bound, upper_bound, int(np.sqrt(stimulus_num_pixels)))
        X, Y = np.meshgrid(r, r)
        X = X.flatten()
        Y = Y[::-1].flatten()
        
        eccentricity = np.sqrt(self.X**2 + self.Y**2)

        diameter = threshold_linear(eccentricity, slope=0.172, intercept=0.25, offset=1)
        sigma = diameter / 4

        self.receptive_fields = np.zeros((self.num_populations, stimulus_num_pixels))
        for i in range(self.num_populations):
            rf = gaussian(X, Y, self.X[i], self.Y[i], sigma[i])
            self.receptive_fields[i, :] = rf / np.sum(rf)    

    def _generate_coupling(self):
        """
        Generate the coupling matrix.
        """
        X_cortex, Y_cortex = inverse_complex_log_transform(self.X, self.Y)
        distances = pairwise_distance(X_cortex, Y_cortex)
        self.coupling = np.exp(-self.decay_rate * distances) * self.max_coupling
    
    def _dynamics(self, state, t):
        """
        The dynamics of the Kuramoto model.

        Parameters
        ----------
        state : array_like
            The current state of the system.
        t : float
            The current time.

        Returns
        -------
        dtheta : array_like
            The change of the state based on the Kuramoto model.
        """
        theta = state.reshape((-1, 1))
        phase_difference = theta.T - theta
        dtheta = self.omega + 1 / self.num_populations * np.sum(self.coupling * np.sin(phase_difference), axis=1)
        return dtheta
    

class StimulusGenerator():
    def __init__(self, parameters):
        self.resolution = parameters['resolution']
        annulus_diameter = parameters['annulus_diameter']
        annulus_frequency = parameters['annulus_frequency']
        self.annulus_resolution = parameters['annulus_resolution']

        self.annulus = create_annulus(annulus_diameter, annulus_frequency, self.annulus_resolution)

    def generate(self, scaling_factor, contrast_range, mean_contrast):
        """
        Generate a stimulus.

        Parameters
        ----------
        scaling_factor : float
            The scaling factor for the grid.
        contrast_range : float
            The range of the contrast.
        mean_contrast : float
            The mean contrast.

        Returns
        -------
        array_like
            The generated stimulus.
        """
        grid = self._get_grid(scaling_factor)
        stimulus = np.ones((self.resolution, self.resolution)) * 0.5
        indices = np.arange(self.annulus_resolution)
        annulus_half_res = self.annulus_resolution // 2
        for row, col in grid:
            left, right = row - annulus_half_res, row + annulus_half_res
            down, up = col - annulus_half_res, col + annulus_half_res

            lower_row, upper_row = np.clip([left, right], 0, self.resolution)
            lower_col, upper_col = np.clip([down, up], 0, self.resolution)

            range_row = upper_row - lower_row
            range_col = upper_col - lower_col

            if left<0:
                row_indices = indices[-range_row:]
            else:
                row_indices = indices[:range_row]

            if down<0:
                col_indices = indices[-range_col:]
            else:
                col_indices = indices[:range_col]

            contrast_factor = np.random.uniform(mean_contrast - contrast_range/2, mean_contrast + contrast_range/2)

            stimulus[lower_row:upper_row, lower_col:upper_col] = self.annulus[row_indices, :][:, col_indices] * contrast_factor + 0.5
        
        return stimulus

    def _get_grid(self, scaling_factor):
        """
        Generate a grid with a specified scaling factor.

        Parameters
        ----------
        scaling_factor : float
            The scaling factor for the grid.

        Returns
        -------
        array_like
            The generated grid.
        """
        step_size = int(self.annulus_resolution * scaling_factor) + 1
        grid_points = np.arange(self.annulus_resolution//2, self.resolution, step_size)
        row_grid, col_grid = np.meshgrid(grid_points, grid_points)
        grid = np.vstack((row_grid.flatten(), col_grid.flatten())).T

        randomness = (self.annulus_resolution * scaling_factor - self.annulus_resolution) // 2
        if randomness > 0:
            grid += np.random.randint(-randomness, randomness, size=grid.shape)

        return grid