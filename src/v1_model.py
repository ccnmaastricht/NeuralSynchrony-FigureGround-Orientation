import numpy as np
from scipy.integrate import odeint

from src.model_utils import *


class V1Model:

    def __init__(self, model_parameters, stimulus_parameters):
        self.X = None
        self.Y = None
        self.omega = None
        self.decay_rate = model_parameters['decay_rate']
        self.max_coupling = model_parameters['max_coupling']
        self.num_hypercolumns = model_parameters['num_hypercolumns']
        self.num_orientations = model_parameters['num_orientations']
        self.num_populations = self.num_hypercolumns * self.num_orientations

        self.contrast_slope = model_parameters['contrast_slope']
        self.contrast_intercept = model_parameters['contrast_intercept']
        self.orientation_slope = model_parameters['orientation_slope']
        self.receptive_field_slope = model_parameters['receptive_field_slope']
        self.receptive_field_intercept = model_parameters[
            'receptive_field_intercept']
        self.receptive_field_minimum_size = model_parameters[
            'receptive_field_minimum_size']
        self.tuning_sigma = model_parameters['tuning_sigma']

        self.coupling_sigma = model_parameters['coupling_sigma']

        self.filter_bank_frequency = convert_frequencies(stimulus_parameters)
        self.filter_bank_sigma = compute_sigma(self.filter_bank_frequency)

        self.effective_learning_rate = None

        self._generate_receptive_fields(stimulus_parameters)
        self.generate_coupling()

    def compute_omega(self, stimulus):
        """
        Compute intrinsic frequencies based on the stimulus.

        Parameters
        ----------
        stimulus : array_like
            The stimulus.
        """

        orientation_map = gabor_filter_bank(
            stimulus, self.filter_bank_frequency,
            np.unique(self.preferred_orientations), self.filter_bank_sigma)

        weighted_kuramoto_order = weighted_kuramoto(orientation_map,
                                                    self.receptive_fields)

        response = orientation_response(self.preferred_orientations,
                                        weighted_kuramoto_order,
                                        self.tuning_sigma)

        frequency = self.orientation_slope * response
        self.omega = 2 * np.pi * frequency

    def generate_coupling(self):
        """
        Generate the coupling matrix.
        """
        X_cortex, Y_cortex = inverse_complex_log_transform(self.X, self.Y)

        X_cortex = np.repeat(X_cortex[:, np.newaxis],
                             self.num_orientations,
                             axis=1).flatten()
        Y_cortex = np.repeat(Y_cortex[:, np.newaxis],
                             self.num_orientations,
                             axis=1).flatten()

        spatial_distances = pairwise_distance(X_cortex, Y_cortex)
        distance_coupling = np.exp(
            -self.decay_rate * spatial_distances) * self.max_coupling

        preferred_orientations = np.repeat(np.linspace(
            0, 180, self.num_orientations, endpoint=False)[np.newaxis, :],
                                           self.num_hypercolumns,
                                           axis=0).flatten()
        self.preferred_orientations = np.radians(preferred_orientations)

        angular_differences = compute_angular_differences(
            self.preferred_orientations)
        angular_coupling = np.exp(-angular_differences**2 /
                                  (2 * self.coupling_sigma**2))

        self.coupling = distance_coupling * angular_coupling

    def update_coupling(self, weighted_locking):
        """
        Update the coupling matrix through Hebbian learning.

        Parameters
        ----------
        weighted_locking : array_like
            The weighted locking matrix.
        """

        if self.effective_learning_rate is None:
            raise ValueError("effective_learning_rate cannot be None")

        decay_factor = np.exp(-self.effective_learning_rate)
        self.coupling = decay_factor * self.coupling + (
            1 - decay_factor) * weighted_locking * self.max_coupling

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
        initial_state = np.random.rand(self.num_populations) * np.pi

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
        xy_offset = np.sqrt(stimulus_eccentricity**2 / 2)
        lower_bound = xy_offset - stimulus_side_length / 2
        upper_bound = xy_offset + stimulus_side_length / 2
        r = np.linspace(lower_bound, upper_bound,
                        int(np.sqrt(self.num_hypercolumns)))
        X, Y = np.meshgrid(r, r)
        self.X = X.flatten()
        self.Y = Y[::-1].flatten()

        r = np.linspace(lower_bound, upper_bound,
                        int(np.sqrt(stimulus_num_pixels)))
        X, Y = np.meshgrid(r, r)
        X = X.flatten()
        Y = Y[::-1].flatten()

        eccentricity = np.sqrt(self.X**2 + self.Y**2)

        diameter = threshold_linear(eccentricity,
                                    slope=self.receptive_field_slope,
                                    intercept=self.receptive_field_intercept,
                                    offset=self.receptive_field_minimum_size)
        sigma = diameter / 4

        self.receptive_fields = np.zeros(
            (self.num_populations, stimulus_num_pixels))
        for i in range(self.num_hypercolumns):
            rf = gaussian(X, Y, self.X[i], self.Y[i], sigma[i])
            self.receptive_fields[i * self.num_orientations:(i + 1) *
                                  self.num_orientations] = rf / np.sum(rf)

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
        dtheta = self.omega + 1 / self.num_populations * np.sum(
            self.coupling * np.sin(phase_difference), axis=1)
        return dtheta
