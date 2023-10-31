import numpy as np
from scipy.integrate import odeint

from utils import gaussian, threshold_linear

class V1Model:
    def __init__(self, parameters):
        self.omega = parameters['omega']
        self.coupling = parameters['coupling']
        self.lambda_ = parameters['lambda']
        self.gamma = parameters['gamma']
        self.num_populations = parameters['num_populations']
        self.side_length = parameters['side_length']
        self.eccentricity = parameters['eccentricity']
        self.rf_parameters = parameters['rf_parameters']
        
    def update_coupling(self, coupling):
        """
        Update the coupling matrix.

        Parameters
        ----------
        coupling : array_like
            The new coupling matrix.
        """
        self.coupling = coupling

    def simulate(self, initial_state, simulation_time, timestep = 1e-3):
        """
        Simulate the model for the given initial state and time.

        Parameters
        ----------
        initial_state : array_like
            The initial state of the system.
        simulation_time : float
            The time to simulate the model for.

        Returns
        -------
        time_vector : array_like
            The time points at which the state was evaluated.
        state : array_like
            The state of the system at each time point.
        """

        time_vector = np.arange(0, simulation_time, self.timestep)
        state = odeint(self._dynamics, initial_state, time_vector)
        return time_vector, state

    def generate_receptive_fields(self, num_pixels):
        """
        Generate the receptive field parameters (coordinates and size) as well as actual receptive fields.

        Parameters
        ----------
        num_pixels : int
            The number of pixels in the stimulus.
        """
        lower_bound = self.eccentricity - self.side_length / 2
        upper_bound = self.eccentricity + self.side_length / 2
        r = np.linspace(lower_bound, upper_bound, self.num_populations)
        X, Y = np.meshgrid(r, r)
        X = X.flatten()
        Y = Y.flatten()
        
        self.rf_coordinates = np.array([X, Y]).T
        eccentricity = np.linalg.norm(self.rf_coordinates, axis=1)

        self.sigma = threshold_linear(eccentricity, *self.rf_parameters) * 0.25

        self.receptive_fields = np.zeros((self.num_populations, num_pixels))
        for i in range(self.num_populations):
            rf = gaussian(X, Y, self.rf_coordinates[i, 0], self.rf_coordinates[i, 1], self.sigma[i])
            self.receptive_fields[i, :] = rf / np.sum(rf)
    
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
        phase_diff = theta.T - theta
        dtheta = self.omega + np.sum(self.coupling * np.sin(phase_diff), axis=1)
        return dtheta
    
    def _generate_coupling(self, N):
        pass


        
