import numpy as np
from scipy.integrate import odeint

from utils import gaussian, threshold_linear, inverse_complex_log_transform, pairwise_distance

class V1Model:
    def __init__(self, model_parameters, stimulus_parameters, rf_parameters):
        self.omega = model_parameters['omega']
        self.decay_rate = model_parameters['lambda']
        self.max_coupling = model_parameters['gamma']
        self.num_populations = model_parameters['num_populations']
        
        self._generate_receptive_fields(stimulus_parameters, rf_parameters)
        
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
        return time_vector, state

    def _generate_receptive_fields(self, stimulus_parameters, rf_parameters):
        """
        Generate receptive fields.

        Parameters
        ----------
        stimulus_parameters : dict
            The parameters of the stimulus.
            - num_pixels : int
                The number of pixels in the stimulus.
            - radius : float
                The radius of the stimulus.
            - side_length : float
                The side length of the stimulus.
        rf_parameters : array_like
            The parameters of the receptive field function.
            - slope : float
                The slope of the threshold linear function.
            - intercept : float
                The intercept of the threshold linear function.
            - offset : float
                The offset of the threshold linear function.
        """

        num_pixels = stimulus_parameters['num_pixels']
        radius = stimulus_parameters['radius']
        side_length = stimulus_parameters['side_length']
        
        lower_bound = radius - side_length / 2
        upper_bound = radius + side_length / 2
        r = np.linspace(lower_bound, upper_bound, self.num_populations)
        X, Y = np.meshgrid(r, r)
        self.X = X.flatten()
        self.Y = Y[::-1].flatten()
        
        eccentricity = np.sqrt(X**2 + Y**2)

        sigma = threshold_linear(eccentricity, *rf_parameters) * 0.25

        self.receptive_fields = np.zeros((self.num_populations, num_pixels))
        for i in range(self.num_populations):
            rf = gaussian(X, Y, X[i], Y[i], sigma[i])
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
        phase_diff = theta.T - theta
        dtheta = self.omega + np.sum(self.coupling * np.sin(phase_diff), axis=1)
        return dtheta


        
