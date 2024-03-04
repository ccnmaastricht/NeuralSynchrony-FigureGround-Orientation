import numpy as np


class StimulusGenerator():

    def __init__(self, parameters):
        self.stimulus_resolution = parameters['stimulus_resolution']
        annulus_diameter = parameters['annulus_diameter']
        annulus_frequency = parameters['annulus_frequency']
        self.annulus_resolution = parameters['annulus_resolution']

        self.annulus = self._create_annulus(annulus_diameter,
                                            annulus_frequency)

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
        stimulus = np.ones(
            (self.stimulus_resolution, self.stimulus_resolution)) * 0.5
        indices = np.arange(self.annulus_resolution)
        annulus_half_res = self.annulus_resolution // 2
        for row, col in grid:
            left, right = row - annulus_half_res, row + annulus_half_res
            down, up = col - annulus_half_res, col + annulus_half_res

            lower_row, upper_row = np.clip([left, right], 0,
                                           self.stimulus_resolution)
            lower_col, upper_col = np.clip([down, up], 0,
                                           self.stimulus_resolution)

            range_row = upper_row - lower_row
            range_col = upper_col - lower_col

            if left < 0:
                row_indices = indices[-range_row:]
            else:
                row_indices = indices[:range_row]

            if down < 0:
                col_indices = indices[-range_col:]
            else:
                col_indices = indices[:range_col]

            contrast_factor = np.random.uniform(
                mean_contrast - contrast_range / 2,
                mean_contrast + contrast_range / 2)

            stimulus[lower_row:upper_row, lower_col:upper_col] = self.annulus[
                row_indices, :][:, col_indices] * contrast_factor + 0.5

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
        step_size = int(self.annulus_resolution * scaling_factor)
        annulus_quarter_res = self.annulus_resolution // 4
        grid_points = np.arange(annulus_quarter_res,
                                self.stimulus_resolution + step_size,
                                step_size)
        row_grid, col_grid = np.meshgrid(grid_points, grid_points)
        grid = np.vstack((row_grid.flatten(), col_grid.flatten())).T

        randomness = (self.annulus_resolution * scaling_factor -
                      self.annulus_resolution) // 2
        if randomness > 0:
            grid += np.random.randint(-randomness, randomness, size=grid.shape)

        return grid

    def _create_annulus(self, diameter, frequency):
        """
        Create a Gabor annulus.

        Parameters
        ----------
        diameter : float
            The diameter of the annulus.
        frequency : float
            The spatial frequency of the radial modulation.

        Returns
        -------
        array_like
            The annulus.
        """
        r = np.linspace(-diameter / 2, diameter / 2, self.annulus_resolution)
        X, Y = np.meshgrid(r, -r)
        radius = np.hypot(X, Y)
        mask = radius <= diameter / 2
        annulus = 0.5 * np.cos(radius * frequency * 2 * np.pi + np.pi) * mask
        return annulus
