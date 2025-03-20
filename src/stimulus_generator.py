import numpy as np


class StimulusGenerator:

    def __init__(self, parameters):
        self.stimulus_resolution = parameters['stimulus_resolution']
        self.patch_type = parameters.get(
            'patch_type',
            'orientation')  # Patch type: 'annulus' or 'orientation'
        self.patch_resolution = parameters['patch_resolution']

        if self.patch_type == 'annulus':
            self.annulus_diameter = parameters['annulus_diameter']
            annulus_frequency = parameters['annulus_frequency']
            self.annulus = self._create_annulus(self.annulus_diameter,
                                                annulus_frequency)
        elif self.patch_type == 'orientation':
            self.orientation_diameter = parameters['orientation_diameter']
            self.reference_orientation = parameters['reference_orientation']
            self.heterogeneity = parameters['heterogeneity']
            self.orientation_frequency = parameters['orientation_frequency']

    def generate(self, scaling_factor):
        """
        Generate a stimulus.

        Parameters
        ----------
        scaling_factor : float
            The scaling factor for the grid.

        Returns
        -------
        array_like
            The generated stimulus.
        """
        grid = self._get_grid(scaling_factor)
        stimulus_grid = np.ones(
            (self.stimulus_resolution, self.stimulus_resolution)) * 0.5
        indices = np.arange(self.patch_resolution)
        patch_half_res = self.patch_resolution // 2
        for row, col in grid:
            left, right = row - patch_half_res, row + patch_half_res
            down, up = col - patch_half_res, col + patch_half_res

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

            if self.patch_type == 'annulus':
                contrast_factor = np.random.uniform(0.5 - 0.2, 0.5 + 0.2)
                patch = self.annulus[
                    row_indices, :][:, col_indices] * contrast_factor
            elif self.patch_type == 'orientation':
                orientation_factor = np.random.uniform(
                    self.reference_orientation - 90 * self.
                    heterogeneity,  #value of 90 taken from oriented_annulus code
                    self.reference_orientation + 90 * self.heterogeneity)
                patch = self._create_orientation_grating(
                    orientation_factor, self.orientation_diameter,
                    self.orientation_frequency)[row_indices, :][:, col_indices]
            else:
                patch = np.zeros(
                    (range_row, range_col))  # Ensures patch is always assigned

            stimulus_grid[lower_row:upper_row,
                          lower_col:upper_col] = patch + 0.5

        return stimulus_grid

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
        step_size = int(self.patch_resolution * scaling_factor)
        patch_quarter_res = self.patch_resolution // 4
        grid_points = np.arange(patch_quarter_res,
                                self.stimulus_resolution + step_size,
                                step_size)
        row_grid, col_grid = np.meshgrid(grid_points, grid_points)
        grid = np.vstack((row_grid.flatten(), col_grid.flatten())).T

        randomness = (self.patch_resolution * scaling_factor -
                      self.patch_resolution) // 2
        if randomness > 0:
            grid += np.random.randint(-randomness, randomness, size=grid.shape)

        return grid

    def _create_annulus(self, annulus_diameter, frequency):
        """
        Create an annulus patch.

        Parameters
        ----------
        annulus_diameter : float
            The annulus_diameter of the annulus.
        frequency : float
            The spatial frequency of the radial modulation.

        Returns
        -------
        array_like
            The annulus patch.
        """
        r = np.linspace(-annulus_diameter / 2, annulus_diameter / 2,
                        self.patch_resolution)
        x, y = np.meshgrid(r, -r)
        radius = np.hypot(x, y)
        mask = radius <= annulus_diameter / 2
        annulus = 0.5 * np.cos(radius * frequency * 2 * np.pi + np.pi) * mask
        return annulus

    def _create_orientation_grating(self, orientation, annulus_diameter,
                                    frequency):
        """
        Create a single orientation grating.

        Parameters
        ----------
        orientation : float
            The orientation of the grating in degrees.
        annulus_diameter : float
            The annulus_diameter of the orientation patch (to match annulus size).
        frequency : float
            The spatial frequency of the orientation grating.

        Returns
        -------
        array_like
            The orientation grating.
        """
        radius = annulus_diameter / 2
        r = np.linspace(-radius, radius, self.patch_resolution)
        x, y = np.meshgrid(r, -r)
        y = -y
        x_rot = self._rotate_x(x, y, orientation)
        grating = np.sin(2 * np.pi * frequency *
                         x_rot)  # Frequency is now adjustable
        eccentricity = np.abs(x + y * 1j)
        mask = eccentricity <= radius  # Circular mask
        return grating * mask

    @staticmethod
    def _rotate_x(x, y, rotation_angle):
        """
        Rotate the x-coordinates.

        Parameters
        ----------
        x : array_like
            The x-coordinates.
        y : array_like
            The y-coordinates.
        rotation_angle : float
            The rotation angle in degrees.

        Returns
        -------
        array_like
            The rotated x-coordinates.
        """
        radian = np.radians(rotation_angle)
        return np.cos(radian) * x + np.sin(radian) * y
