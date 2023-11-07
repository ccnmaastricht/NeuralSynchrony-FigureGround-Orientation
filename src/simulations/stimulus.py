diameter = 0.7
frequency = 5.7
resolution = int(round(np.sqrt(num_pixels) * diameter / side_length))

stim_res = int(np.sqrt(num_pixels))

scaling = 1

def create_annulus(diameter, frequency, resolution):
    r = np.linspace(-diameter/2, diameter/2, resolution)
    X, Y = np.meshgrid(r, r)
    Y = -Y
    radius = np.sqrt(X**2 + Y**2)
    mask = radius <= diameter/2
    annulus = np.cos(radius * frequency * 2 * np.pi + np.pi) * mask
    return annulus


stimulus = np.zeros((stim_res, stim_res))

grid = np.arange(resolution//2, stim_res, int(resolution * scaling))
X_grid, Y_grid = np.meshgrid(grid, grid)
grid = np.vstack((X_grid.flatten(), Y_grid.flatten())).T


randomness = (resolution * scaling - resolution) // 2
if randomness > 0:
    grid = grid + np.random.randint(-randomness, randomness, size=grid.shape)

annulus = create_annulus(diameter, frequency, resolution)


for x, y in grid:
    lower_x = np.maximum(x - resolution//2, 0)
    upper_x = np.minimum(x + resolution//2, stim_res)
    lower_y = np.maximum(y - resolution//2, 0)
    upper_y = np.minimum(y + resolution//2, stim_res)

    range_x = upper_x - lower_x
    range_y = upper_y - lower_y

    stimulus[lower_x:upper_x, lower_y:upper_y] = annulus[:range_x, :range_y]