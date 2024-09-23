import itertools
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


def make_correlated_noise(rng, nx: int, ny: int, stdev: float, x_length_scale: float, y_length_scale: float):
    """
    Produce an array of spatially correlated noise. The size of the grid is nx by ny. The standard deviation is
    controlled by stdev and there are separate length scales for the x and y directions

    :param rng:
        Numpy random number generator
    :param nx: int
        Number of grid cells in x-direction
    :param ny: int
        Number of grid cells in y-direction
    :param stdev: float
        standard deviation of field
    :param x_length_scale: float
        length scale in x direction. Larger numbers give larger features
    :param y_length_scale: float
        length scale in the y direction. Larger numbers give larger features.
    :return: numpy array
        Returns an nx by ny numpy array containing spatially correlated noise
    """

    # The first row and column are noise so expand array by one row and column and remove at final step
    nx += 1
    ny += 1

    data_grid = np.zeros((nx, ny))

    # Set up the x and y coordinates for the region
    x_coords = np.zeros((nx, ny))
    y_coords = np.zeros((nx, ny))

    for x in range(nx):
        x_coords[x, :] = float(x)
    for y in range(ny):
        y_coords[:, y] = float(y)

    x_coords = x_coords.reshape(1, nx * ny)
    y_coords = y_coords.reshape(1, nx * ny)

    x_coords = np.repeat(x_coords, nx * ny, 0)
    y_coords = np.repeat(y_coords, nx * ny, 0)

    # Calculate distances between all pairs of points and calculate covariance matrix
    scaled_distances = np.sqrt(
        ((x_coords - np.transpose(x_coords)) / x_length_scale) ** 2
        +
        ((y_coords - np.transpose(y_coords)) / y_length_scale) ** 2
    )
    covariance = stdev * stdev * np.exp(-1 * scaled_distances)

    # Draw samples from the covariance matrix using the cholesky decomposition
    chol = np.linalg.cholesky(covariance)
    noise = rng.normal(loc=0, scale=1.0, size=(1, nx * ny))
    sample = np.matmul(noise, chol)

    # Put sample vector into the original 2D grid
    data_grid[:, :] = np.reshape(sample, (nx, ny))
    return data_grid[1:, 1:]


def quad_trend_grid(nx: int, ny: int):
    """
    Create an nx by ny grid that has a quadratic trend in the y direction.

    :param nx: int
        Number of grid cells in x-direction
    :param ny: int
        Number of grid cells in y-direction
    :return: numpy array
        Returns an nx by ny numpy array containing a quadratic trend in the y-direction
    """
    data_grid = np.zeros((nx, ny))

    for y in range(ny):
        data_grid[:, y] = (y - 0.006 * y * y) / 80

    return data_grid


def make_periodic_grid(nx: int, ny: int, var, freq: float = 5):
    """
    Create an nx by ny grid that has periodic variations in the y direction.

    :param nx: int
        Number of grid cells in x-direction
    :param ny: int
        Number of grid cells in y-direction
    :param var: float
        scale variance of the series. If set to one, the series will vary from -1 to +1
    :param freq: float
        Number of cycles to include
    :return: numpy array
        Returns an nx by ny numpy array containing periodic variations in the y-direction
    """
    data_grid = np.zeros((nx, ny))

    for y in range(ny):
        data_grid[:, y] = var * np.cos(y * freq * 2 * np.pi / ny)

    return data_grid


def make_outliers(rng, nx: int, ny: int, cut_off: float, scale: float):
    """
    Designate random gridcells as outliers. Calculation is done by generating normally distributed noise. Setting
    any absolute value less than the cut_off to zero and scaling the remaining values by scale

    :param rng:
        Numpy random number generator
    :param nx: int
        Number of grid cells in x-direction
    :param ny: int
        Number of grid cells in y-direction
    :param cut_off: float
        Cut off value. Values closer to zero will increase the number of outliers
    :param scale: float
        Outlier values are drawn from a normal distribution scaled by this parameter
    :return: numpy arrau
        Returns an nx by ny numpy array containing random outliers
    """
    data_grid = rng.normal(0, 1.0, size=(nx, ny))
    outliers = abs(data_grid) > cut_off
    rest = abs(data_grid) < cut_off
    data_grid[rest] = 0.0
    data_grid[outliers] = scale * data_grid[outliers]
    return data_grid


def make_white_noise(rng, nx, ny, stdev):
    """
    Create an nx by ny grid containing white noise with standard deviation equal to stdev

    :param rng:
        Numpy random number generator
    :param nx: int
        Number of grid cells in x-direction
    :param ny: int
        Number of grid cells in y-direction
    :param stdev: float
        standard deviation of gaussian from which values are to be drawn
    :return: numpy array
        Returns an nx by ny numpy array containing white noise
    """
    data_grid = rng.normal(0, stdev, size=(nx, ny))
    return data_grid


def make_sampling(rng, nx, ny, cut_off):
    """
    Create a grid which has a random number of cells set to np.NaN. The missing cells are decided by generating
    correlated noise and removing any cell whose value exceeds the cut_off value.

    :param rng:
        Numpy random number Generator
    :param nx: int
        Number of grid cells in x-direction
    :param ny: int
        Number of grid cells in y-direction
    :param cut_off: float
        Used to control the amount of missing data. Values closer to zero will generate more missing data
    :return: numpy array
        Returns an nx by ny numpy array with randomly chosen cells set to np.NaN
    """
    data_grid = make_correlated_noise(rng, nx, ny, 1, 3, 3)
    outliers = abs(data_grid) > cut_off
    data_grid[:, :] = 0.0
    data_grid[outliers] = np.nan
    return data_grid


def make_bias_error(nx, ny):
    """
    Create a grid which starts by decreasing with a sinusoidal variation, then jumping to a higher constant value

    :param nx: int
        Number of grid cells in x-direction
    :param ny: int
        Number of grid cells in y-direction
    :return: numpy array
        Returns an nx by ny numpy array which starts by decreasing with a sinusoidal variation, then jumping to a
        higher constant value
    """
    data_grid = np.zeros((nx, ny))
    for i in range(int(ny / 2)):
        data_grid[:, i] = 0 - i * 0.001 + 0.02 * np.sin(i * 2 * np.pi * 8 / ny)
    for i in range(int(ny / 2), ny):
        data_grid[:, i] = 0.05

    return data_grid

if __name__ == '__main__':

    # Make fonts in pdf and svg render correctly in other programs
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'

    seed = 45234573570987982
    rng = np.random.default_rng(seed)

    # Set grid size
    nx = 100
    ny = 100

    # Plot parameters
    ypos_text = 103
    cmap_real = plt.cm.BuPu_r
    cmap_artificial = plt.cm.YlOrBr_r
    cmap_diverge = plt.cm.plasma

    mid = int(nx / 2)

    # Generate the different components of real variability
    corr_noise_grid = make_correlated_noise(rng, nx, ny, 0.1, 4, 12)
    trend_grid = quad_trend_grid(nx, ny)
    trend_grid = trend_grid - np.mean(trend_grid)

    periodic = make_periodic_grid(nx, ny, 0.05, 5)

    # and artificial variability
    outliers = make_outliers(rng, nx, ny, 1.8, 0.25)
    white_noise = make_white_noise(rng, nx, ny, 0.1)
    bias = make_bias_error(nx, ny)
    sampling = make_sampling(rng, nx, ny, 1.0)

    # Start the plot
    fig, axs = plt.subplots(3, 6)
    fig.set_size_inches(18, 9)

    # Set parameters for each plot so there are no axes, no tick marks and no labels
    for i, j in itertools.product(range(3), range(6)):
        axs[i, j].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                              labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        if j in [1, 3, 5]:
            axs[i, j].set_axis_off()

    # Actual variability
    axs[0, 0].pcolormesh(trend_grid, cmap=cmap_real)
    axs[0, 1].plot(trend_grid[mid, :], color=cmap_real(0.5))
    axs[0, 0].text(1, ypos_text, 'Trend, slow variations', fontsize=18)
    axs[0, 0].text(100, 125, 'Real variability', fontsize=24, ha='center')

    axs[1, 0].pcolormesh(corr_noise_grid, cmap=cmap_real)
    axs[1, 1].plot(corr_noise_grid[mid, :], color=cmap_real(0.5))
    axs[1, 0].text(1, ypos_text, 'Internal variability', fontsize=18)

    axs[2, 0].pcolormesh(periodic, cmap=cmap_real)
    axs[2, 1].plot(periodic[mid, :], color=cmap_real(0.5))
    axs[2, 0].text(1, ypos_text, 'Periodic variability', fontsize=18)

    # Middle columns - combined variability
    combined = trend_grid + corr_noise_grid + periodic
    combined2 = trend_grid + periodic

    vmin = -0.5
    vmax = 0.5

    # For real variability show separate cumulative combination as light overlay
    axs[0, 2].pcolormesh(combined, vmin=vmin, vmax=vmax, cmap=cmap_real)
    axs[0, 3].plot(combined[mid, :], color=cmap_real(0.5))
    axs[0, 3].plot(trend_grid[mid, :], color=cmap_real(0.5), alpha=0.2)
    axs[0, 3].plot(combined2[mid, :], color=cmap_real(0.5), alpha=0.2)
    axs[0, 2].text(1, ypos_text, 'Combined real variability', fontsize=18)

    combined_error = outliers + white_noise + bias

    axs[1, 2].pcolormesh(combined_error, vmin=vmin, vmax=vmax, cmap=cmap_artificial)
    axs[1, 3].plot(combined_error[mid, :], color=cmap_artificial(0.5))
    axs[1, 2].text(1, ypos_text, 'Combined artificial variability', fontsize=18)

    # Adjust the y-position of the individual combined plots so they are more central
    for i, j in itertools.product(range(0, 2), range(2, 4)):
        pos = axs[i, j].get_position()
        pos.y0 = pos.y0 - 0.125 + 0.0625 - i * 0.125
        pos.y1 = pos.y1 - 0.125 + 0.0625 - i * 0.125
        axs[i, j].set_position(pos)

    combined_everything = combined + combined_error + sampling

    axs[2, 2].pcolormesh(combined_everything, vmin=vmin, vmax=vmax, cmap=cmap_diverge)
    axs[2, 3].plot(combined_everything[mid, :], color=cmap_diverge(0.5))
    axs[2, 2].text(1, ypos_text, 'Observed variability and sampling', fontsize=18)

    # Adjust y-position of the combined plots so they appear below the level of the others
    for j in range(2, 4):
        pos = axs[2, j].get_position()
        pos.y0 = pos.y0 - 0.25
        pos.y1 = pos.y1 - 0.25
        axs[2, j].set_position(pos)

    # Artificial variability
    axs[0, 4].pcolormesh(outliers, cmap=cmap_artificial)
    axs[0, 5].plot(outliers[mid, :], color=cmap_artificial(0.5))
    axs[0, 4].text(1, ypos_text, 'Outliers, transcription errors', fontsize=18)
    axs[0, 4].text(100, 125, 'Artificial variability', fontsize=24, ha='center')

    axs[1, 4].pcolormesh(white_noise, cmap=cmap_artificial)
    axs[1, 5].plot(white_noise[mid, :], color=cmap_artificial(0.5))
    axs[1, 4].text(1, ypos_text, 'Noise', fontsize=18)

    axs[2, 4].pcolormesh(bias, cmap=cmap_artificial)
    axs[2, 5].plot(bias[mid, :], color=cmap_artificial(0.5))
    axs[2, 4].text(1, ypos_text, 'Observational biases', fontsize=18)

    plt.savefig('variability_schematic.png', bbox_inches='tight', dpi=50)

    # If higher resolution is needed, you can change the dpi or use the svg option below. File size is large ~18 Mbytes
    plt.savefig('variability_schematic.svg', bbox_inches='tight')
    plt.savefig('variability_schematic.pdf', bbox_inches='tight')

    plt.close()
