"""
Plot the coverage time series for six latitude bands using the non-infilled and analysis datasets to show how
coverage has changed over time and the effect of infilling.
"""
import copy
import xarray as xa
import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs


def plot_map(inarry, filename):
    data = inarry.data
    data[data == 0] = np.nan
    inarry.data = data

    data = inarry
    lon = inarry.coords['longitude']
    lon_idx = data.dims.index('longitude')

    wrap_data, wrap_lon = add_cyclic_point(data.values, coord=lon, axis=lon_idx)

    proj = ccrs.EqualEarth(central_longitude=0)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection=proj, aspect='auto')

    # Vmin and Vmax aren't set to 0 and 1, which are the limits because we want the bottom of the scale not to be
    # white and the top of the scale not to be overly dark/saturated.
    p = ax.pcolormesh(wrap_lon, data.latitude, wrap_data[:, :], shading='auto', transform=ccrs.PlateCarree(),
                      cmap=mpl.cm.Purples, vmin=-0.3, vmax=1.7)

    p.axes.coastlines(color='#222222', linewidth=2)
    p.axes.set_global()

    plt.savefig(Path('OutputFigures') / filename, transparent=True, bbox_inches='tight')
    plt.savefig(Path('OutputFigures') / filename.replace('.png', '.svg'), transparent=True, bbox_inches='tight')
    plt.close()

    return


def plot_anomaly_map(inarry, filename):
    data = inarry.data
    data[data == 0] = np.nan
    inarry.data = data

    data = inarry
    lon = inarry.coords['longitude']
    lon_idx = data.dims.index('longitude')

    wrap_data, wrap_lon = add_cyclic_point(data.values, coord=lon, axis=lon_idx)

    proj = ccrs.EqualEarth(central_longitude=0)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection=proj, aspect='auto')

    p = ax.pcolormesh(wrap_lon, data.latitude, wrap_data[:, :],
                      shading='auto', transform=ccrs.PlateCarree(),
                      cmap=mpl.cm.RdYlBu_r, vmin=-2, vmax=2)
    p.axes.coastlines(color='#222222', linewidth=2)
    p.axes.set_global()

    plt.savefig(Path('OutputFigures') / filename, transparent=True, bbox_inches='tight')
    plt.savefig(Path('OutputFigures') / filename.replace('.png', '.svg'), transparent=True, bbox_inches='tight')
    plt.close()

    return


def plot_colorbar(filename):
    """
    All this to plot a colour bar all on its own

    :param filename:
    :return:
    """
    fig, axs = plt.subplots(1)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1)
    fig.tight_layout()
    img = axs.imshow(np.array([[0, 1]]), cmap="RdYlBu_r", vmin=-2, vmax=2)
    img.set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['bottom'].set_visible(False)
    axs.spines['left'].set_visible(False)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
    cbar = fig.colorbar(orientation="horizontal", mappable=img)
    cbar.set_ticks([-2, -1.5, -1.0, -0.5, 0, 0.5, 1, 1.5, 2])
    cbar.set_ticklabels(['', '', '', '', '', '', '', '', ''])
    cbar.outline.set_linewidth(3)
    cbar.ax.tick_params(width=3)

    plt.savefig(Path('OutputFigures') / filename, bbox_inches=mpl.transforms.Bbox([[0, 0.3], [6.5, 1.3]]))
    plt.close()


def coverage_map(data, tag):
    """
    Convert non-missing data to one and missing data to zero then plot a series of time averages every 50 years.

    :param data:
    :param tag:
    :return:
    """
    data2 = copy.deepcopy(data)

    non_missing = ~np.isnan(data2.data)
    missing = np.isnan(data2.data)

    data2.data[non_missing] = 1.0
    data2.data[missing] = 0.0

    for year in [1750, 1800, 1850, 1900, 1950, 2000]:
        time_mean = data2.sel(time=slice(f'{year}-01-01', f'{year + 49}-12-31')).mean('time')
        plot_map(time_mean, f'coverage_{year}_{year + 49}_{tag}.png')


def anomaly_map(data, tag):
    """
    Plot a series of time averages of the data in a dataframe.

    :param data: xarray dataframe
        The dataset to plot
    :param tag: str
        a tag to add to the standard filename.
    :return:
    """
    data2 = copy.deepcopy(data)
    for year in [1750, 1800, 1850, 1900, 1950, 2000]:
        time_mean = data2.sel(time=slice(f'{year}-01-01', f'{year + 49}-12-31')).mean('time')
        plot_anomaly_map(time_mean, f'anomalies_{year}_{year + 49}_{tag}.png')


def areas_map(ds):
    """
    Plot a map of the different regions plotted in the coverage timeseries

    :param ds: xarray dataframe
        It doesn't matter what this is as long as it's global.
    :return: None
    """
    data = ds.tas_median
    latsr = np.repeat(np.reshape(data.latitude.data, (36, 1)), 72, axis=1)
    pull_data = data.data[0, :, :]

    selection = (latsr >= 65.0)
    pull_data[selection] = 0 - 0.1
    selection = (latsr < 65.0) & (latsr >= 30.0)
    pull_data[selection] = 1.0 - 0.1
    selection = (latsr < 30.0) & (latsr >= 0)
    pull_data[selection] = 2.0 - 0.1
    selection = (latsr < 0.0) & (latsr >= -30)
    pull_data[selection] = 2.0 - 0.1
    selection = (latsr >= -65.0) & (latsr < -30)
    pull_data[selection] = 3.0 - 0.1
    selection = (latsr < -65.0)
    pull_data[selection] = 4.0 - 0.1

    data.data[0, :, :] = pull_data

    wmo_cols = ['#58c9db', '#8ebf84', '#ffa473', '#8ebf84', '#58c9db', '#58c9db']
    wmo_levels = [0, 1, 2, 3, 4]

    lon = ds.tas_median.coords['longitude']
    lon_idx = ds.tas_median.dims.index('longitude')

    wrap_data, wrap_lon = add_cyclic_point(data.values, coord=lon, axis=lon_idx)

    proj = ccrs.Orthographic(central_longitude=20, central_latitude=0)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection=proj)
    p = ax.contourf(wrap_lon, data.latitude, wrap_data[0, :, :],
                    transform=ccrs.PlateCarree(),
                    levels=wmo_levels,
                    colors=wmo_cols,
                    extend='both'
                    )

    p.axes.coastlines(color='#222222', linewidth=2)
    p.axes.set_global()

    filename = 'areas_map.png'
    plt.savefig(Path('OutputFigures') / filename, transparent=True, bbox_inches='tight')
    plt.savefig(Path('OutputFigures') / filename.replace('.png', '.svg'), transparent=True, bbox_inches='tight')
    plt.close()


def calculate_coverage_timeseries(ds):
    """
    Calculate the coverage timeseries in six latitude bands. Timeseries are smoothed using a 12-month running mean.

    :param ds: xarray dataframe
        Dataframe for which we want to plot coverage
    :return: (ndarray, ndarray, float)
        Returns an array containing the times of each data step, an array containing the coverage for each time step
        in seven different latitude bands (globe, arctic, northern extratropics, northern tropics, southern tropics,
        southern extratropics, antarctic), and an estimate of the polar area.
    """
    data = ds.tas_median
    time = ds.time.data
    latsr = xa.ufuncs.deg2rad(data.latitude)
    weights = xa.ufuncs.cos(latsr)
    weights = np.repeat(np.reshape(weights.data, (36, 1)), 72, axis=1)
    weights = weights / np.sum(weights)

    latsr = np.repeat(np.reshape(data.latitude.data, (36, 1)), 72, axis=1)

    data = data.data

    ntime = data.shape[0]

    coverage = np.zeros((ntime, 7))
    smooth_coverage = np.zeros((ntime, 8))

    polar_area = np.sum(weights[latsr < 65.0])

    for i in range(ntime):
        # For each area, sum the area weighs for non-missing data
        selection = ~np.isnan([data[i, :, :]])
        total_area_globe = np.sum(weights[selection[0, :, :]])
        coverage[i, 0] = total_area_globe

        selection = ~np.isnan([data[i, :, :]]) & (latsr >= 65.0)
        total_area_arctic = np.sum(weights[selection[0, :, :]])
        coverage[i, 1] = total_area_arctic

        selection = ~np.isnan([data[i, :, :]]) & (latsr < 65.0) & (latsr >= 30.0)
        total_area_north_extra_tropics = np.sum(weights[selection[0, :, :]])
        coverage[i, 2] = total_area_north_extra_tropics

        selection = ~np.isnan([data[i, :, :]]) & (latsr < 30.0) & (latsr >= 0)
        total_area_north_tropics = np.sum(weights[selection[0, :, :]])
        coverage[i, 3] = total_area_north_tropics

        selection = ~np.isnan([data[i, :, :]]) & (latsr < 0.0) & (latsr >= -30)
        total_area_south_tropics = np.sum(weights[selection[0, :, :]])
        coverage[i, 4] = total_area_south_tropics

        selection = ~np.isnan([data[i, :, :]]) & (latsr >= -65.0) & (latsr < -30)
        total_area_south_extra_tropics = np.sum(weights[selection[0, :, :]])
        coverage[i, 5] = total_area_south_extra_tropics

        selection = ~np.isnan([data[i, :, :]]) & (latsr < -65.0)
        total_area_antarctic = np.sum(weights[selection[0, :, :]])
        coverage[i, 6] = total_area_antarctic

    for i in range(11, ntime):
        for j in range(7):
            smooth_coverage[i, j] = np.mean(coverage[i - 11:i + 1, j])

    return time, smooth_coverage, polar_area


def plot_non_centred_coverage_timeseries(time, smooth_coverage):
    """
    Plot the coverage time series

    :param time:
    :param smooth_coverage:
    :return:
    """
    spole = '#58c9db'
    npole = '#58c9db'
    sh_extra = '#8ebf84'
    nh_extra = '#8ebf84'
    tropics = '#ffa473'

    ntime = len(time)
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(12, 3)

    plt.fill_between(time, smooth_coverage[:, 6],
                     0.0 - smooth_coverage[:, 4] - smooth_coverage[:, 5] - smooth_coverage[:, 6], color=spole)
    plt.fill_between(time, smooth_coverage[:, 6], 0.0 - smooth_coverage[:, 4] - smooth_coverage[:, 5], color=sh_extra)
    plt.fill_between(time, smooth_coverage[:, 6], 0.0 - smooth_coverage[:, 4], color=tropics)

    plt.fill_between(time, smooth_coverage[:, 6], smooth_coverage[:, 3] + smooth_coverage[:, 2] + smooth_coverage[:, 1],
                     color=npole)
    plt.fill_between(time, smooth_coverage[:, 6], smooth_coverage[:, 3] + smooth_coverage[:, 2], color=nh_extra)
    plt.fill_between(time, smooth_coverage[:, 6], smooth_coverage[:, 3], color=tropics)

    nz = np.zeros(ntime)
    plt.plot(time, nz + polar_area - 0.50, color='black', alpha=0.5)
    plt.plot(time, nz + 0.50, color='black', alpha=0.5)
    plt.plot(time, nz + 0.25, color='black', alpha=0.5)
    plt.plot(time, nz + 0.00, color='black', alpha=0.5)
    plt.plot(time, nz - 0.25, color='black', alpha=0.5)
    plt.plot(time, nz - 0.50, color='black', alpha=0.5)
    plt.plot(time, nz - polar_area + 0.50, color='black', alpha=0.5)

    plt.gca().tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                          labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    plt.gca().set_axis_off()

    plt.savefig(Path('OutputFigures') / 'coverage_centred.png', dpi=300, transparent=True, bbox_inches='tight')
    plt.savefig(Path('OutputFigures') / 'coverage_centred.svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.savefig(Path('OutputFigures') / 'coverage_centred.pdf', dpi=300, transparent=True, bbox_inches='tight')
    plt.close()


def plot_centred_coverage_timeseries(time, smooth_coverage):
    spole = '#58c9db'
    npole = '#58c9db'
    sh_extra = '#8ebf84'
    nh_extra = '#8ebf84'
    tropics = '#ffa473'

    ntime = len(time)
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(24, 4)

    nz = np.zeros(ntime)
    plt.fill_between(time, nz - polar_area + 0.50, nz - polar_area + 0.50 - smooth_coverage[:, 6], color=spole)
    plt.fill_between(time, nz - 0.25, nz - 0.25 - smooth_coverage[:, 5], color=sh_extra)
    plt.fill_between(time, nz, 0.0 - smooth_coverage[:, 4], color=tropics)

    plt.fill_between(time, nz + polar_area - 0.5, nz + polar_area - 0.5 + smooth_coverage[:, 1], color=npole)
    plt.fill_between(time, nz + 0.25, nz + 0.25 + smooth_coverage[:, 2], color=nh_extra)
    plt.fill_between(time, nz, smooth_coverage[:, 3], color=tropics)

    plt.plot(time, nz + polar_area - 0.50, color='black', linewidth=0.5)
    plt.plot(time, nz + 0.50, color='black', linewidth=0.5)
    plt.plot(time, nz + 0.25, color='black', linewidth=0.5)
    plt.plot(time, nz + 0.00, color='black', linewidth=0.5)
    plt.plot(time, nz - 0.25, color='black', linewidth=0.5)
    plt.plot(time, nz - 0.50, color='black', linewidth=0.5)
    plt.plot(time, nz - polar_area + 0.50, color='black', linewidth=0.5)

    plt.gca().tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                          labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    plt.gca().set_axis_off()

    plt.savefig(Path('OutputFigures') / 'coverage_individual.png', dpi=300, transparent=True, bbox_inches='tight')
    plt.savefig(Path('OutputFigures') / 'coverage_individual.svg', dpi=300, transparent=True, bbox_inches='tight')
    plt.savefig(Path('OutputFigures') / 'coverage_individual.pdf', dpi=300, transparent=True, bbox_inches='tight')
    plt.close()


def plot_centred_coverage_comparison_timeseries(time, smooth_coverage, smooth_coverage_unfilled):
    spole = '#58c9db'
    npole = '#58c9db'
    sh_extra = '#8ebf84'
    nh_extra = '#8ebf84'
    tropics = '#ffa473'

    ntime = len(time)
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(24, 4)

    nz = np.zeros(ntime)

    # Analysis layers
    plt.fill_between(time, nz - polar_area + 0.50, nz - polar_area + 0.50 - smooth_coverage[:, 6], color=spole)
    plt.fill_between(time, nz - 0.25, nz - 0.25 - smooth_coverage[:, 5], color=sh_extra)
    plt.fill_between(time, nz, 0.0 - smooth_coverage[:, 4], color=tropics)

    plt.fill_between(time, nz + polar_area - 0.5, nz + polar_area - 0.5 + smooth_coverage[:, 1], color=npole)
    plt.fill_between(time, nz + 0.25, nz + 0.25 + smooth_coverage[:, 2], color=nh_extra)
    plt.fill_between(time, nz, smooth_coverage[:, 3], color=tropics)

    # And unfilled
    alfa = 0.3
    plt.fill_between(time, nz - polar_area + 0.50, nz - polar_area + 0.50 - smooth_coverage_unfilled[:, 6],
                     color='white', alpha=alfa)
    plt.fill_between(time, nz - 0.25, nz - 0.25 - smooth_coverage_unfilled[:, 5], color='white', alpha=alfa)
    plt.fill_between(time, nz, 0.0 - smooth_coverage_unfilled[:, 4], color='white', alpha=alfa)

    plt.fill_between(time, nz + polar_area - 0.5, nz + polar_area - 0.5 + smooth_coverage_unfilled[:, 1], color='white',
                     alpha=alfa)
    plt.fill_between(time, nz + 0.25, nz + 0.25 + smooth_coverage_unfilled[:, 2], color='white', alpha=alfa)
    plt.fill_between(time, nz, smooth_coverage_unfilled[:, 3], color='white', alpha=alfa)

    plt.plot(time, nz + polar_area - 0.50, color='black', linewidth=0.5)
    plt.plot(time, nz + 0.50, color='black', linewidth=0.5)
    plt.plot(time, nz + 0.25, color='black', linewidth=0.5)
    plt.plot(time, nz + 0.00, color='black', linewidth=0.5)
    plt.plot(time, nz - 0.25, color='black', linewidth=0.5)
    plt.plot(time, nz - 0.50, color='black', linewidth=0.5)
    plt.plot(time, nz - polar_area + 0.50, color='black', linewidth=0.5)

    plt.gca().tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                          labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    plt.gca().set_axis_off()

    text_color = '#4F589D'

    for y in [1750, 1800, 1850, 1900, 1950, 2000]:
        t = plt.text(date(y, 1, 1), 0.53, f'{y}', fontsize=24, va='bottom', ha='center', color=text_color)
        plt.text(date(y, 1, 1), -0.53, f'{y}', fontsize=24, va='top', ha='center', color=text_color)

        r = fig.canvas.get_renderer()
        bb = t.get_window_extent(renderer=r).transformed(axs.transData.inverted())
        height = bb.height * 1.4

        gap = 9
        ystart = date(y + gap, 1, 1)
        yend = date(y + 50 - gap, 1, 1)

        if y == 2000:
            yend = date(2023 - gap + 2, 1, 1)

        plt.plot(
            [ystart, yend],
            [0.53 + height / 2, 0.53 + height / 2], color=text_color
        )
        plt.plot(
            [ystart, yend],
            [-0.53 - height / 3, -0.53 - height / 3], color=text_color
        )

    plt.text(date(2023, 1, 1), 0.53, f'Now', fontsize=24, va='bottom', ha='center', color=text_color)
    plt.text(date(2023, 1, 1), -0.53, f'Now', fontsize=24, va='top', ha='center', color=text_color)

    plt.savefig(Path('OutputFigures') / 'coverage_comparison_individual.png', dpi=300, transparent=True,
                bbox_inches='tight')
    plt.savefig(Path('OutputFigures') / 'coverage_comparison_individual.svg', dpi=300, transparent=True,
                bbox_inches='tight')
    plt.savefig(Path('OutputFigures') / 'coverage_comparison_individual.pdf', dpi=300, transparent=True,
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    data_dir_env = os.getenv('DATADIR')

    glosat_dir = Path(data_dir_env) / 'GloSAT' / 'analysis' / 'diagnostics'
    filename = 'GloSATref.1.0.0.0.analysis.anomalies.ensemble_median.nc'

    plot_colorbar('colorbar.svg')

    ds = xa.open_dataset(glosat_dir / filename)
    data = ds.tas_median
    anomaly_map(data, 'anomalies')

    ds = xa.open_dataset(glosat_dir / filename)
    data = ds.tas_median
    coverage_map(data, 'analysis')
    areas_map(ds)
    time, smooth_coverage, polar_area = calculate_coverage_timeseries(ds)

    glosat_dir = Path(data_dir_env) / 'GloSAT' / 'unfilled' / 'diagnostics'
    filename = 'GloSATref.1.0.0.0.noninfilled.anomalies.ensemble_median.nc'

    ds = xa.open_dataset(glosat_dir / filename)
    data = ds.tas_median
    coverage_map(data, 'unfilled')
    time_unfilled, smooth_coverage_unfilled, _ = calculate_coverage_timeseries(ds)

    # plot_non_centred_coverage_timeseries(time, smooth_coverage)
    # plot_centred_coverage_timeseries(time, smooth_coverage)
    plot_centred_coverage_comparison_timeseries(time, smooth_coverage, smooth_coverage_unfilled)
