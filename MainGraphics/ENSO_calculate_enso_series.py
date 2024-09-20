"""
Calculate global series from the ensemble members of GloSAT and HadCRUT
"""

import xarray as xa
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.dates as mdates


def calculate_area_average(ds_orig, region, window=3):
    lon0 = region[0]
    lon1 = region[1]
    lat0 = region[2]
    lat1 = region[3]

    # Select region and variable (tas)
    ds = ds_orig.sel(latitude=slice(lat0, lat1), longitude=slice(lon0, lon1))
    data = ds.tas

    # Do area weighted averaging and then rolling average
    latsr = xa.ufuncs.deg2rad(data.latitude)
    weights = xa.ufuncs.cos(latsr)
    weighted_mean = data.weighted(weights).mean(dim=("latitude", "longitude"))
    df = weighted_mean.to_dataframe(name='tas')
    df = df.rolling(window=window).mean()

    return df.tas.values[:], df.index.array


def read_standard_time_series(filename, window=12):
    ds = xa.open_dataset(glosat_diagnostic_dir / filename)
    data = ds.tas_mean
    uncertainty = ds.tas_total_unc
    df_data = data.to_dataframe(name='tas')
    df_data = df_data.rolling(window=12).mean()
    df_uncertainty = uncertainty.to_dataframe(name='tas_total_unc')
    df_uncertainty = df_uncertainty.rolling(window=window).mean()
    glosat = df_data.tas.array[:]
    glosat_unc = df_uncertainty.tas_total_unc.array[:]
    glosat_time = df_data.index.array
    return glosat, glosat_unc, glosat_time


def summarise_ensemble(ensemble):
    ensemble_mean = np.mean(ensemble, axis=1)
    ensemble_max = np.max(ensemble, axis=1)
    ensemble_min = np.min(ensemble, axis=1)

    return ensemble_mean, ensemble_min, ensemble_max


def get_el_nino_years():
    # Table 2 in Meyers et al. (2006) From  The Years of El Niño, La Niña, and Interactions with the Tropical Indian Ocean.
    # Journal of Climate
    el_nino_years = [
        1877, 1888, 1896, 1899,
        1902,1905, 1911, 1914, 1918, 1923, 1925, 1930, 1940, 1941,
        1957,1963, 1965, 1972, 1982,1986,1987,1991, 1997,
        2002, 2009, 2015, 2023
    ]
    return el_nino_years


def get_la_nina_years():
    # Table 2 in Meyers et al. (2006) From  The Years of El Niño, La Niña, and Interactions with the Tropical Indian Ocean.
    # Journal of Climate
    la_nina_years = [
        1878, 1879, 1886, 1889, 1890, 1892, 1893, 1897,
        1903, 1906, 1909, 1910, 1916, 1917, 1922, 1924, 1928, 1933, 1938, 1942, 1949,
        1950, 1954, 1955, 1964, 1970, 1971, 1973, 1975, 1978, 1981, 1984, 1988, 1996, 1998, 1999,
        2000, 2007, 2010, 2011, 2017, 2020, 2021, 2022,
    ]
    return la_nina_years


def get_volcanoes():
    volcanoes = {
        'Pinatubo': [datetime.date(1991, 6, 12), datetime.date(1992, 6, 12)],
        'El Chichon': [datetime.date(1982, 3, 29), datetime.date(1983, 3, 29)],
        'Mt Agung': [datetime.date(1963, 2, 18), datetime.date(1964, 2, 18)],
        'Krakatoa': [datetime.date(1883, 8, 26), datetime.date(1884, 8, 26)],
        'Santa Maria': [datetime.date(1902, 10, 24), datetime.date(1903, 10, 24)],
    }
    return volcanoes


def plot_rectangles_for_list_of_years(year_list, color):
    el_nino_list = {}
    for y in year_list:
        el_nino_list[str(y)] = [datetime.date(y, 1, 1), datetime.date(y + 1, 1, 1)]
    for key in el_nino_list:
        plt.fill_between(el_nino_list[key], [-4, -4], [4, 4], color=color, alpha=0.1)


if __name__ == '__main__':

    data_dir_env = os.getenv('DATADIR')
    glosat_dir = Path(data_dir_env) / 'GloSAT' / 'glosatref1000'
    glosat_diagnostic_dir = Path(data_dir_env) / 'GloSAT' / 'analysis' / 'diagnostics'
    hadcrut_dir = Path(data_dir_env) / 'GloSAT' / 'hadcrut5'

    filename = f'GloSATref.1.0.0.0.analysis.component_series.global.monthly.nc'
    glosat, glosat_unc, glosat_time = read_standard_time_series(glosat_diagnostic_dir / filename)

    areas = {
        'enso': [-180, -120, -5, 5],
        'nh': [-180, 180, 0, 90],
        'sh': [-180, 180, -90, 0]
    }

    n_ensemble = 200

    glosat_summaries = {}
    hadcrut_summaries = {}
    for area in areas:
        glosat_summaries[area] = np.zeros((2892, n_ensemble))
        hadcrut_summaries[area] = np.zeros((2092, n_ensemble))

    # Calculate area averages from all ensemble members for the specified areas
    for i in range(1, n_ensemble + 1):
        print(f"Ensemble member {i}")
        filename = f'GloSATref.1.0.0.0.analysis.anomalies.{i}.nc'
        ds_orig = xa.open_dataset(glosat_dir / filename)
        for area in areas:
            ts, time_axis = calculate_area_average(ds_orig, areas[area])
            glosat_summaries[area][:, i - 1] = ts[:]
            glosat_time = time_axis

        filename = f'HadCRUT.5.0.2.0.analysis.anomalies.{i}.nc'
        ds_orig = xa.open_dataset(hadcrut_dir / filename)
        for area in areas:
            ts, time_axis = calculate_area_average(ds_orig, areas[area])
            hadcrut_summaries[area][:, i - 1] = ts[:]
            hadcrut_time = time_axis

    # Calculate ensemble means, mins and maxs
    glosat_mean, glosat_low, glosat_high = summarise_ensemble(glosat_summaries['enso'])
    hadcrut_mean, hadcrut_low, hadcrut_high = summarise_ensemble(hadcrut_summaries['enso'])
    hadcrut_globe, hadcrut_globe_low, hadcrut_globe_high = summarise_ensemble(
        0.5 * hadcrut_summaries['nh'] + 0.5 * hadcrut_summaries['sh'])

    # Plot the global mean temperature with red and blue stripes to indicate El Nino and La Nina events
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(12, 4)

    plt.fill_between(glosat_time, glosat - glosat_unc, glosat + glosat_unc, color='black', alpha=0.2)
    plt.plot(glosat_time, glosat, color="black")

    el_nino_years = get_el_nino_years()
    plot_rectangles_for_list_of_years(el_nino_years, "red")

    la_nina_years = get_la_nina_years()
    plot_rectangles_for_list_of_years(la_nina_years, "blue")

    plt.gca().set_ylim(-1, 1.2)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlim([datetime.date(1870, 1, 1), datetime.date(2024, 12, 31)])
    axs.xaxis.set_major_locator(mdates.YearLocator(base=10))
    axs.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.savefig(Path('OutputFigures') / 'globe.png', dpi=300, transparent=False, bbox_inches='tight')
    plt.savefig(Path('OutputFigures') / 'globe.svg', dpi=300, transparent=False, bbox_inches='tight')
    plt.close()

    # Plot the ENSO time series
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(12, 4)

    plt.fill_between(hadcrut_time, hadcrut_low, hadcrut_high, color="#ff7f0e", alpha=0.1)
    plt.plot(hadcrut_time, hadcrut_mean, color="#ff7f0e", linewidth=0.5)

    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_xlim([datetime.date(1870, 1, 1), datetime.date(2024, 12, 31)])
    axs.xaxis.set_major_locator(mdates.YearLocator(base=10))
    axs.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    volcanoes = get_volcanoes()
    for key in volcanoes:
        plt.fill_between(volcanoes[key], [-4, -4], [4, 4], color="green", alpha=0.1)

    plt.gca().set_ylim(-3.1, 3.1)

    plt.savefig(Path('OutputFigures') / 'enso.png', dpi=300, transparent=False, bbox_inches='tight')
    plt.savefig(Path('OutputFigures') / 'enso.svg', dpi=300, transparent=False, bbox_inches='tight')
    plt.close()
