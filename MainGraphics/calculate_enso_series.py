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

data_dir_env = os.getenv('DATADIR')
glosat_dir = Path(data_dir_env) / 'GloSAT' / 'glosatref1000'
glosat_diagnostic_dir = Path(data_dir_env) / 'GloSAT' / 'analysis' / 'diagnostics'
hadcrut_dir = Path(data_dir_env) / 'GloSAT' / 'hadcrut5'

filename = f'GloSATref.1.0.0.0.analysis.component_series.global.monthly.nc'
print(filename)
ds = xa.open_dataset(glosat_diagnostic_dir / filename)
data = ds.tas_mean
uncertainty = ds.tas_total_unc
df_data = data.to_dataframe(name='tas')
df_data = df_data.rolling(window=12).mean()
df_uncertainty = uncertainty.to_dataframe(name='tas_total_unc')
df_uncertainty = df_uncertainty.rolling(window=12).mean()
glosat = df_data.tas.array[:]
glosat_unc = df_uncertainty.tas_total_unc.array[:]
glosat_time = df_data.index.array

areas = {
    'enso': [-180, -120, -5, 5],
    'nh': [-180, 180, 0, 90],
    'sh': [-180, 180, -90, 0]
}

# lon0 = -180
# lon1 = -120
# lat0 = -5
# lat1 = 5

n_ensemble = 200

glosat_summaries = {}
hadcrut_summaries = {}
for area in areas:
    glosat_summaries[area] = np.zeros((2892, n_ensemble))
    hadcrut_summaries[area] = np.zeros((2092, n_ensemble))

for i in range(1, n_ensemble + 1):
    filename = f'GloSATref.1.0.0.0.analysis.anomalies.{i}.nc'
    print(filename)
    ds_orig = xa.open_dataset(glosat_dir / filename)

    for area in areas:
        lon0 = areas[area][0]
        lon1 = areas[area][1]
        lat0 = areas[area][2]
        lat1 = areas[area][3]

        ds = ds_orig.sel(latitude=slice(lat0, lat1), longitude=slice(lon0, lon1))
        data = ds.tas
        latsr = xa.ufuncs.deg2rad(data.latitude)
        weights = xa.ufuncs.cos(latsr)
        weighted_mean = data.weighted(weights).mean(dim=("latitude", "longitude"))
        df = weighted_mean.to_dataframe(name='tas')
        df = df.rolling(window=3).mean()

        glosat_summaries[area][:, i - 1] = df.tas.values[:]

        glosat_time = df.index.array

    filename = f'HadCRUT.5.0.2.0.analysis.anomalies.{i}.nc'
    print(filename)
    ds_orig = xa.open_dataset(hadcrut_dir / filename)

    for area in areas:
        lon0 = areas[area][0]
        lon1 = areas[area][1]
        lat0 = areas[area][2]
        lat1 = areas[area][3]

        ds = ds_orig.sel(latitude=slice(lat0, lat1), longitude=slice(lon0, lon1))
        data = ds.tas
        latsr = xa.ufuncs.deg2rad(data.latitude)
        weights = xa.ufuncs.cos(latsr)
        weighted_mean = data.weighted(weights).mean(dim=("latitude", "longitude"))
        df = weighted_mean.to_dataframe(name='tas')
        df = df.rolling(window=3).mean()

        hadcrut_summaries[area][:, i - 1] = df.tas.values[:]

        hadcrut_time = df.index.array

ntime = 2892
glosat_mean = np.zeros(ntime)
glosat_low = np.zeros(ntime)
glosat_high = np.zeros(ntime)

for i in range(ntime):
    glosat_mean[i] = np.mean(glosat_summaries['enso'][i, :])
    glosat_low[i] = np.min(glosat_summaries['enso'][i, :])
    glosat_high[i] = np.max(glosat_summaries['enso'][i, :])

ntime = 2092
hadcrut_mean = np.zeros(ntime)
hadcrut_low = np.zeros(ntime)
hadcrut_high = np.zeros(ntime)

hadcrut_globe = np.zeros(ntime)
hadcrut_globe_low = np.zeros(ntime)
hadcrut_globe_high = np.zeros(ntime)

for i in range(ntime):
    hadcrut_mean[i] = np.mean(hadcrut_summaries['enso'][i, :])
    hadcrut_low[i] = np.min(hadcrut_summaries['enso'][i, :])
    hadcrut_high[i] = np.max(hadcrut_summaries['enso'][i, :])

    hadcrut_globe[i] = (np.mean(hadcrut_summaries['nh'][i, :]) + np.mean(hadcrut_summaries['sh'][i, :])) / 2.
    hadcrut_globe_low[i] = np.min(hadcrut_summaries['nh'][i, :] + hadcrut_summaries['sh'][i, :]) / 2.
    hadcrut_globe_high[i] = np.max(hadcrut_summaries['nh'][i, :] + hadcrut_summaries['sh'][i, :]) / 2.

fig, axs = plt.subplots(1, 1)
fig.set_size_inches(12, 4)

plt.fill_between(glosat_time, glosat - glosat_unc, glosat + glosat_unc, color='black', alpha=0.2)
plt.plot(glosat_time, glosat, color="black")

#axs.spines['bottom'].set_position('zero')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_xlim([datetime.date(1870, 1, 1), datetime.date(2024, 12, 31)])
axs.xaxis.set_major_locator(mdates.YearLocator(base=10))
axs.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# years from https://psl.noaa.gov/enso/past_events.html and https://www.researchgate.net/figure/List-of-El-Nino-and-La-Nina-years-1875-1997_tbl2_360776327
el_nino_years = [
    1877, 1880, 1884, 1887, 1891, 1896,
    1897, 1900, 1903, 1906, 1915, 1919, 1926, 1931, 1941, 1942, 1958, 1966, 1973, 1978, 1980, 1983, 1987,
    1988, 1992, 1995, 1998, 2003, 2007, 2010, 2016
]
el_nino_list = {}
for y in el_nino_years:
    el_nino_list[str(y)] = [datetime.date(y, 1, 1), datetime.date(y + 1, 1, 1)]
for key in el_nino_list:
    plt.fill_between(el_nino_list[key], [-4, -4], [4, 4], color="red", alpha=0.1)

la_nina_years = [
    1886, 1889, 1892, 1898, 1903, 1906,
    1904, 1909, 1910, 1911, 1917, 1918, 1925, 1934, 1939, 1943, 1950, 1951, 1955, 1956, 1962, 1971, 1974, 1976, 1989,
    1999, 2000, 2008, 2011, 2012, 2021, 2022
]
la_nina_list = {}
for y in la_nina_years:
    la_nina_list[str(y)] = [datetime.date(y, 1, 1), datetime.date(y + 1, 1, 1)]
for key in la_nina_list:
    plt.fill_between(la_nina_list[key], [-4, -4], [4, 4], color="blue", alpha=0.1)

plt.gca().set_ylim(-1, 1.2)

plt.savefig(Path('OutputFigures') / 'globe.png', dpi=300, transparent=False, bbox_inches='tight')
plt.savefig(Path('OutputFigures') / 'globe.svg', dpi=300, transparent=False, bbox_inches='tight')
plt.close()

fig, axs = plt.subplots(1, 1)
fig.set_size_inches(12, 4)

#plt.fill_between(glosat_time, glosat - glosat_unc, glosat + glosat_unc, color="#1f77b4", alpha=0.2)
plt.fill_between(hadcrut_time, hadcrut_low, hadcrut_high, color="#ff7f0e", alpha=0.1)
plt.plot(hadcrut_time, hadcrut_mean, color="#ff7f0e", linewidth=0.5)
#plt.plot(glosat_time, glosat, color="#1f77b4", linewidth=0.5)

#axs.spines['bottom'].set_position('zero')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_xlim([datetime.date(1870, 1, 1), datetime.date(2024, 12, 31)])
axs.xaxis.set_major_locator(mdates.YearLocator(base=10))
axs.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

volcanoes = {
    'pinatubo': [datetime.date(1991, 6, 12), datetime.date(1992, 6, 12)],
    'chichon': [datetime.date(1982, 3, 29), datetime.date(1983, 3, 29)],
    'agung': [datetime.date(1963, 2, 18), datetime.date(1964, 2, 18)],
    'krakatoa': [datetime.date(1883, 8, 26), datetime.date(1884, 8, 26)],
    'santamaria': [datetime.date(1902, 10, 24), datetime.date(1903, 10, 24)],
}

for key in volcanoes:
    plt.fill_between(volcanoes[key], [-4, -4], [4, 4], color="green", alpha=0.1)

plt.gca().set_ylim(-3.1, 3.1)

plt.savefig(Path('OutputFigures') / 'enso.png', dpi=300, transparent=False, bbox_inches='tight')
plt.savefig(Path('OutputFigures') / 'enso.svg', dpi=300, transparent=False, bbox_inches='tight')
plt.close()
