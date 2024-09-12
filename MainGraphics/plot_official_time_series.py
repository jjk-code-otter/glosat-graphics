"""
Plot global series from GloSAT and HadCRUT together with the model data from the GloSAT papers.
"""

import xarray as xa
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.dates as mdates

data_dir_env = os.getenv('DATADIR')
glosat_dir = Path(data_dir_env) / 'GloSAT' / 'analysis' / 'diagnostics'
hadcrut_dir = Path(data_dir_env) / 'ManagedData' / 'Data' / 'HadCRUT5'

climatology = [1850, 1900]

n_ensemble = 200

glosat = np.zeros((2892, n_ensemble))
hadcrut = np.zeros((2092, n_ensemble))

# Read in the GLOSAT pre-calculated time series
filename = f'GloSATref.1.0.0.0.analysis.component_series.global.monthly.nc'
ds = xa.open_dataset(glosat_dir / filename)
data = ds.tas_mean
uncertainty = ds.tas_total_unc

df_data = data.to_dataframe(name='tas')
df_data = df_data.rolling(window=12).mean()

df_uncertainty = uncertainty.to_dataframe(name='tas_total_unc')
df_uncertainty = df_uncertainty.rolling(window=12).mean()

glosat = df_data.tas.array[:]
glosat_unc = df_uncertainty.tas_total_unc.array[:]
glosat_time = df_data.index.array

# Read in the HadCRUT pre-calculated time series
filename = 'HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.nc'
ds = xa.open_dataset(hadcrut_dir / filename)
data = ds.tas_mean
uncertainty_high = ds.tas_upper
uncertainty_low = ds.tas_lower

df_data = data.to_dataframe(name='tas')
df_data = df_data.rolling(window=12).mean()

df_uncertainty_high = uncertainty_high.to_dataframe(name='tas_upper')
df_uncertainty_high = df_uncertainty_high.rolling(window=12).mean()

df_uncertainty_low = uncertainty_low.to_dataframe(name='tas_lower')
df_uncertainty_low = df_uncertainty_low.rolling(window=12).mean()

hadcrut = df_data.tas.array[:]
hadcrut_unc_lower = df_uncertainty_low.tas_lower.array[:]
hadcrut_unc_upper = df_uncertainty_high.tas_upper.array[:]
hadcrut_time = df_data.index.array

# Load the model data (see calculate_time_series.py for actual calculations)
summary_ukesm_model = np.load(f'ukesm_model_summary_{climatology[0]}-{climatology[1]}.npy', allow_pickle=True)
ukesm_model_time = np.load(f'ukesm_model_time_{climatology[0]}-{climatology[1]}.npy', allow_pickle=True)
summary_model = np.load(f'model_summary_{climatology[0]}-{climatology[1]}.npy', allow_pickle=True)
model_time = np.load(f'model_time_{climatology[0]}-{climatology[1]}.npy', allow_pickle=True)
particle_time = np.load(f'particle_time_{climatology[0]}-{climatology[1]}.npy', allow_pickle=True)
particle = np.load(f'particle_{climatology[0]}-{climatology[1]}.npy', allow_pickle=True)

hadcrut_year = np.array([x.year for x in hadcrut_time])
glosat_year = np.array([x.year for x in glosat_time])
hadcrut_select = (hadcrut_year >= climatology[0]) & (hadcrut_year <= climatology[1])
glosat_select = (glosat_year >= climatology[0]) & (glosat_year <= climatology[1])

volcanoes = {
    'pinatubo': [datetime.date(1991, 6, 12), datetime.date(1991, 6, 12)],
    'chichon': [datetime.date(1982, 3, 29), datetime.date(1982, 3, 29)],
    'agung': [datetime.date(1963, 2, 18), datetime.date(1963, 2, 18)],
    'krakatoa': [datetime.date(1883, 8, 26), datetime.date(1883, 8, 26)],
    'santamaria': [datetime.date(1902, 10, 24), datetime.date(1902, 10, 24)],
    'hthh': [datetime.date(2022, 1, 15), datetime.date(2022, 1, 15)],
    'consiguina': [datetime.date(1835, 1, 20), datetime.date(1835, 1, 20)],
    'galunggung': [datetime.date(1822, 10, 8), datetime.date(1822, 10, 8)],
    'tambora': [datetime.date(1815, 4, 10), datetime.date(1815, 4, 10)],
    'unknown': [datetime.date(1808, 1, 1), datetime.date(1808, 1, 1)]
}

fig, axs = plt.subplots(1, 1)
fig.set_size_inches(18, 6)

for volcano in volcanoes:
    plt.plot(volcanoes[volcano], [-1.6, 1.3], color='#555555')

axs.fill_between(hadcrut_time, hadcrut_unc_lower, hadcrut_unc_upper, alpha=0.5, facecolor='#ffe100', edgecolor=None)
axs.plot(hadcrut_time, hadcrut, color='#ffe100', linewidth=1)

axs.fill_between(ukesm_model_time, summary_ukesm_model[:, 1], summary_ukesm_model[:, 2], alpha=0.5, facecolor='#55ff55',
                 edgecolor=None)
axs.plot(ukesm_model_time, summary_ukesm_model[:, 0], color='#55ff55', linewidth=1)
axs.fill_between(model_time, summary_model[:, 1], summary_model[:, 2], alpha=0.5, facecolor='#aaaaaa', edgecolor=None)
axs.plot(model_time, summary_model[:, 0], color='#555555', linewidth=1)
axs.plot(particle_time, particle, '--', color='#555555', linewidth=1, )

axs.fill_between(glosat_time, glosat - glosat_unc, glosat + glosat_unc, alpha=0.5, facecolor='#ab4be3', edgecolor=None)
axs.plot(glosat_time, glosat, color='#ab4be3', linewidth=1)

# axs.spines['bottom'].set_position('zero')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_xlim([datetime.date(1750, 1, 1), datetime.date(2024, 12, 31)])
axs.xaxis.set_major_locator(mdates.YearLocator(base=20))
axs.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.gca().set_ylim(-1.6, 1.3)
plt.gca().set_title("Global mean surface air temperature 1790-2021", pad=5, fontdict={'fontsize': 20}, loc='left')

plt.savefig(Path('OutputFigures') / 'long_time_series_official.png', bbox_inches='tight', dpi=600, transparent=True)
plt.savefig(Path('OutputFigures') / 'long_time_series_official.svg', bbox_inches='tight', transparent=True)
plt.close()

# Now repeat vs 1850-1900
fig, axs = plt.subplots(1, 1)
fig.set_size_inches(18, 6)

for volcano in volcanoes:
    plt.plot(volcanoes[volcano], [-1.6, 1.3], color='#555555')

axs.fill_between(
    hadcrut_time,
    hadcrut_unc_lower - np.mean(hadcrut[hadcrut_select]),
    hadcrut_unc_upper - np.mean(hadcrut[hadcrut_select]),
    alpha=0.5, facecolor='#ffe100', edgecolor=None
)
axs.plot(hadcrut_time, hadcrut - np.mean(hadcrut[hadcrut_select]), color='#ffe100', linewidth=1)

axs.fill_between(
    ukesm_model_time,
    summary_ukesm_model[:, 1],
    summary_ukesm_model[:, 2],
    alpha=0.5, facecolor='#55ff55', edgecolor=None
)
axs.plot(ukesm_model_time, summary_ukesm_model[:, 0],
         color='#55ff55', linewidth=1)

axs.fill_between(
    model_time,
    summary_model[:, 1] ,
    summary_model[:, 2] ,
    alpha=0.5, facecolor='#aaaaaa', edgecolor=None
)
axs.plot(model_time, summary_model[:, 0] , color='#555555', linewidth=1)

#axs.plot(particle_time, particle, '--', color='#555555', linewidth=1, )

glosat_low = glosat - glosat_unc
glosat_high = glosat + glosat_unc

axs.fill_between(
    glosat_time,
    glosat_low - np.mean(glosat[glosat_select]),
    glosat_high - np.mean(glosat[glosat_select]),
    alpha=0.5, facecolor='#ab4be3', edgecolor=None
)
axs.plot(glosat_time, glosat - np.mean(glosat[glosat_select]), color='#ab4be3', linewidth=1)

# axs.spines['bottom'].set_position('zero')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_xlim([datetime.date(1790, 1, 1), datetime.date(2024, 12, 31)])
axs.xaxis.set_major_locator(mdates.YearLocator(base=20))
axs.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.gca().set_ylim(-1.1, 1.6)
plt.gca().set_title("Global mean surface air temperature 1790-2023", pad=5, fontdict={'fontsize': 20}, loc='left')

plt.savefig(Path('OutputFigures') / 'long_time_series_official_1850_1900.png', bbox_inches='tight', dpi=600, transparent=True)
plt.savefig(Path('OutputFigures') / 'long_time_series_official_1850_1900.svg', bbox_inches='tight', transparent=True)
plt.close()
