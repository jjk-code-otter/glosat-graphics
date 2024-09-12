"""
Calculate global series from the ensemble members of GloSAT and HadCRUT as well as the HadCM3 runs from
Schurer et al.
"""

import xarray as xa
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.dates as mdates


def calculate_global_mean(ds, climatology):
    #Extract tas
    data = ds.tas

    if data.latitude.values[0] == 90.:
        nh = data.sel(latitude=slice(90, 0))
        sh = data.sel(latitude=slice(0, -90))
    else:
        nh = data.sel(latitude=slice(0, 90))
        sh = data.sel(latitude=slice(-90, 0))

    weights = np.cos(np.deg2rad(nh.latitude))
    nh_weighted_mean = nh.weighted(weights).mean(("longitude", "latitude"))

    weights = np.cos(np.deg2rad(sh.latitude))
    sh_weighted_mean = sh.weighted(weights).mean(("longitude", "latitude"))

    global_weighted_mean = 0.5 * nh_weighted_mean + 0.5 * sh_weighted_mean

    # Calculate area weights abd take the weighted mean
    latsr = xa.ufuncs.deg2rad(data.latitude)
    weights = xa.ufuncs.cos(latsr)
    weighted_mean = data.weighted(weights).mean(dim=("latitude", "longitude"))
    df = weighted_mean.to_dataframe(name='tas')
    df.tas = global_weighted_mean

    # Calculate anomalies
    time = df.index.array
    years = np.array([x.year for x in time])
    climatological_average = np.mean(df.tas.array[(years >= climatology[0]) & (years <= climatology[1])])
    df.tas = df.tas - climatological_average

    df = df.rolling(window=12).mean()

    return df, time


data_dir_env = os.getenv('DATADIR')
glosat_dir = Path(data_dir_env) / 'GloSAT' / 'glosatref1000'
hadcrut_dir = Path(data_dir_env) / 'GloSAT' / 'hadcrut5'
model_dir = Path(data_dir_env) / 'Model'
glosat_model_dir = Path(data_dir_env) / 'GloSAT' / 'model'

climatology = [1850, 1900]
#climatology = [1961, 1990]

n_ukesm = 6
ukesm_models = np.zeros((12 * (2014 - 1750 + 1), n_ukesm))
for i in range(1, n_ukesm + 1):
    filename = f'tas_historical1750_UKESM1-1LL_r{i}i1p1_175001-201412.nc'
    ds = xa.open_dataset(glosat_model_dir / filename)
    df, ukesm_model_time = calculate_global_mean(ds, climatology)
    ukesm_models[:, i - 1] = df.tas.array[:]

particle = np.zeros((2736))
filename = 'tas_Amon_HadCM3_DataAssimilationMean_r1i1p1_17812008.nc'
ds = xa.open_dataset(model_dir / filename)
df, particle_time = calculate_global_mean(ds, climatology)
particle[:] = df.tas.array[:]

n_model = 10
models = np.zeros((2749, n_model))
for i in range(1, n_model + 1):
    filename = f'tas_Amon_HadCM3_FreeRunning_r{i}i1p1_178012-200912.nc'
    ds = xa.open_dataset(model_dir / filename)
    df, model_time = calculate_global_mean(ds, climatology)
    models[:, i - 1] = df.tas.array[:]

n_ensemble = 200

glosat = np.zeros((2892, n_ensemble))
hadcrut = np.zeros((2092, n_ensemble))

for i in range(1, n_ensemble + 1):
    filename = f'GloSATref.1.0.0.0.analysis.anomalies.{i}.nc'
    print(filename)
    ds = xa.open_dataset(glosat_dir / filename)
    df, glosat_time = calculate_global_mean(ds, climatology)
    glosat[:, i - 1] = df.tas.array[:]

    filename = f'HadCRUT.5.0.2.0.analysis.anomalies.{i}.nc'
    print(filename)
    ds = xa.open_dataset(hadcrut_dir / filename)
    df, hadcrut_time = calculate_global_mean(ds, climatology)
    hadcrut[:, i - 1] = df.tas.array[:]

fig, axs = plt.subplots(1, 1)
fig.set_size_inches(12, 6)

summary_model = np.zeros((2749, 3))
summary_model[:, 0] = np.mean(models, axis=1)
summary_model[:, 1] = np.min(models, axis=1)
summary_model[:, 2] = np.max(models, axis=1)

summary_ukesm_model = np.zeros((3180, 3))
summary_ukesm_model[:, 0] = np.mean(ukesm_models, axis=1)
summary_ukesm_model[:, 1] = np.min(ukesm_models, axis=1)
summary_ukesm_model[:, 2] = np.max(ukesm_models, axis=1)

np.save(f'ukesm_model_summary_{climatology[0]}-{climatology[1]}.npy', summary_ukesm_model)
np.save(f'model_summary_{climatology[0]}-{climatology[1]}.npy', summary_model)
np.save(f'ukesm_model_time_{climatology[0]}-{climatology[1]}.npy', ukesm_model_time)
np.save(f'model_time_{climatology[0]}-{climatology[1]}.npy', model_time)
np.save(f'particle_time_{climatology[0]}-{climatology[1]}.npy', particle_time)
np.save(f'particle_{climatology[0]}-{climatology[1]}.npy', particle)

summary_glosat = np.zeros((2892, 3))
summary_glosat[:, 0] = np.mean(glosat, axis=1)
summary_glosat[:, 1] = np.quantile(glosat, 0.01, axis=1)
summary_glosat[:, 2] = np.quantile(glosat, 0.99, axis=1)

summary_hadcrut = np.zeros((2092, 3))
summary_hadcrut[:, 0] = np.mean(hadcrut, axis=1)
summary_hadcrut[:, 1] = np.quantile(hadcrut, 0.01, axis=1)
summary_hadcrut[:, 2] = np.quantile(hadcrut, 0.99, axis=1)

axs.fill_between(glosat_time, summary_glosat[:, 1], summary_glosat[:, 2], alpha=0.5, facecolor='#ab4be3',
                 edgecolor=None)
axs.plot(glosat_time, summary_glosat[:, 0], color='#ab4be3', linewidth=1)

axs.fill_between(ukesm_model_time, summary_ukesm_model[:, 1], summary_ukesm_model[:, 2], alpha=0.5, facecolor='#aaffaa',
                 edgecolor=None)
axs.plot(ukesm_model_time, summary_ukesm_model[:, 0], color='#55ff55', linewidth=3)

axs.fill_between(model_time, summary_model[:, 1], summary_model[:, 2], alpha=0.5, facecolor='#aaaaaa', edgecolor=None)
axs.plot(model_time, summary_model[:, 0], color='#555555', linewidth=1)

axs.plot(particle_time, particle, '--', color='#555555', linewidth=1, )

axs.fill_between(hadcrut_time, summary_hadcrut[:, 1], summary_hadcrut[:, 2], alpha=0.5, facecolor='#fcba03',
                 edgecolor=None)
axs.plot(hadcrut_time, summary_hadcrut[:, 0], color='#fcba03', linewidth=1)

# axs.spines['bottom'].set_position('zero')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_xlim([datetime.date(1780, 1, 1), datetime.date(2024, 12, 31)])
axs.xaxis.set_major_locator(mdates.YearLocator(base=20))
axs.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.gca().set_ylim(-2.4, 1.4)

plt.savefig(Path('OutputFigures') / f'long_time_series_{climatology[0]}-{climatology[1]}.png', bbox_inches='tight', dpi=300)
plt.savefig(Path('OutputFigures') / f'long_time_series_{climatology[0]}-{climatology[1]}.svg', bbox_inches='tight')
plt.close()
