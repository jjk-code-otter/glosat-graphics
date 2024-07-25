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
hadcrut_dir = Path(data_dir_env) / 'GloSAT' / 'hadcrut5'

n_ensemble = 200

glosat = np.zeros((2892, n_ensemble))
hadcrut = np.zeros((2092, n_ensemble))

for i in range(1, n_ensemble+1):
    filename = f'GloSATref.1.0.0.0.analysis.anomalies.{i}.nc'
    print(filename)
    ds = xa.open_dataset(glosat_dir / filename)
    data = ds.tas
    latsr = xa.ufuncs.deg2rad(data.latitude)
    weights = xa.ufuncs.cos(latsr)
    weighted_mean = data.weighted(weights).mean(dim=("latitude", "longitude"))
    df = weighted_mean.to_dataframe(name='tas')
    df = df.rolling(window=12).mean()

    glosat[:, i-1] = df.tas.array[:]
    glosat_time = df.index.array

    # plt.plot(df['tas'], color='black', alpha=0.2)

    filename = f'HadCRUT.5.0.2.0.analysis.anomalies.{i}.nc'
    print(filename)
    ds = xa.open_dataset(hadcrut_dir / filename)
    data = ds.tas
    latsr = xa.ufuncs.deg2rad(data.latitude)
    weights = xa.ufuncs.cos(latsr)
    weighted_mean = data.weighted(weights).mean(dim=("latitude", "longitude"))
    df = weighted_mean.to_dataframe(name='tas')
    df = df.rolling(window=12).mean()

    hadcrut[:, i-1] = df.tas.array[:]
    hadcrut_time = df.index.array

    # plt.plot(df['tas'], color='red', alpha=0.2)

#
# plt.show()
# plt.close()

fig, axs = plt.subplots(1, 1)
fig.set_size_inches(12, 6)

summary_glosat = np.zeros((2892,3))
summary_glosat[:,0] = np.mean(glosat, axis=1)
summary_glosat[:,1] = np.quantile(glosat, 0.01, axis=1)
summary_glosat[:,2] = np.quantile(glosat, 0.99, axis=1)

summary_hadcrut = np.zeros((2092,3))
summary_hadcrut[:,0] = np.mean(hadcrut, axis=1)
summary_hadcrut[:,1] = np.quantile(hadcrut, 0.01, axis=1)
summary_hadcrut[:,2] = np.quantile(hadcrut, 0.99, axis=1)

axs.fill_between(glosat_time, summary_glosat[:,1], summary_glosat[:,2],alpha=0.5,facecolor='#ab4be3',edgecolor=None)
axs.plot(glosat_time, summary_glosat[:,0],color='#ab4be3',linewidth=1)

axs.fill_between(hadcrut_time, summary_hadcrut[:,1], summary_hadcrut[:,2],alpha=0.5,facecolor='#fcba03',edgecolor=None)
axs.plot(hadcrut_time, summary_hadcrut[:,0],color='#fcba03',linewidth=1)

#axs.spines['bottom'].set_position('zero')
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_xlim([datetime.date(1780, 1, 1), datetime.date(2024, 12, 31)])
axs.xaxis.set_major_locator(mdates.YearLocator(base=20))
axs.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.gca().set_ylim(-2.4, 1.4)

plt.savefig('long_time_series.png', bbox_inches='tight', dpi=300)
plt.close()