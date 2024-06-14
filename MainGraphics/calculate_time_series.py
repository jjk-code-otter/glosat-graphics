"""
Calculate global series from the ensemble members of GloSAT and HadCRUT
"""

import xarray as xa
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

data_dir_env = os.getenv('DATADIR')
glosat_dir = Path(data_dir_env) / 'GloSAT' / 'glosatref1000'
hadcrut_dir = Path(data_dir_env) / 'GloSAT' / 'hadcrut5'

for i in range(1, 201):
    filename = f'GloSATref.1.0.0.0.analysis.anomalies.{i}.nc'
    print(filename)
    ds = xa.open_dataset(glosat_dir / filename)
    data = ds.tas
    latsr = xa.ufuncs.deg2rad(data.latitude)
    weights = xa.ufuncs.cos(latsr)
    weighted_mean = data.weighted(weights).mean(dim=("latitude", "longitude"))
    df = weighted_mean.to_dataframe(name='tas')
    df = df.rolling(window=12).mean()

    plt.plot(df['tas'], color='black', alpha=0.2)

    filename = f'HadCRUT.5.0.2.0.analysis.anomalies.{i}.nc'
    print(filename)
    ds = xa.open_dataset(hadcrut_dir / filename)
    data = ds.tas
    latsr = xa.ufuncs.deg2rad(data.latitude)
    weights = xa.ufuncs.cos(latsr)
    weighted_mean = data.weighted(weights).mean(dim=("latitude", "longitude"))
    df = weighted_mean.to_dataframe(name='tas')
    df = df.rolling(window=12).mean()

    plt.plot(df['tas'], color='red', alpha=0.2)


plt.show()
