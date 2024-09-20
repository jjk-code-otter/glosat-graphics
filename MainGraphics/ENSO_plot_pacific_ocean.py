import xarray as xr
import numpy as np
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from pathlib import Path

if __name__ == '__main__':
    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    # Pacific Ocean
    ax.set_extent([110, 300, -60, 60], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='0.5')
    ax.coastlines(linewidth=3)
    plt.savefig(Path('OutputFigures') / 'pacific_ocean.png', bbox_inches='tight', dpi=1200)
    plt.close()

    data_dir_env = Path(os.getenv('DATADIR'))

    # From https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz
    # Load the HadISST NetCDF dataset
    file_path = data_dir_env / 'ManagedData' / 'Data' / 'HadISST' / 'HadISST_sst.nc'
    ds = xr.open_dataset(file_path)

    # Select the SST variable
    sst = ds['sst']
    sst = sst.where((sst != 1000) & (sst != -1000), np.nan)
    sst = sst.sel(time=slice('1971-01-01', '2024-12-31'))

    # Define the time period for the climatology (1991 to 2020)
    climatology_period = sst.sel(time=slice('1991-01-01', '2020-12-31'))

    # Calculate the climatology for the period 1991-2020
    climatology = climatology_period.groupby('time.month').mean('time')

    # Calculate the SST anomaly relative to the 1991-2020 climatology
    sst_anomaly = sst.groupby('time.month') - climatology

    # Define the El Niño 3.4 region (5N-5S, 170W-120W)
    el_nino_region = sst_anomaly.sel(latitude=slice(5, -5), longitude=slice(-170, -120))

    # Calculate the Niño 3.4 index by averaging SST anomalies over the El Niño 3.4 region
    nino34_index = el_nino_region.mean(dim=['latitude', 'longitude'], skipna=True)

    # Identify El Niño events (using a threshold of +0.5°C for at least 5 consecutive months)
    el_nino_events = nino34_index.rolling(time=5, center=True).mean() > 0.5

    # Create a composite of El Niño events
    el_nino_composite = sst_anomaly.where(el_nino_events).mean('time')

    # Identify El Niño events (using a threshold of +0.5°C for at least 5 consecutive months)
    la_nina_events = nino34_index.rolling(time=5, center=True).mean() < -0.5

    # Create a composite of El Niño events
    la_nina_composite = sst_anomaly.where(la_nina_events).mean('time')

    # Plot the composite map
    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    lon, lat = np.meshgrid(el_nino_composite.longitude, el_nino_composite.latitude)
    pcm = ax.pcolormesh(lon, lat, el_nino_composite, transform=ccrs.PlateCarree(),
                        cmap='RdBu_r', shading='auto', vmin=-1.5, vmax=1.5)

    ax.set_extent([110, 300, -60, 60], crs=ccrs.PlateCarree())
    ax.coastlines(zorder=2, linewidth=3)
    ax.add_feature(cfeature.LAND, facecolor='0.5', zorder=1)  # Redraw land on top
    plt.savefig(Path('OutputFigures') / 'pacific_ocean_nino_composite.png', bbox_inches='tight', dpi=1200)
    plt.close()

    # Plot the composite map
    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    lon, lat = np.meshgrid(el_nino_composite.longitude, el_nino_composite.latitude)
    pcm = ax.pcolormesh(lon, lat, la_nina_composite, transform=ccrs.PlateCarree(),
                        cmap='RdBu_r', shading='auto', vmin=-1.5, vmax=1.5)

    ax.set_extent([110, 300, -60, 60], crs=ccrs.PlateCarree())
    ax.coastlines(zorder=2, linewidth=3)
    ax.add_feature(cfeature.LAND, facecolor='0.5', zorder=1)  # Redraw land on top
    plt.savefig(Path('OutputFigures') / 'pacific_ocean_nina_composite.png', bbox_inches='tight', dpi=1200)
    plt.close()
