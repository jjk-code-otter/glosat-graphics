import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig = plt.figure(figsize=(20, 10))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
# Pacific Ocean
ax.set_extent([110, 300, -60, 60], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor='0.5')
ax.coastlines()
plt.savefig('pacific_ocean.png', bbox_inches='tight', dpi=300)
plt.close()