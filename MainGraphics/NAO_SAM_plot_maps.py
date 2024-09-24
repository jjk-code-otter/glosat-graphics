"""
Plot some land/ocean maps using different projections
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

if __name__ == '__main__':
    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes(projection=ccrs.NearsidePerspective(central_longitude=10, central_latitude=45,
                                                      satellite_height=3578583100))
    ax.add_feature(cfeature.LAND, facecolor='#9DC887')
    ax.add_feature(cfeature.OCEAN, facecolor='#8AC3DC')
    ax.coastlines(linewidth=3, color='#6E7E67')
    plt.savefig(Path('OutputFigures') / 'nao_map.svg', bbox_inches='tight', transparent=True)
    plt.close()


    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes(projection=ccrs.NearsidePerspective(central_longitude=0, central_latitude=-90,
                                                      satellite_height=3578583100))
    ax.add_feature(cfeature.LAND, facecolor='#9DC887')
    ax.add_feature(cfeature.OCEAN, facecolor='#8AC3DC')
    ax.coastlines(linewidth=3, color='#6E7E67')
    plt.savefig(Path('OutputFigures') / 'sam_map.svg', bbox_inches='tight', transparent=True)
    plt.close()


    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))
    ax.add_feature(cfeature.LAND, facecolor='#9DC887')
    ax.add_feature(cfeature.OCEAN, facecolor='#8AC3DC')
    ax.coastlines(linewidth=3, color='#6E7E67')
    plt.savefig(Path('OutputFigures') / 'base_map.svg', bbox_inches='tight', transparent=True)
    plt.close()


