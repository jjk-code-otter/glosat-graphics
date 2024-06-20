import xarray as xa
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

data_dir_env = os.getenv('DATADIR')
glosat_dir = Path(data_dir_env) / 'GloSAT' / 'analysis' / 'diagnostics'

filename = 'GloSATref.1.0.0.0.noninfilled.anomalies.ensemble_median.nc'
filename = 'GloSATref.1.0.0.0.analysis.anomalies.ensemble_median.nc'

print(filename)

ds = xa.open_dataset(glosat_dir / filename)

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
    selection = ~np.isnan([data[i, :, :]])
    total_area_globe = np.sum(weights[selection[0, :, :]])
    coverage[i, 0] = total_area_globe

    selection = ~np.isnan([data[i, :, :]]) & (latsr >= 65.0)
    total_area_arctic = np.sum(weights[selection[0, :, :]])
    coverage[i, 1] = total_area_arctic

    selection = ~np.isnan([data[i, :, :]]) & (latsr < 65.0) & (latsr >= 30.0)
    total_area_nextra = np.sum(weights[selection[0, :, :]])
    coverage[i, 2] = total_area_nextra

    selection = ~np.isnan([data[i, :, :]]) & (latsr < 30.0) & (latsr >= 0)
    total_area_ntropics = np.sum(weights[selection[0, :, :]])
    coverage[i, 3] = total_area_ntropics

    selection = ~np.isnan([data[i, :, :]]) & (latsr < 0.0) & (latsr >= -30)
    total_area_stropics = np.sum(weights[selection[0, :, :]])
    coverage[i, 4] = total_area_stropics

    selection = ~np.isnan([data[i, :, :]]) & (latsr >= -65.0) & (latsr < -30)
    total_area_sextra = np.sum(weights[selection[0, :, :]])
    coverage[i, 5] = total_area_sextra

    selection = ~np.isnan([data[i, :, :]]) & (latsr < -65.0)
    total_area_antarctic = np.sum(weights[selection[0, :, :]])
    coverage[i, 6] = total_area_antarctic

for i in range(11, ntime):
    for j in range(7):
        smooth_coverage[i, j] = np.mean(coverage[i - 11:i + 1, j])

spole = '#58c9db'
npole = '#58c9db'
sh_extra = '#8ebf84'
nh_extra = '#8ebf84'
tropics = '#ffa473'

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

plt.plot(time, np.zeros(ntime) + polar_area - 0.50, color='black', alpha=0.5)
plt.plot(time, np.zeros(ntime) + 0.50, color='black', alpha=0.5)
plt.plot(time, np.zeros(ntime) + 0.25, color='black', alpha=0.5)
plt.plot(time, np.zeros(ntime) + 0.00, color='black', alpha=0.5)
plt.plot(time, np.zeros(ntime) - 0.25, color='black', alpha=0.5)
plt.plot(time, np.zeros(ntime) - 0.50, color='black', alpha=0.5)
plt.plot(time, np.zeros(ntime) - polar_area + 0.50, color='black', alpha=0.5)

plt.gca().tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                      labelbottom=False, labeltop=False, labelleft=False, labelright=False)
plt.gca().set_axis_off()

plt.savefig('coverage_centred.png', dpi=300)
plt.savefig('coverage_centred.svg', dpi=300)
plt.savefig('coverage_centred.pdf', dpi=300)
plt.close()

fig, axs = plt.subplots(1, 1)
fig.set_size_inches(12, 3)

plt.fill_between(time, np.zeros(ntime) - polar_area + 0.50, np.zeros(ntime) - polar_area + 0.50 - smooth_coverage[:, 6], color=spole)
plt.fill_between(time, np.zeros(ntime) - 0.25, np.zeros(ntime) - 0.25 - smooth_coverage[:, 5], color=sh_extra)
plt.fill_between(time, np.zeros(ntime), 0.0 - smooth_coverage[:, 4], color=tropics)

plt.fill_between(time, np.zeros(ntime) + polar_area - 0.5, np.zeros(ntime) + polar_area - 0.5 + smooth_coverage[:, 1], color=npole)
plt.fill_between(time, np.zeros(ntime) + 0.25, np.zeros(ntime) + 0.25 + smooth_coverage[:, 2], color=nh_extra)
plt.fill_between(time, np.zeros(ntime), smooth_coverage[:, 3], color=tropics)

plt.plot(time, np.zeros(ntime) + polar_area - 0.50, color='black', alpha=0.5)
plt.plot(time, np.zeros(ntime) + 0.50, color='black', alpha=0.5)
plt.plot(time, np.zeros(ntime) + 0.25, color='black', alpha=0.5)
plt.plot(time, np.zeros(ntime) + 0.00, color='black', alpha=0.5)
plt.plot(time, np.zeros(ntime) - 0.25, color='black', alpha=0.5)
plt.plot(time, np.zeros(ntime) - 0.50, color='black', alpha=0.5)
plt.plot(time, np.zeros(ntime) - polar_area + 0.50, color='black', alpha=0.5)

plt.gca().tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                      labelbottom=False, labeltop=False, labelleft=False, labelright=False)
plt.gca().set_axis_off()

plt.savefig('coverage_individual.png', dpi=300)
plt.savefig('coverage_individual.svg', dpi=300)
plt.savefig('coverage_individual.pdf', dpi=300)
plt.close()
