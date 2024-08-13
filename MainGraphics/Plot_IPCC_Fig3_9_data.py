import xarray as xa
import matplotlib.pyplot as plt
import matplotlib as mpl

# Data from https://data.ceda.ac.uk/badc/ar6_wg1/data/ch_03/ch3_fig09/v20211028
ds = xa.open_dataset('fig_3_9_b.nc')

data_array = ds.tas.data

time = ds.time.data

colors = ['#B47202', 'darkgreen', 'grey', 'dodgerblue', 'black']

hfont = {'fontsize': 32}

fig, axs = plt.subplots(1, 1)
fig.set_size_inches(18, 9)

for reg in range(5):
    plt.fill_between(time, data_array[reg, 1, :], data_array[reg, 2, :], color=colors[reg], alpha=0.5, edgecolor=None)

for reg in range(5):
    plt.plot(time, data_array[reg, 0, :], color=colors[reg], linewidth=5)

axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_xlim(1850, 2021)

plt.gca().set_ylim(-1.3, 2.5)
plt.gca().set_title("Global mean temperature 1850-2020", pad=5, loc='left', **hfont)
axs.tick_params(axis='both', which='major', labelsize=20)

plt.savefig('ipcc_3_9.svg', bbox_inches='tight')
