import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import seaborn as sns
import numpy as np

STANDARD_PARAMETER_SET = {
    'axes.axisbelow': False,
    'axes.labelsize': 23,
    'xtick.labelsize': 23,
    'ytick.labelsize': 23,
    'axes.edgecolor': 'lightgrey',
    'axes.facecolor': 'None',

    'axes.grid.axis': 'y',
    'grid.color': 'lightgrey',
    'grid.alpha': 0.5,

    'axes.labelcolor': 'dimgrey',

    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,

    'figure.facecolor': 'white',
    'lines.solid_capstyle': 'round',
    'patch.edgecolor': 'w',
    'patch.force_edgecolor': True,
    'text.color': 'dimgrey',

    'xtick.bottom': True,
    'xtick.color': 'dimgrey',
    'xtick.direction': 'out',
    'xtick.top': False,
    'xtick.labelbottom': True,

    'ytick.major.width': 0.4,
    'ytick.color': 'dimgrey',
    'ytick.direction': 'out',
    'ytick.left': False,
    'ytick.right': False
}


def annoying(instr):
    if instr == '':
        return None
    else:
        return float(instr)


years = []
berkeley = []
era5 = []
gistemp = []
hadcrut = []
jra55 = []
noaa = []

with open('InputData/tas_summary.csv', 'r') as f:
    for i in range(79):
        f.readline()
    for line in f:

        if 'end data' in line:
            break

        columns = line.split(',')

        years.append(int(columns[1]))
        berkeley.append(annoying(columns[2]))
        era5.append(annoying(columns[3]))
        gistemp.append(annoying(columns[4]))
        hadcrut.append(annoying(columns[5]))
        jra55.append(annoying(columns[6]))
        noaa.append(annoying(columns[7]))


sns.set(font='Franklin Gothic Book', rc=STANDARD_PARAMETER_SET)

fig, axs = plt.subplots(1, 1)
fig.set_size_inches(16*1.25, 5*1.5)

axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
#axs.set_title("")
axs.tick_params(axis='both', which='major', labelsize=15)

axs.set_ylim(-0.4, 1.5)
axs.set_xlim(1775, 2025)

limits = [-0.4, 1.5]
spacing = 0.2
lo = spacing * (1 + (limits[0] // spacing))
hi = spacing * (1 + (limits[1] // spacing))
ticks = np.arange(lo, hi, spacing)
plt.yticks(ticks)

limits = [1775, 2025]
spacing = 20
lo = spacing * (1 + (limits[0] // spacing))
hi = spacing * (1 + (limits[1] // spacing))
ticks = np.arange(lo, hi, spacing)
plt.xticks(ticks)

plt.plot(years, berkeley, linewidth=3)
plt.plot(years, era5, linewidth=3)
plt.plot(years, gistemp, linewidth=3)
plt.plot(years, hadcrut, linewidth=3)
plt.plot(years, jra55, linewidth=3)
plt.plot(years, noaa, linewidth=3)

plt.savefig('OutputFigures/gmt.svg', bbox_inches='tight', transparent=True)
plt.close()
