import copy

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

years = []
sam_recon = []
sam_unc = []
sam_marshall = []

with open('InputData/abram2014sam-noaa.txt', 'r') as f:
    for i in range(112):
        f.readline()
    for line in f:
        columns = line.split()

        years.append(int(columns[0]))
        sam_recon.append(float(columns[1]))
        sam_unc.append(float(columns[2]))

        if columns[8] == 'NaN':
            sam_marshall.append(None)
        else:
            sam_marshall.append(float(columns[8]))


years = np.array(years)
sam_recon = np.array(sam_recon)
sam_unc = np.array(sam_unc)


bas2_years = []
bas2_sam = []

with open('InputData/newsam.1957.2007.seas.txt', 'r') as f:
    f.readline()
    f.readline()

    for line in f:
        columns = line.split()
        if len(columns) == 6:
            bas2_years.append(int(columns[0]))
            bas2_sam.append(float(columns[1]))

# shift baseline to match
bas2_years = np.array(bas2_years)
bas2_sam = np.array(bas2_sam)
bas2_sam = bas2_sam - np.mean(bas2_sam[(bas2_years >= 1961) & (bas2_years <= 1990)])

# Plot NAO
fig, ax = plt.subplots(figsize=(16, 5))

# Stripes every 25 years
for i in range(4):
    ax.add_patch(
        Rectangle((1825+i*50, -5), 25, 11, color='lightgrey', alpha=0.5)
    )



for i in range(len(sam_recon)):
    if sam_recon[i] is not None:
        col = 'pink'
        col2 = 'red'
        if sam_recon[i] < 0:
            col = 'lightblue'
            col2 = 'blue'

        delta = 0.1

    if years[i] < 1957:
        ax.add_patch(
            Rectangle((years[i] + delta, sam_recon[i]-sam_unc[i]), 1-2*delta, 2*sam_unc[i], facecolor=col, edgecolor=None)
        )

        ax.add_patch(
            Rectangle((years[i] + delta, 0), 1 - 2 * delta, sam_recon[i], facecolor=col2,edgecolor='black')
        )

        #plt.plot([years[i] + delta, years[i] + 1 - delta], [sam_recon[i], sam_recon[i]], color='black')

for i in range(len(bas2_sam)):
    if bas2_sam[i] is not None:
        col = 'red'
        if bas2_sam[i] < 0:
            col = 'blue'

        delta = 0.1

        ax.add_patch(
            Rectangle((bas2_years[i] + delta, 0), 1-2*delta, bas2_sam[i], facecolor=col, edgecolor='black')
        )

#plt.fill_between(years, sam_recon-sam_unc, sam_recon+sam_unc, color='green', alpha=0.5)
#plt.plot(years, sam_recon, color='black')


ax.set_xlim(1825,2025)
ax.set_ylim(-5.5,5.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title("Southern Annular Mode 1826-2024", pad=5, fontdict={'fontsize': 20}, loc='left', color='black')

plt.plot(years, sam_marshall, linewidth=0)
plt.savefig('OutputFigures/sam.svg', transparent=True, bbox_inches='tight')
plt.savefig('OutputFigures/sam.png', transparent=True, bbox_inches='tight', dpi=1200)
plt.close()
