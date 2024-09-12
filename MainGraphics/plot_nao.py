import copy

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# Read NAO
years = []
months = []
nao = []

with open('InputData/nao_3dp.dat.txt', 'r') as f:
    for line in f:
        columns = line.split()
        year = int(columns[0])

        for i in range(1, 13):
            if year >= 1825:
                years.append(year)
                months.append(i)
                nao.append(float(columns[i]))

years = np.array(years)
months = np.array(months)
nao = np.array(nao)

years = years[nao != -99.99]
months = months[nao != -99.99]
nao = nao[nao != -99.99]

# Calculate seasonal averages
nao_3month = copy.deepcopy(nao)
nao_3month[:] = np.nan

for i in range(2, len(nao)):
    select = nao[i - 2:i + 1]
    nao_3month[i] = np.mean(select)

time = years + (months - 1) / 12.

# Select winter, DJF
sub_years = years[months == 2]
sub_nao = nao_3month[months == 2]
sub_month = months[months == 2]

# Plot NAO
fig, ax = plt.subplots(figsize=(16, 5))

# Stripes every 25 years
for i in range(4):
    ax.add_patch(
        Rectangle((1825+i*50, -3.5), 25, 7, color='lightgrey', alpha=0.5)
    )

# Stripes every 5 years
# for i in range(20):
#     ax.add_patch(
#         Rectangle((1825+i*10, -3.5), 5, 7, color='lightgrey', alpha=0.5)
#     )

for i in range(len(sub_years)):
    col = 'red'
    if sub_nao[i] < 0:
        col = 'blue'

    delta = 0.1

    ax.add_patch(
        Rectangle((sub_years[i] + delta, 0), 1-2*delta, sub_nao[i], facecolor=col, edgecolor='black')
    )
ax.set_xlim(1825,2025)
ax.set_ylim(-3.5,3.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title("Winter North Atlantic Oscillation 1826-2024", pad=5, fontdict={'fontsize': 20}, loc='left', color='black')

plt.plot(sub_years, sub_nao, linewidth=0)
plt.savefig('OutputFigures/nao.svg', transparent=True, bbox_inches='tight')
plt.savefig('OutputFigures/nao.png', transparent=True, bbox_inches='tight', dpi=1200)
plt.close()
