import os
from pathlib import Path
import matplotlib.pyplot as plt

data_dir_env = Path(os.getenv('DATADIR'))

sunspot_file = Path('InputData') / 'GreenwichSSNvstime.txt'

time = []
years = []
months = []
data = []

lookup = {'04': 1, '13': 2, '21': 3, '29': 4, '38': 5, '46': 6, '54': 7, '63': 8, '71': 9, '79': 10, '88': 11,
          '96': 12, }

with open(sunspot_file, 'r') as f:
    for line in f:
        columns = line.split()
        date = columns[0]
        date_columns = date.split('.')
        year = int(date_columns[0])
        value = float(columns[1])
        month = lookup[date_columns[1]]

        time.append(float(columns[0]))
        years.append(year)
        months.append(months)
        data.append(value)

fig, axs = plt.subplots(1, 1)
fig.set_size_inches(18, 6)

axs.plot(time, data, color='#FBD200')

axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_title("Sunspot Number 1749-2004", pad=5, fontdict={'fontsize': 20}, loc='left', color='#ffffff')
axs.tick_params(axis='both', which='major', labelsize=15)

axs.spines['bottom'].set_color('#ffffff')
axs.spines['top'].set_color('#ffffff')
axs.spines['right'].set_color('#ffffff')
axs.spines['left'].set_color('#ffffff')
axs.tick_params(axis='x', colors='#ffffff')
axs.tick_params(axis='y', colors='#ffffff')

plt.savefig(Path('OutputFigures') / 'sunspots.png', bbox_inches='tight', dpi=600, transparent=True)
plt.savefig(Path('OutputFigures') / 'sunspots.svg', bbox_inches='tight', dpi=600, transparent=True)
plt.close()
