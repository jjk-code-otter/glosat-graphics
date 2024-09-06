import matplotlib.pyplot as plt

solar_years = []
solar_erf = []

with open('InputData/solar_erf.csv', 'r') as f:
    f.readline()
    for line in f:
        columns = line.split(',')

        year = int(columns[0])
        if year >= 1750 and year <= 2023:
            solar_years.append(year)
            solar_erf.append(float(columns[1]))

volcanic_years = []
volcanic_erf = []

with open('InputData/volcanic_erf.csv', 'r') as f:
    f.readline()
    for line in f:
        columns = line.split(',')

        year = int(columns[0])
        if year >= 1750 and year <= 2023:
            volcanic_years.append(year)
            volcanic_erf.append(float(columns[1]))
fig, axs = plt.subplots(1, 1)
fig.set_size_inches(18, 6)

plt.plot(solar_years, solar_erf, color='#FBD200', linewidth=3)

axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_title("Solar forcing 1750-2004", pad=5, fontdict={'fontsize': 20}, loc='left', color='#ffffff')
axs.tick_params(axis='both', which='major', labelsize=15)

axs.spines['bottom'].set_color('#ffffff')
axs.spines['top'].set_color('#ffffff')
axs.spines['right'].set_color('#ffffff')
axs.spines['left'].set_color('#ffffff')
axs.tick_params(axis='x', colors='#ffffff')
axs.tick_params(axis='y', colors='#ffffff')

plt.savefig('OutputFigures/solar_erf.svg', transparent=True, bbox_inches='tight')
plt.close()

fig, axs = plt.subplots(1, 1)
fig.set_size_inches(18, 6)

plt.plot(volcanic_years, volcanic_erf, color='#555555', linewidth=3)

axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.set_title("Volcanic forcing 1750-2004", pad=5, fontdict={'fontsize': 20}, loc='left', color='#555555')
axs.tick_params(axis='both', which='major', labelsize=15)

axs.spines['bottom'].set_color('#555555')
axs.spines['top'].set_color('#555555')
axs.spines['right'].set_color('#555555')
axs.spines['left'].set_color('#555555')
axs.tick_params(axis='x', colors='#555555')
axs.tick_params(axis='y', colors='#555555')

plt.gca().invert_yaxis()

plt.savefig('OutputFigures/volcanic_erf.svg', transparent=True, bbox_inches='tight')
plt.close()
