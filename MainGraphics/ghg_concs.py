import numpy as np
import matplotlib.pyplot as plt

def read_annual_wdcgg(filename):
    years = []
    anomalies = []
    uncertainty = []

    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            columns = line.split(',')
            year = columns[0]
            years.append(int(year))
            if columns[1] != '':
                anomalies.append(float(columns[1]))
                uncertainty.append(float(columns[2]))
            else:
                anomalies.append(np.nan)
                uncertainty.append(np.nan)

    return years, anomalies, uncertainty


def read_chunk(f, search_string):
    years = []
    values = []

    for line in f:
        print(line)
        if line == '			\n' or line == '\n':
            break
        columns = line.split()
        if len(columns) < 3:
            break
        years.append(int(np.floor(float(columns[1]))))

        value_column = 2
        if search_string == 'CH4':
            value_column = 3
        values.append(float(columns[value_column]))

    return years, values


def read_annual_law_dome(filename, variable):
    years = []
    anomalies = []

    if variable == 'ch4':
        search_string = 'CH4'
    elif variable == 'n2o':
        search_string = 'N2O'
    elif variable == 'co2':
        search_string = 'CO2'
    else:
        raise ValueError

    with open(filename, 'r') as f:
        for line in f:
            if search_string in line:
                break

        years, values = read_chunk(f, search_string)

    return years, values


co2_years, co2, _ = read_annual_wdcgg('co2_annual_20231115.csv')
ch4_years, ch4, _ = read_annual_wdcgg('ch4_annual_20231115.csv')
n2o_years, n2o, _ = read_annual_wdcgg('n2o_annual_20231115.csv')

long_co2_years, long_co2 = read_annual_law_dome('grl21511-sup-0002-ts01.txt', 'co2')
long_ch4_years, long_ch4 = read_annual_law_dome('grl21511-sup-0002-ts01.txt', 'ch4')
long_n2o_years, long_n2o = read_annual_law_dome('grl21511-sup-0002-ts01.txt', 'n2o')

fig, axs = plt.subplots(3, 1)
fig.set_size_inches(9, 9)

axs[0].plot(co2_years, co2, linewidth=5)
axs[0].plot(long_co2_years, long_co2, linewidth=5)
axs[0].set_xlim(1750,2024)

axs[1].plot(ch4_years, ch4, linewidth=5)
axs[1].plot(long_ch4_years, long_ch4, linewidth=5)
axs[1].set_xlim(1750,2024)

axs[2].plot(n2o_years, n2o, linewidth=5)
axs[2].plot(long_n2o_years, long_n2o, linewidth=5)
axs[2].set_xlim(1750,2024)


for i in range(3):
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)
    axs[i].tick_params(axis='both', which='major', labelsize=12)

plt.savefig('ghg_concs.svg', bbox_inches='tight')