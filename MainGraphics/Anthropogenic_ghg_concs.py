import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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


def read_law_dome_chunk(f, variable):
    """
    Read a chunk of the Law Dome file format

    :param f: file handle
        File handle we are reading data from
    :param variable: str
        Variable name to be read, one of CO2, CH4 or N2O
    :return:
    """
    years = []
    values = []

    for line in f:
        if line == '			\n' or line == '\n':
            break
        columns = line.split()
        if len(columns) < 3:
            break
        years.append(int(np.floor(float(columns[1]))))

        value_column = 2
        if variable == 'CH4':
            value_column = 3
        values.append(float(columns[value_column]))

    return years, values


def read_annual_law_dome(filename, variable):
    """
    The law dome data file contains all three variables in one file, so we need to specify the filename and the
    variable we weant to extract

    :param filename: str
        Filename to be read in
    :param variable: str
        The variable to be read from the file, one of CO2, CH4, or N2O.
    :return:
    """
    # Open file, read through the file till the chosen variable is reached
    with open(filename, 'r') as f:
        for line in f:
            if variable in line:
                break
        years, values = read_law_dome_chunk(f, variable)
    return years, values


if __name__ == '__main__':

    # https://gaw.kishou.go.jp/static/publications/global_mean_mole_fractions/2023/co2_annual_20231115.csv
    co2_years, co2, _ = read_annual_wdcgg('InputData/co2_annual_20231115.csv')
    # https://gaw.kishou.go.jp/static/publications/global_mean_mole_fractions/2023/ch4_annual_20231115.csv
    ch4_years, ch4, _ = read_annual_wdcgg('InputData/ch4_annual_20231115.csv')
    # https://gaw.kishou.go.jp/static/publications/global_mean_mole_fractions/2023/n2o_annual_20231115.csv
    n2o_years, n2o, _ = read_annual_wdcgg('InputData/n2o_annual_20231115.csv')

    # https://agupubs.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1029%2F2006GL026152&file=grl21511-sup-0002-ts01.txt
    long_co2_years, long_co2 = read_annual_law_dome('InputData/grl21511-sup-0002-ts01.txt', 'CO2')
    long_ch4_years, long_ch4 = read_annual_law_dome('InputData/grl21511-sup-0002-ts01.txt', 'CH4')
    long_n2o_years, long_n2o = read_annual_law_dome('InputData/grl21511-sup-0002-ts01.txt', 'N2O')

    fig, axs = plt.subplots(3, 1)
    fig.set_size_inches(9, 9)

    axs[0].plot(co2_years, co2, linewidth=5)
    axs[0].plot(long_co2_years, long_co2, linewidth=5)
    axs[0].set_xlim(1750, 2024)

    axs[1].plot(ch4_years, ch4, linewidth=5)
    axs[1].plot(long_ch4_years, long_ch4, linewidth=5)
    axs[1].set_xlim(1750, 2024)

    axs[2].plot(n2o_years, n2o, linewidth=5)
    axs[2].plot(long_n2o_years, long_n2o, linewidth=5)
    axs[2].set_xlim(1750, 2024)

    for i in range(3):
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        axs[i].tick_params(axis='both', which='major', labelsize=12)

    plt.savefig(Path('OutputFigures') / 'ghg_concs.svg', bbox_inches='tight')
