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


def parse_strings(instr):
    """
    Quick function to parse the input strings as floats or, if empty, Nones.

    :param instr: str
        String to be parsed.
    :return: None or float
        If the string is empty then return None else return a float
    """
    if instr == '':
        return None
    else:
        return float(instr)


def read_datasets(infile):
    years = []
    berkeley = []
    era5 = []
    gistemp = []
    hadcrut = []
    jra55 = []
    noaa = []

    with open(infile, 'r') as f:
        for i in range(79):
            f.readline()
        for line in f:

            if 'end data' in line:
                break

            columns = line.split(',')

            years.append(int(columns[1]))
            berkeley.append(parse_strings(columns[2]))
            era5.append(parse_strings(columns[3]))
            gistemp.append(parse_strings(columns[4]))
            hadcrut.append(parse_strings(columns[5]))
            jra55.append(parse_strings(columns[6]))
            noaa.append(parse_strings(columns[7]))

    return years, berkeley, era5, gistemp, hadcrut, jra55, noaa


if __name__ == '__main__':

    # Input file is from WMO dashboard: https://jjk-code-otter.github.io/demo-dash/Dashboard2023/formatted_data/Global_temperature_data_files.zip
    years, berkeley, era5, gistemp, hadcrut, jra55, noaa = read_datasets('InputData/tas_summary.csv')

    sns.set(font='Franklin Gothic Book', rc=STANDARD_PARAMETER_SET)

    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(16 * 1.25, 5 * 1.5)

    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
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
