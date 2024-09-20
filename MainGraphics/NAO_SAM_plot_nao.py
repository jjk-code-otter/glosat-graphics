import copy

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def read_nao(filename):
    # Read NAO file
    years = []
    months = []
    nao = []

    missing_data_indicator = -99.99
    first_complete_year = 1825

    with open(filename, 'r') as f:
        for line in f:
            columns = line.split()
            year = int(columns[0])

            for i in range(1, 13):
                if year >= first_complete_year:
                    years.append(year)
                    months.append(i)
                    nao.append(float(columns[i]))

    years = np.array(years)
    months = np.array(months)
    nao = np.array(nao)

    # Remove missing data
    years = years[nao != missing_data_indicator]
    months = months[nao != missing_data_indicator]
    nao = nao[nao != missing_data_indicator]

    return years, months, nao


def seasonal_running_average(nao):
    # Calculate seasonal averages
    nao_3month = copy.deepcopy(nao)
    nao_3month[:] = np.nan

    for i in range(2, len(nao)):
        select = nao[i - 2:i + 1]
        nao_3month[i] = np.mean(select)

    return nao_3month


def plot_bar(ax, time, value, delta, color_positive, color_negative):
    x = time + delta
    y = 0
    width = 1 - 2 * delta
    height = value

    if value > 0:
        ax.add_patch(Rectangle((x, y), width, height, facecolor=color_positive, edgecolor=None))
    else:
        ax.add_patch(Rectangle((x, y), width, height, facecolor=color_negative, edgecolor=None))


if __name__ == '__main__':

    # https://crudata.uea.ac.uk/cru/data/nao/nao_3dp.dat
    years, months, nao = read_nao('InputData/nao_3dp.dat.txt.txt')

    # Calculate seasonal averages
    nao_3month = seasonal_running_average(nao)

    time = years + (months - 1) / 12.

    # Select winter, DJF (last month is 2)
    sub_years = years[months == 2]
    sub_nao = nao_3month[months == 2]
    sub_month = months[months == 2]

    # Plot NAO
    fig, ax = plt.subplots(figsize=(16, 5))

    # Sets the gaps between the bars.
    delta = 0.1

    # Stripes every 25 years
    for i in range(4):
        ax.add_patch(
            Rectangle((1825 + i * 50, -3.5), 25, 7, facecolor='lightgrey', edgecolor=None, alpha=0.5)
        )

    for i in range(len(sub_years)):
        plot_bar(ax, sub_years[i], sub_nao[i], delta, "red", "blue")

    ax.set_xlim(1825, 2025)
    ax.set_ylim(-3.5, 3.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_title("Winter North Atlantic Oscillation 1826-2024", pad=5, fontdict={'fontsize': 20}, loc='left',
                 color='black')

    plt.plot(sub_years, sub_nao, linewidth=0)
    plt.savefig('OutputFigures/nao.svg', transparent=True, bbox_inches='tight')
    plt.savefig('OutputFigures/nao.png', transparent=True, bbox_inches='tight', dpi=1200)
    plt.close()
