import copy

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def read_sam(filename):
    years = []
    sam_recon = []
    sam_unc = []
    sam_marshall = []

    with open(filename, 'r') as f:
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

    return years, sam_recon, sam_unc, sam_marshall


def read_bas_sam(filename):
    bas2_years = []
    bas2_sam = []

    with open(filename, 'r') as f:
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

    return bas2_years, bas2_sam


def plot_bar(ax, time, value, delta, color_positive, color_negative):
    x = time + delta
    y = 0
    width = 1 - 2 * delta
    height = value

    if value > 0:
        ax.add_patch(Rectangle((x, y), width, height, facecolor=color_positive, edgecolor=None))
    else:
        ax.add_patch(Rectangle((x, y), width, height, facecolor=color_negative, edgecolor=None))

def plot_uncertainty_bar(ax, time, value, unc, delta, color_positive, color_negative):
    x = time + delta
    y = value - unc
    width = 1 - 2 * delta
    height = 2 * unc

    if value > 0:
        ax.add_patch(Rectangle((x, y), width, height, facecolor=color_positive, edgecolor=None))
    else:
        ax.add_patch(Rectangle((x, y), width, height, facecolor=color_negative, edgecolor=None))


if __name__ == '__main__':
    # https://www.ncei.noaa.gov/pub/data/paleo/contributions_by_author/abram2014/abram2014sam-noaa.txt
    years, sam_recon, sam_unc, sam_marshall = read_sam('InputData/abram2014sam-noaa.txt')
    # http://www.nerc-bas.ac.uk/public/icd/gjma/newsam.1957.2007.seas.txt
    bas2_years, bas2_sam = read_bas_sam('InputData/newsam.1957.2007.seas.txt')

    # Plot NAO
    fig, ax = plt.subplots(figsize=(16, 5))

    # Sets the gaps between the bars.
    delta = 0.1

    # Grey stripes every 25 years to help readability of x axis values
    for i in range(4):
        ax.add_patch(Rectangle((1825 + i * 50, -5.5), 25, 11, color='lightgrey', alpha=0.5))

    for i in range(len(sam_recon)):
        if sam_recon[i] is not None and years[i] < 1957:
            plot_uncertainty_bar(ax, years[i], sam_recon[i], sam_unc[i], delta, "pink", "lightblue")
            plot_bar(ax, years[i], sam_recon[i], delta, "red", "blue")

    for i in range(len(bas2_sam)):
        if bas2_sam[i] is not None:
            plot_bar(ax, bas2_years[i], bas2_sam[i], delta, "red", "blue")

    # ax.text(1955, 5, 'Abram et al. 2014 reconstruction', fontsize=10, ha='right')
    # ax.text(1959, 5, 'Marshall 2003', fontsize=10, ha='left')

    ax.set_xlim(1825, 2025)
    ax.set_ylim(-5.5, 5.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_title("Southern Annular Mode 1826-2024", pad=5, fontdict={'fontsize': 20}, loc='left', color='black')

    plt.plot(years, sam_marshall, linewidth=0)
    plt.savefig('OutputFigures/sam.svg', transparent=True, bbox_inches='tight')
    plt.savefig('OutputFigures/sam.png', transparent=True, bbox_inches='tight', dpi=1200)
    plt.close()
