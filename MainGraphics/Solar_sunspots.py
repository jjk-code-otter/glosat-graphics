"""
plot sunspot numbers
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt


def read_monthly_sunspot_file(filename):
    time = []
    data = []
    with open(filename, 'r') as f:
        for line in f:
            columns = line.split(';')
            time.append(float(columns[2]))
            data.append(float(columns[3]))
    return data, time


if __name__ == '__main__':
    data_dir_env = Path(os.getenv('DATADIR'))

    # https://www.sidc.be/SILSO/datafiles - Monthly mean total sunspot number [1/1749 - now]
    data, time = read_monthly_sunspot_file(Path('InputData') / 'SN_m_tot_V2.0.csv')

    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(18, 6)

    axs.plot(time, data, color='#FBD200')

    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_title("Sunspot Number 1749-2024", pad=5, fontdict={'fontsize': 20}, loc='left', color='#ffffff')
    axs.tick_params(axis='both', which='major', labelsize=15)

    axs.spines['bottom'].set_color('#ffffff')
    axs.spines['top'].set_color('#ffffff')
    axs.spines['right'].set_color('#ffffff')
    axs.spines['left'].set_color('#ffffff')
    axs.tick_params(axis='x', colors='#ffffff')
    axs.tick_params(axis='y', colors='#ffffff')

    axs.text(1750, -12, "Source: WDC-SILSO, Royal Observatory of Belgium, Brussels", color='#ffffff')

    axs.set_xlim(1749, 2025)

    plt.savefig(Path('OutputFigures') / 'sunspots.png', bbox_inches='tight', dpi=600, transparent=True)
    plt.savefig(Path('OutputFigures') / 'sunspots.svg', bbox_inches='tight', dpi=600, transparent=True)
    plt.close()
