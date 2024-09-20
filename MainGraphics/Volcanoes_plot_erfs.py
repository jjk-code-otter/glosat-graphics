import matplotlib.pyplot as plt


def read_timeseries_file(filename):
    solar_years = []
    solar_erf = []
    solar_zero = []

    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            columns = line.split(',')

            year = int(columns[0])
            if year >= 1750 and year <= 2023:
                solar_years.append(year)
                solar_erf.append(float(columns[1]))
                solar_zero.append(0.5)

    return solar_years, solar_erf, solar_zero


# ERF files provided by Andrew Schurer
solar_years, solar_erf, _ = read_timeseries_file('InputData/solar_erf.csv')
volcanic_years, volcanic_erf, volcanic_zero = read_timeseries_file('InputData/volcanic_erf.csv')


def get_volcanoes():
    volcanoes = {
        'pinatubo': 1991,
        'chichon': 1982,
        'agung': 1963,
        'krakatoa': 1883,
        'santamaria': 1902,
        'hthh': 2022,
        'consiguina': 1835,
        'unknown2': 1831,
        'galunggung': 1822,
        'tambora': 1815,
        'unknown': 1808,
        'laki': 1783
    }
    return volcanoes

if __name__ == '_main__':
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

    axs.set_xlim(1749, 2025)

    plt.savefig('OutputFigures/solar_erf.svg', transparent=True, bbox_inches='tight')
    plt.close()

    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(27, 6)

    plt.plot(volcanic_years, volcanic_erf, color='#555555', linewidth=3)
    plt.fill_between(volcanic_years, volcanic_zero, volcanic_erf, color='#555555')

    axs.spines['right'].set_visible(False)
    axs.spines['bottom'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.set_title("")
    axs.tick_params(axis='both', which='major', labelsize=15)

    axs.spines['bottom'].set_color('#555555')
    axs.spines['top'].set_color('#555555')
    axs.spines['right'].set_color('#555555')
    axs.spines['left'].set_color('#555555')
    axs.tick_params(axis='x', colors='#555555')
    axs.tick_params(axis='y', colors='#555555')

    plt.gca().set_ylim(-6, 0.5)

    volcanoes = get_volcanoes()
    for v in volcanoes:
        plt.plot([volcanoes[v], volcanoes[v]], [-4.8, -5], color='#eeeeee')

    # Turn those erf dips into mountains
    plt.gca().invert_yaxis()

    plt.savefig('OutputFigures/volcanic_erf.svg', transparent=True, bbox_inches='tight')
    plt.close()
