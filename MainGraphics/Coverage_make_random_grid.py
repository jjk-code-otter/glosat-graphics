"""
Generates a grid of white squares of varying alpha. This is used in the title of the "Coverage" panel
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from matplotlib.patches import Rectangle

if __name__ == '__main__':
    seed = 326234623462457839626
    rng = np.random.default_rng(seed)
    print(rng.uniform(0, 1))

    xsize=72
    ysize=10

    # Create plot and ensure that it completely fills available space
    fig, axs = plt.subplots(1)
    fig.set_size_inches(xsize, ysize)
    fig.tight_layout()
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1)
    plt.plot([0, xsize], [0, ysize], color=None, linewidth=0)

    # Loop over all the x and y points
    for x, y in itertools.product(range(xsize), range(ysize)):

        # alpha decreases the further to the right one goes but is chosen randomly between low and high limits.
        # There is a sinusoidal (sine) variation in the y direction controlled by the lat_factor factor.
        lat_factor = 0.5
        low = 0.8 * (1 - lat_factor + lat_factor * np.sin(np.pi * y / ysize)) - x / (xsize-1)
        high = 1.4 * (1 - x / (xsize-1))

        # Choose random alpha between bounds but limit to range 0-1
        alpha = rng.uniform(low, high)
        if alpha < 0:
            alpha = 0
        if alpha > 1:
            alpha = 1

        axs.add_patch(
            Rectangle(xy=(x, y), width=1, height=1,
                      facecolor='#FFFFFF', edgecolor=None, alpha=alpha)
        )

    axs.set_xlim(0, xsize)
    axs.set_ylim(0, 10)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['bottom'].set_visible(False)
    axs.spines['left'].set_visible(False)
    plt.savefig(Path('OutputFigures') / 'random_alpha_grid.svg', transparent=True)
    plt.savefig(Path('OutputFigures') / 'random_alpha_grid.png', transparent=True)
    plt.close()
