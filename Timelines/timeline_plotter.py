import json
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from PIL import Image
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle


class Timeline():

    def __init__(self):
        self.start_year = 1850
        self.end_year = 2024
        self.labels = []
        self.images = []

    @staticmethod
    def from_json(filename):
        with open(filename, 'r') as f:
            metadata = json.load(f)

            new_timeline = Timeline()
            new_timeline.start_year = metadata['start_year']
            new_timeline.end_year = metadata['end_year']
            new_timeline.labels = metadata['labels']
            new_timeline.images = metadata['images']

        return new_timeline

    def plot(self, ax):
        start_date = date(self.start_year, 1, 1)
        end_date = date(self.end_year, 1, 1)

        ax.plot([start_date, end_date], [-20, 20], linestyle='None')
        ax.set_xlim(start_date, end_date)
        ax.set_ylim(-100, 100)

        high_image = True

        for label in self.labels:
            year1 = label[0]
            year2 = label[1]
            tag = label[2]

            tx = mpl.dates.date2num(date(year1, 1, 1))
            tx2 = mpl.dates.date2num(date(year1, 12, 31))
            if year2 is not None:
                tx2 = mpl.dates.date2num(date(year2, 12, 31))
            ax.text(date(year1, 1, 1), 32, tag, rotation=45)
            for i in range(5):
                delta = 5
                ax.add_patch(Rectangle((tx + delta, 1 + i * 6), tx2 - tx - 2 * delta, 5,
                                       color='green', edgecolor=None, alpha=0.5, linewidth=0))

        for label in self.images:
            year1 = label[0]
            tag = label[2]

            img = Image.open(tag)

            width, height = img.size

            tx = mpl.dates.date2num(date(year1, 1, 1))
            ty = -15
            ty2 = -50

            if high_image:
                high_image = False
                ty = -50
                ty2 = -85
            else:
                high_image = True

            ax.imshow(img, extent=[tx - 5 * 365, tx + 365 * 5, ty2, ty], aspect=100, cmap='gray')

        return


f = Timeline.from_json('example.json')

print(f.start_year)
print(f.end_year)

fig, axs = plt.subplots(1, sharex=True)
fig.set_size_inches(16, 8)

f.plot(axs)

axs.spines['bottom'].set_position('zero')
axs.spines['right'].set_visible(False)
axs.spines['left'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.xaxis.set_major_locator(mdates.YearLocator(base=10))
axs.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axs.get_yaxis().set_visible(False)

plt.savefig('time_line.svg')
plt.savefig('time_line.png', bbox_inches="tight", dpi=300)
plt.close()
