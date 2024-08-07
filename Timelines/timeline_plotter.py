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

#            ax.text(date(year1, 1, 1), 32, tag, rotation=45)
            ax.text((tx+tx2)/2, 32, tag, rotation=45)

            for i in range(5):
                delta = 5
                ax.add_patch(Rectangle((tx + delta, 1 + i * 6), tx2 - tx - 2 * delta, 5,
                                       color='green', edgecolor=None, alpha=0.5, linewidth=0))

        for label in self.images:
            year1 = label[0]
            tag = label[2]

            img = Image.open(tag)



            # Set up transforms from axis coordinates to normalised coordinates (0 to 1)
            axis_to_data = ax.transAxes + ax.transData.inverted()
            data_to_axis = axis_to_data.inverted()

            # Get the image width and height and scale to unit height. To preserve the aspect ratio, need
            # to further scale the width by the ratio of the figure width and height
            width, height = img.size
            width = width / height
            width = width / 2

            # Transform the width and height into the axis coordinate system
            points_data0 = axis_to_data.transform((0, 0))
            points_data1 = axis_to_data.transform((width, 1))

            # Scale the transformed coordinates to the desired size (difference between ty1 and ty2
            scale = (points_data1[1] - points_data0[1]) / 35.

            # Set the location of the image. Alternate images are plotted high and low.
            tx = mpl.dates.date2num(date(year1, 1, 1))
            ty1 = -15
            ty2 = -50
            if high_image:
                high_image = False
                ty1 = -50
                ty2 = -85
            else:
                high_image = True

            scaled_width = (points_data1[0] - points_data0[0]) / scale

            # plt.plot(
            #     [tx - scaled_width / 2, tx + scaled_width / 2, tx + scaled_width / 2, tx - scaled_width / 2, tx - scaled_width / 2],
            #          [ty2, ty2, ty1, ty1, ty2],
            #          color='black'
            # )

            ax.imshow(img, extent=[tx - scaled_width / 2, tx + scaled_width / 2, ty2, ty1],
                      aspect='auto')  # , cmap='gray')



        axs.spines['bottom'].set_position('zero')
        axs.spines['right'].set_visible(False)
        axs.spines['left'].set_visible(False)
        axs.spines['top'].set_visible(False)
        axs.xaxis.set_major_locator(mdates.YearLocator(base=10))
        axs.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axs.get_yaxis().set_visible(False)

        return


f = Timeline.from_json('example.json')

print(f.start_year)
print(f.end_year)

fig, axs = plt.subplots(1, sharex=True)
fig.set_size_inches(16, 8)

f.plot(axs)

plt.savefig('time_line.svg')
plt.savefig('time_line.png', bbox_inches="tight", dpi=300)
plt.close()
