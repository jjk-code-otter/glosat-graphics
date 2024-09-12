import copy
import json
from pathlib import Path
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from PIL import Image
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
import os

PLOT_DIAGNOSTICS = False


def plot_it(n_things, shifted_things, new_lower_bound, new_upper_bound):
    if not PLOT_DIAGNOSTICS:
        return

    fig, axs = plt.subplots(1)
    for i in range(n_things):
        l = shifted_things[i, 0] - shifted_things[i, 1] / 2
        r = shifted_things[i, 0] + shifted_things[i, 1] / 2
        m = shifted_things[i, 0]
        plt.plot([l, r, r, l, l], [i * 10, i * 10, (i + 1) * 10, (i + 1) * 10, i * 10])
    plt.plot([new_lower_bound, new_lower_bound], [0, n_things * 10])
    plt.plot([new_upper_bound, new_upper_bound], [0, n_things * 10])
    plt.show()
    plt.close()


def simple_stack(things, lower_bound, upper_bound):
    n_things = things.shape[0]
    shifted_things = copy.deepcopy(things)

    left_side = lower_bound

    for i in range(n_things):
        shifted_things[i, 0] = left_side + shifted_things[i, 1] / 2.
        left_side = left_side + shifted_things[i, 1]

    full_width = (
            np.max(shifted_things[:, 0] + shifted_things[:, 1] / 2) -
            np.min(shifted_things[:, 0] - shifted_things[:, 1] / 2)
    )

    return shifted_things, full_width


def remove_overlaps_with_limits(things, lower_bound, upper_bound):
    n_things = things.shape[0]
    shifted_things = copy.deepcopy(things)
    total_width = np.sum(things, axis=0)[1]

    # If the total width exceeds the plot size then just stack the images and they will be scaled down later
    if total_width > upper_bound - lower_bound:
        shifted_things, full_width = simple_stack(things, lower_bound, upper_bound)
        return shifted_things, full_width

    # Set up for iterations
    shift = True
    diff = 0.1
    iterations = 0
    new_lower_bound = lower_bound
    new_upper_bound = upper_bound

    # Loop till nothing is meaningfully adjusted
    while shift and diff > 0.00001:

        iterations += 1
        shift = False

        plot_it(n_things, shifted_things, new_lower_bound, new_upper_bound)

        # If the lefthand side of the leftmost box is outside the limits then push it into limits
        # set a new lower bound and recursively call the function on the other objects
        left_end_a = shifted_things[0, 0] - shifted_things[0, 1] / 2.
        if left_end_a < new_lower_bound:
            diff = new_lower_bound - left_end_a
            shifted_things[0, 0] = shifted_things[0, 0] + diff
            new_lower_bound = shifted_things[0, 0] + shifted_things[0, 1] / 2.
            if n_things > 1:
                selected = shifted_things[1:, :]
                fixed, width = remove_overlaps_with_limits(selected, new_lower_bound, new_upper_bound)
                shifted_things[1:, :] = fixed[:, :]

        plot_it(n_things, shifted_things, new_lower_bound, new_upper_bound)

        # If the righthand side of the rightmost box is outside the limits then push it into limits
        # set a new upper bound and recursively call the function on the other objects
        right_end_b = shifted_things[n_things - 1, 0] + shifted_things[n_things - 1, 1] / 2.
        if right_end_b > new_upper_bound:
            diff = right_end_b - new_upper_bound
            shifted_things[n_things - 1, 0] = shifted_things[n_things - 1, 0] - diff
            new_upper_bound = shifted_things[n_things - 1, 0] - shifted_things[n_things - 1, 1] / 2.
            if n_things > 1:
                selected = shifted_things[0:n_things - 1, :]
                fixed, width = remove_overlaps_with_limits(selected, new_lower_bound, new_upper_bound)
                shifted_things[0:n_things - 1, :] = fixed[:, :]

        plot_it(n_things, shifted_things, new_lower_bound, new_upper_bound)

        # If everything fits inside the box then iterate through the pairs and shuffle any overlaps.
        for i in range(n_things - 1):

            left_end_a = shifted_things[i, 0] - shifted_things[i, 1] / 2.
            right_end_a = shifted_things[i, 0] + shifted_things[i, 1] / 2.

            left_end_b = shifted_things[i + 1, 0] - shifted_things[i + 1, 1] / 2.
            right_end_b = shifted_things[i + 1, 0] + shifted_things[i + 1, 1] / 2.

            if right_end_a > left_end_b:
                diff = right_end_a - left_end_b
                half_diff = diff / 2.
                shifted_things[i, 0] = shifted_things[i, 0] - half_diff
                shifted_things[i + 1, 0] = shifted_things[i + 1, 0] + half_diff

                shift = True

            plot_it(n_things, shifted_things, new_lower_bound, new_upper_bound)

        full_width = (
                np.max(shifted_things[:, 0] + shifted_things[:, 1] / 2) -
                np.min(shifted_things[:, 0] - shifted_things[:, 1] / 2)
        )

        if iterations > 1500:
            return shifted_things, full_width

    return shifted_things, full_width


def remove_overlaps(things):
    """
    Take array n x 2 containing object midpoint and width

    :param things:
    :return:
    """

    n_things = things.shape[0]

    shifted_things = copy.deepcopy(things)

    # Set up for iterations
    shift = True
    diff = 0.1
    iterations = 0

    # Loop till nothing is meaningfully adjusted
    while shift and diff > 0.00001:

        iterations += 1
        shift = False

        for i in range(n_things - 1):

            right_end = shifted_things[i, 0] + shifted_things[i, 1] / 2.
            left_end = shifted_things[i + 1, 0] - shifted_things[i + 1, 1] / 2.

            if right_end > left_end:
                diff = right_end - left_end

                half_diff = max([diff / 2, 0.5])

                shifted_things[i, 0] = shifted_things[i, 0] - half_diff
                shifted_things[i + 1, 0] = shifted_things[i + 1, 0] + half_diff

                shift = True

        full_width = (
                np.max(shifted_things[:, 0] + shifted_things[:, 1] / 2) -
                np.min(shifted_things[:, 0] - shifted_things[:, 1] / 2)
        )

        if iterations > 500:
            return shifted_things, full_width

    return shifted_things, full_width


class Label():
    """
    Simple Label class to handle combination, sorting and comparison of labels
    """

    def __init__(self, listicle):
        self.start = listicle[0]
        self.end = listicle[1]
        self.text = listicle[2]
        if len(listicle) == 4:
            self.offset = listicle[3]
        else:
            self.offset = 0

    def __eq__(self, a):
        """
        Two labels are equal if they have the same start date

        :param a: Label to be compared
        :return: bool
        """
        if self.start == a.start:
            return True
        return False

    def __add__(self, a):
        """
        Adding two labels will add the text element together with a new-line character between elements

        :param a: Label to be added
        :return: Label
        """
        start = self.start
        end = self.end
        text = self.text + '\n' + a.text
        if self.end is not None and a.end is not None:
            end = max([self.end, a.end])
        elif self.end is None:
            end = a.end

        return Label([start, end, text])


class Timeline():

    def __init__(self):
        self.start_year = 1850
        self.end_year = 2024
        self.dir = None
        self.labels = []
        self.images = []

    @staticmethod
    def from_json(filename):
        with open(filename, 'r') as f:
            metadata = json.load(f)

            new_timeline = Timeline()

            new_timeline.dir = Path(filename).parents[0]
            new_timeline.start_year = metadata['start_year']
            new_timeline.end_year = metadata['end_year']
            new_timeline.labels = []
            for item in metadata['labels']:
                new_timeline.labels.append(Label(item))

            new_timeline.images = metadata['images']

            new_timeline.merge_labels()

        return new_timeline

    def merge_labels(self):
        """
        If there are multiple labels with the same start date then combine them with a new line
        separating the text items to avoid overwriting

        :return:
        """
        burner = copy.deepcopy(self.labels)
        new_labels = []
        while len(burner) > 0:
            label1 = burner.pop()
            to_remove = []
            for label2 in burner:
                if label1 == label2:
                    label1 = label1 + label2
                    to_remove.append(label2)

            if len(to_remove) > 0:
                for r in to_remove:
                    burner.remove(r)

            new_labels.append(label1)

        self.labels = new_labels

    @staticmethod
    def plot_objects(ax, objects, objects_orig, images):

        full_plot_width = ax.get_xlim()
        plot_width = full_plot_width[1] - full_plot_width[0]

        img_min = np.min(objects[:, 0] - objects[:, 1] / 2)
        img_max = np.max(objects[:, 0] + objects[:, 1] / 2)
        img_full_width = img_max - img_min

        for i in range(objects.shape[0]):
            tx = objects[i, 0]
            ty1 = objects[i, 2]
            ty2 = objects[i, 3]
            scaled_width = objects[i, 1]

            x0 = tx - scaled_width / 2
            x1 = tx + scaled_width / 2

            # If the images strip is wider than the whole width of the plot then scale down
            if img_full_width > plot_width:
                offset = np.min(objects[:, 0] - objects[:, 1] / 2)
                x0 = (x0 - offset) * plot_width / img_full_width + full_plot_width[0]
                x1 = (x1 - offset) * plot_width / img_full_width + full_plot_width[0]
                tx = (tx - offset) * plot_width / img_full_width + full_plot_width[0]
            elif img_min < full_plot_width[0]:
                offset = img_min - full_plot_width[0]
                x0 = x0 - offset
                x1 = x1 - offset
                tx = tx - offset
            elif img_max > full_plot_width[1]:
                offset = img_max - full_plot_width[1]
                x0 = x0 - offset
                x1 = x1 - offset
                tx = tx - offset

            ax.imshow(images[i], extent=[x0, x1, ty2, ty1], aspect='auto', zorder=100)

            p = plt.plot(
                [tx, objects_orig[i, 0], objects_orig[i, 0]],
                [ty1, -10, 0],
                linewidth=5
            )
            plt.plot(
                [tx, x1, x1, x0, x0, tx],
                [ty1, ty1, ty2, ty2, ty1, ty1],
                linewidth=5, zorder=200, color=p[-1].get_color()
            )

    def sort_images(self, ax):
        high_image = True
        hi_objects = np.zeros((len(self.images), 4))
        lo_objects = np.zeros((len(self.images), 4))
        hi_images = []
        lo_images = []
        hi_count = 0
        lo_count = 0

        for label in self.images:
            year1 = label[0]
            tag = label[2]

            img = Image.open(self.dir / tag)

            # Set up transforms from axis coordinates to normalised coordinates (0 to 1)
            axis_to_data = ax.transAxes + ax.transData.inverted()
            data_to_axis = axis_to_data.inverted()  # Kept in case the reverse transform is ever needed

            # Get the image width and height and scale to unit height. To preserve the aspect ratio, need
            # to further scale the width by the ratio of the figure width and height
            width, height = img.size
            width = width / height
            size = fig.get_size_inches()
            width = width / (size[0] / size[1])

            # Transform the width and height into the axis coordinate system
            points_data0 = axis_to_data.transform((0, 0))
            points_data1 = axis_to_data.transform((width, 1))

            # Scale the transformed coordinates to the desired size (difference between ty1 and ty2
            image_height_in_data = 35.
            scale = (points_data1[1] - points_data0[1]) / image_height_in_data

            # Set the location of the image. Alternate images are plotted high and low.
            scaled_width = (points_data1[0] - points_data0[0]) / scale
            tx = mpl.dates.date2num(date(year1, 1, 1))
            if high_image:
                high_image = False
                hi_objects[hi_count, :] = np.array([tx, scaled_width, -55, -55 - image_height_in_data])
                hi_images.append(img)
                hi_count += 1
            else:
                high_image = True
                lo_objects[lo_count, :] = np.array([tx, scaled_width, -15, -15 - image_height_in_data])
                lo_images.append(img)
                lo_count += 1

        hi_objects_orig = hi_objects[0:hi_count, :]
        lo_objects_orig = lo_objects[0:lo_count, :]

        return hi_objects_orig, lo_objects_orig, hi_images, lo_images

    def plot_labels(self, ax):

        for label in self.labels:
            # Set up the range one year for labels with no end date or as specified
            tx = mpl.dates.date2num(date(label.start, 1, 1))
            tx2 = mpl.dates.date2num(date(label.start, 12, 31))
            if label.end is not None:
                tx2 = mpl.dates.date2num(date(label.end, 12, 31))

            # plot the text
            # ax.text((tx + tx2) / 2, 32, label.text, rotation=45)
            tx_with_offset = mpl.dates.date2num(date(label.start + label.offset, 1, 1))
            ax.text(tx_with_offset, 32, label.text, rotation=45)

            # plot the pips that indicate the range
            for i in range(5):
                delta = 5
                ax.add_patch(Rectangle((tx + delta, 1 + i * 6), tx2 - tx - 2 * delta, 5,
                                       color='green', edgecolor=None, alpha=0.5, linewidth=0))

    def plot_images(self, ax, n_rows):

        # Sort the images into two strips by alternating high and low, transform axes, scale etc.
        high_objects_orig, low_objects_orig, high_images, low_images = self.sort_images(ax)

        # Jiggle the images so that they don't overlap
        fpw = ax.get_xlim()
        high_objects, high_full_width = remove_overlaps_with_limits(high_objects_orig, fpw[0], fpw[1])
        low_objects, low_full_width = remove_overlaps_with_limits(low_objects_orig, fpw[0], fpw[1])

        # Now plot it all out
        Timeline.plot_objects(ax, high_objects, high_objects_orig, high_images)
        Timeline.plot_objects(ax, low_objects, low_objects_orig, low_images)

    def plot(self, ax, n_rows=2):
        start_date = date(self.start_year, 1, 1)
        end_date = date(self.end_year, 1, 1)

        ax.plot([start_date, end_date], [-20, 20], linestyle='None')
        ax.set_xlim(start_date, end_date)
        ax.set_ylim(-100, 100)

        self.plot_labels(ax)
        self.plot_images(ax, n_rows)

        axs.spines['bottom'].set_position('zero')
        axs.spines['right'].set_visible(False)
        axs.spines['left'].set_visible(False)
        axs.spines['top'].set_visible(False)
        axs.xaxis.set_major_locator(mdates.YearLocator(base=10))
        axs.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axs.get_yaxis().set_visible(False)


if __name__ == '__main__':
    f = Timeline.from_json('InputImages/example.json')

    print(f.start_year)
    print(f.end_year)

    fig, axs = plt.subplots(1, sharex=True)
    fig.set_size_inches(16, 8)

    f.plot(axs, n_rows=1)

    plt.savefig('OutputFigures/time_line.svg')
    plt.savefig('OutputFigures/time_line.png', bbox_inches="tight", dpi=300)
    plt.close()
