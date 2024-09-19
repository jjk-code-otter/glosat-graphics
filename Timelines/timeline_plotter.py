"""
The timeline plotter requires a configuration file and a bunch of figures in the InputImages directory.

Run like so:

python timeline_plotter.py -c InputImages/example.json -o OutputFigures/time_line.png -r 3

The configuration file is a json file of a dictionary with four keys:

start_year - the start year for the plot. Together with end_year these specify the extent of the x-axis
end_year - the end year for the plot. Together with start_year these specify the extent of the x-axis
labels - a list of lists. Each list member should be a list indicating:
    Start year (int) for the label,
    End year (int) for the label (can be set to null),
    String (str) to print
    (optional) offset (int). Sometimes the labels get a little crowded, so the offset can be used to move them by whole
    numbers of years. Negative numbers move them earlier (left) and positive numbers move them later (right).
images - a list of lists. Each list member should contain three elements:
    Year (int) which indicates the point at which the mid point of the image will be plotted
    Offset (float or null) Allows manual placing of each image. If a value (in years) is given then the image will
    be offset by that number of years. This will override shuffling to fit, but there may still be some automatic
    adjustment to make sure images are within the plot boundaries. If all values are set to null then the script will
    attempt to fit the images by adjusting their locations automatically
    Name of image (str). The image should be a filename of a file in the same directory as the json configuration file.

Running this script will then generate the timeline. You can choose how many rows of images you want in the timeline
by changing the n_rows keyword argument in the plot function. The fewer the rows, the larger the images, which can
lead to crowding. Experiment with the number of rows. The images will always fit in the number of rows you specify, but
the results won't always be pretty.
"""
import copy
import json
from argparse import ArgumentParser
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

# If you want to watch what the algorithm does, you can change this to True. It breaks everything, but it's kind
# of fun. It's probably not worth it.
PLOT_DIAGNOSTICS = False


def plot_it(n_things, shifted_things, new_lower_bound, new_upper_bound):
    """
    Plot out some simple diagnostics that show the locations of each image and the upper and lower bounds.

    :param n_things: int
        number of things being fitted
    :param shifted_things:  ndarray
        Array (n,2) containing the midpoints [:, 0] and widths [:, 1] of the items being fitted
    :param new_lower_bound: float
        Lower edge of the container into which the tings are being fitted
    :param new_upper_bound:  float
        Upper edge of the container into which the tings are being fitted
    :return: None
    """
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


def parse_list(in_list):
    """
    Take a list which is all Nones or a mix of floats and Nones. If it is all Nones return the original list
    if it is a mixture, replace Nones with zeros.

    :param in_list:
    :return:
    """
    all_none = True
    for element in in_list:
        if element is not None:
            all_none = False
    if not all_none:
        out_list = [0.0 if x is None else x for x in in_list]
    else:
        out_list = in_list

    return out_list


def simple_stack(things, lower_bound, upper_bound):
    """
    The things are placed end to end starting at the lower_bound and continuing till we run out of things. This can end up with
    things stacked past the upper bound.

    :param things: ndarray(n_things, 2)
        This will contain the things. The second axis can be any size as long as it's 2 or more. The [n,0] elements
        are the mid points of the things and the [n,1] elements are the widths of the things.
    :param lower_bound: float
        The left hand edge of the container.
    :param upper_bound:  float
        The right hand edge of the container.
    :return: (ndarray, float)
        The ndarray will contain the shifted things that no longer overlap and the float will be the full width of
        all the elements after shuffling from the left side of the leftmost object to the right edge of the rightmost
        object.
    """
    n_things = things.shape[0]
    shifted_things = copy.deepcopy(things)

    left_side = lower_bound

    for i in range(n_things):
        shifted_things[i, 0] = left_side + shifted_things[i, 1] / 2.
        left_side = left_side + shifted_things[i, 1]

    full_width = calculate_full_width(shifted_things)

    return shifted_things, full_width


def calculate_full_width(things):
    return (np.max(things[:, 0] + things[:, 1] / 2) - np.min(things[:, 0] - things[:, 1] / 2))


def use_offsets(things, offsets):
    n_things = things.shape[0]
    shifted_things = copy.deepcopy(things)

    for i in range(n_things):
        shifted_things[i, 0] = things[i, 0] + offsets[i]

    full_width = calculate_full_width(shifted_things)

    return shifted_things, full_width


def remove_overlaps_with_limits(things, lower_bound, upper_bound):
    """
    Given some things in a ndarray(n_things, 2) specifying mid point and width for each of the things. The algorithm
    will attempt to fit the things between the lower_bound and upper_bound specified. If there are too many things, to
    fit between the bounds, then they will be returned as a simple stack.

    If there is more than enough space to fit the things then the algorithm proceeds like so:
    1. check if the left edge of the first thing is below the lower bound. If it is, move the thing so it is in bounds,
       remove it from the list, reset the lower bound and recursively call the function on the remaining elements
    2. check if the right edge of the last thing is above the upper bound. If it is, move the thing so it is in bounds,
       remove it from the list, reset the upper bound and recursively call the function on the remaining elements
    3. Go through each thing. If it overlaps with the next thing, then move the things apart so they are just touching.
    4. Return to 1. until no more changes are made.

    :param things: ndarray(n_things, 2)
        This will contain the things. The second axis can be any size as long as it's 2 or more. The [n,0] elements
        are the mid points of the things and the [n,1] elements are the widths of the things.
    :param lower_bound: float
        The left hand edge of the container.
    :param upper_bound: float
        The right hand edge of the container.
    :return: (ndarray, float)
        The ndarray will contain the shifted things that no longer overlap and the float will be the full width of
        all the elements after shuffling from the left side of the leftmost object to the right edge of the rightmost
        object.
    """
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

        full_width = calculate_full_width(shifted_things)

        if iterations > 1500:
            return shifted_things, full_width

    return shifted_things, full_width


def remove_overlaps(things):
    """
    Given some things in a ndarray(n_things, 2) specifying mid point and width for each of the things. The algorithm
    will attempt to remove overlaps iteratively in a way that keeps the things as close to their original positions
    as possible.

    The algorithm proceeds like so:
    1. Go through each thing. If it overlaps with the next thing, then move the things apart - the left item moves
       left and the righ item moves right - so they are just touching.
    2. Return to 1. until no more changes are made.

    :param things: ndarray(n_things, 2)
        This will contain the things. The second axis can be any size as long as it's 2 or more. The [n,0] elements
        are the mid points of the things and the [n,1] elements are the widths of the things.
    :return: (ndarray, float)
        The ndarray will contain the shifted things that no longer overlap and the float will be the full width of
        all the elements after shuffling from the left side of the leftmost object to the right edge of the rightmost
        object.
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

                half_diff = diff / 2

                shifted_things[i, 0] = shifted_things[i, 0] - half_diff
                shifted_things[i + 1, 0] = shifted_things[i + 1, 0] + half_diff

                shift = True

        full_width = calculate_full_width(shifted_things)

        if iterations > 500:
            return shifted_things, full_width

    return shifted_things, full_width


class Label():
    """
    Simple Label class to handle combination and comparison of labels
    """

    def __init__(self, input_list):
        """
        Inititate with a list of 3 or 4 elements

        :param input_list: List[int, int, str, int]
            The listicle is a list containing the start_year, end_year, label text and an optional offset for when the
            text is drawn.
        """
        self.start = input_list[0]
        self.end = input_list[1]
        self.text = input_list[2]
        if len(input_list) == 4:
            self.offset = input_list[3]
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
        Adding two labels will add the text element together with a new-line character between elements. The end date
        is set to the later of the two end dates.

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
    """
    The main timeline class.
    """

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

            # Set the directory from the json file path
            new_timeline.dir = Path(filename).parents[0]
            new_timeline.start_year = metadata['start_year']
            new_timeline.end_year = metadata['end_year']

            # Make a label object for each label in the configuration file
            new_timeline.labels = []
            for item in metadata['labels']:
                new_timeline.labels.append(Label(item))

            new_timeline.images = metadata['images']

            # Merge labels to avoid over printing
            new_timeline.merge_labels()

        return new_timeline

    def merge_labels(self) -> None:
        """
        If there are multiple labels with the same start date then these are combined into a single label. The text
        elements are concatenated with a new-line character between them, which separates the text items to avoid
        overwriting

        :return: None
        """

        burner = copy.deepcopy(self.labels)
        new_labels = []

        # As long as there are labels left in the burner copy
        while len(burner) > 0:

            # Pop a label off the end of the list then iterate through the remaining items and look for matches.
            # If there are matches, combine the labels and add the later label to the remove list
            label1 = burner.pop()
            to_remove = []
            for label2 in burner:
                if label1 == label2:
                    label1 = label1 + label2
                    to_remove.append(label2)

            # If there were matches then remove any labels on the "to remove" list.
            if len(to_remove) > 0:
                for r in to_remove:
                    burner.remove(r)

            new_labels.append(label1)

        self.labels = new_labels

    @staticmethod
    def plot_objects(ax, objects, objects_orig, images):
        """
        Plot the images at the locations specified by the shuffled objects. The original object locations are also
        provided so that lines can be drawn connecting the image at its shuffled location with the poitn on the
        timeline to which it corresponds.

        :param ax: Axes
            Matplotlibe axis for plotting on.
        :param objects: ndarray
            An (n, 2) ndarray
        :param objects_orig:
        :param images:
        :return:
        """

        full_plot_width = ax.get_xlim()
        plot_width = full_plot_width[1] - full_plot_width[0]

        # Find the left edges and right edges and full width of the objects as arranged
        img_min = np.min(objects[:, 0] - objects[:, 1] / 2)
        img_max = np.max(objects[:, 0] + objects[:, 1] / 2)
        img_full_width = img_max - img_min

        # For each of the objects extract the
        for i in range(objects.shape[0]):
            tx = objects[i, 0]
            ty1 = objects[i, 2]
            ty2 = objects[i, 3]
            scaled_width = objects[i, 1]

            x0 = tx - scaled_width / 2
            x1 = tx + scaled_width / 2

            # If the images strip is wider than the whole width of the plot then scale down.
            if img_full_width > plot_width:
                offset = np.min(objects[:, 0] - objects[:, 1] / 2)
                x0 = (x0 - offset) * plot_width / img_full_width + full_plot_width[0]
                x1 = (x1 - offset) * plot_width / img_full_width + full_plot_width[0]
                tx = (tx - offset) * plot_width / img_full_width + full_plot_width[0]
            # if the image pokes out at either end, deal with it.
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

            # Plot the image
            ax.imshow(images[i], extent=[x0, x1, ty2, ty1], aspect='auto', zorder=100)

            # Draw a line connecting the image at the midpoint of its upper edge to the time line. These will appear
            # beneath the images.
            p = plt.plot(
                [tx, objects_orig[i, 0], objects_orig[i, 0]],
                [ty1, -10, 0],
                linewidth=5
            )
            # Draw a box around the image using whatever colour the line was drawn in. These will appear above the
            # images.
            plt.plot(
                [tx, x1, x1, x0, x0, tx],
                [ty1, ty1, ty2, ty2, ty1, ty1],
                linewidth=5, zorder=200, color=p[-1].get_color()
            )

    def sort_images(self, ax, n_rows):
        """
        Sort the images into n_rows and handle the image scaling into the appropriate coordinates.

        :param ax: Axes
            Matplotlibe axis on which the images and everything else will appear.
        :param n_rows: int
            Number of rows across which the images will be distributed.
        :return: (List , List)
            Returns two lists each with n_rows elements. Each of the elements contains an ndarray(n, 4) in size which
            contains all the scaled images.
        """
        high_image = True

        all_objects = []
        all_offsets = []
        all_images = []
        all_counts = []

        for i in range(n_rows):
            all_objects.append(np.zeros((len(self.images), 4)))
            all_images.append([])
            all_counts.append(0)
            all_offsets.append([])

        row_index = 0

        for label in self.images:
            year1 = label[0]
            offset = label[1]
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
            full_y = 100
            max_y = -15
            image_height_in_data = (full_y + max_y) / n_rows
            scale = (points_data1[1] - points_data0[1]) / image_height_in_data

            # Set the location of the image. Alternate images are plotted high and low.
            scaled_width = (points_data1[0] - points_data0[0]) / scale
            tx = mpl.dates.date2num(date(year1, 1, 1))

            ypos = max_y - row_index * image_height_in_data

            all_objects[row_index][all_counts[row_index], :] = np.array(
                [tx, scaled_width, ypos, ypos - image_height_in_data])
            all_images[row_index].append(img)
            if offset is not None:
                all_offsets[row_index].append(
                    mpl.dates.date2num(date(year1 + offset, 1, 1)) -
                    mpl.dates.date2num(date(year1, 1, 1))
                )
            else:
                all_offsets[row_index].append(offset)
            all_counts[row_index] += 1

            row_index += 1
            if row_index == n_rows:
                row_index = 0

        # Snip out any bits of the arrays we don't need
        for row_index in range(n_rows):
            all_objects[row_index] = all_objects[row_index][0:all_counts[row_index]]

        # Check whether there are any non-none offsets and replace Nones with zeros
        all_offsets = [parse_list(x) for x in all_offsets]

        return all_objects, all_images, all_offsets

    def plot_labels(self, ax):
        """
        Plot the labels on the provided axes

        :param ax: Axes
            Matplotlibe axes on which the labels will be plotted.
        :return: None
            Nowt
        """

        for label in self.labels:
            # Set up the range one year for labels with no end date or as specified
            tx = mpl.dates.date2num(date(label.start, 1, 1))
            tx2 = mpl.dates.date2num(date(label.start, 12, 31))
            if label.end is not None:
                tx2 = mpl.dates.date2num(date(label.end, 12, 31))

            # plot the text
            tx_with_offset = mpl.dates.date2num(date(label.start + label.offset, 1, 1))
            ax.text(tx_with_offset, 32, label.text, rotation=45)

            # plot the pips that indicate the range
            for i in range(5):
                delta = 5
                ax.add_patch(Rectangle((tx + delta, 1 + i * 6), tx2 - tx - 2 * delta, 5,
                                       color='green', edgecolor=None, alpha=0.5, linewidth=0))

    def plot_images(self, ax, n_rows):
        """
        Plot the images on the specified axes in n_rows rows.

        :param ax: Axes
            Matplotlib axes on which the images will be plotted.
        :param n_rows: int
            Number of rows of images to plot. don't use a negative number or zero, or kittens will die.
        :return: None
            Nothing
        """

        # Sort the images into n_rows strips by iterating through them, transform axes, scale etc.
        objects_orig, images, offsets = self.sort_images(ax, n_rows)

        # Jiggle the images in each strip so that they don't overlap
        fpw = ax.get_xlim()
        objects = []
        for i in range(n_rows):
            if None in offsets[i]:
                sorted_objects, _ = remove_overlaps_with_limits(objects_orig[i], fpw[0], fpw[1])
            else:
                sorted_objects, _ = use_offsets(objects_orig[i], offsets[i])
            objects.append(sorted_objects)

        # Now plot each strip out
        for i in range(n_rows):
            Timeline.plot_objects(ax, objects[i], objects_orig[i], images[i])

    def plot(self, ax, n_rows=2):
        """
        Plot the timeline on the specified indices

        :param ax: Axes
            Matplotlib axes on which the timeline will be plotted
        :param n_rows: int
            Number of rows across which the images will tbe spread
        :return:
        """
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
    parser = ArgumentParser()
    parser.add_argument("-c", "--configuration", help="Configuration file", required=True, type=str, dest='config_file')
    parser.add_argument("-o", "--output", help="Output file", required=True, type=str, dest='output_file')
    parser.add_argument("-r", "--rows", help="Number of rows of images in output image", default=2, type=int, dest='n_rows')

    args = parser.parse_args()

    f = Timeline.from_json(args.config_file)

    fig, axs = plt.subplots(1, sharex=True)
    fig.set_size_inches(16, 10)

    f.plot(axs, n_rows=args.n_rows)

    plt.savefig(args.output_file, bbox_inches="tight", dpi=300)
    plt.close()
