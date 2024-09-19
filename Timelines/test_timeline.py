import pytest
import numpy as np
from Timelines.timeline_plotter import remove_overlaps, remove_overlaps_with_limits, parse_list


def test_parse_list_all_none():
    test_list = [None, None, None]
    return_list = parse_list(test_list)
    for item in return_list:
        assert item is None


def test_parse_list_one_zero():
    test_list = [None, 0.0, None]
    return_list = parse_list(test_list)
    for item in return_list:
        assert item == 0.0


def test_parse_list_one_non_none():
    test_list = [None, 2.0, None]
    return_list = parse_list(test_list)
    expected_list = [0, 2, 0]
    for i in range(3):
        assert return_list[i] == expected_list[i]


def test_parse_list_all_numbers():
    test_list = [1, 2, 3]
    return_list = parse_list(test_list)

    for i in range(3):
        assert test_list[i] == return_list[i]


def test_overlap_bounds_overfull():
    things = np.zeros((2, 2))
    things[0, :] = np.array([0, 5])
    things[1, :] = np.array([4, 5])
    moved_objects, full_width = remove_overlaps_with_limits(things, 0, 8)

    assert full_width == 10
    assert moved_objects[0, 0] - moved_objects[0, 1] / 2 == 0


def test_overlap_bounds_two_objects():
    things = np.zeros((2, 2))
    things[0, :] = np.array([0, 5])
    things[1, :] = np.array([20, 5])
    moved_objects, full_width = remove_overlaps_with_limits(things, 0, 20)


def test_overlap_bounds_four_objects():
    things = np.zeros((4, 2))
    things[0, :] = np.array([3, 5])
    things[1, :] = np.array([10, 5])
    things[2, :] = np.array([15, 5])
    things[3, :] = np.array([20, 5])
    moved_objects, full_width = remove_overlaps_with_limits(things, 0, 20.1)


def test_overlap_removal():
    things = np.zeros((4, 2))

    things[0, :] = np.array([0, 4])
    things[1, :] = np.array([1, 4])
    things[2, :] = np.array([2, 4])
    things[3, :] = np.array([10, 4])

    moved_objects, full_width = remove_overlaps(things)

    # Final object shouldn't move
    assert moved_objects[3, 0] == things[3, 0]

    # None of the objects should change size
    for i in range(things.shape[0]):
        assert moved_objects[i, 1] == things[i, 1]


def test_overlap_removal_many_objects():
    n_objects = 20
    things = np.zeros((n_objects, 2))
    for i in range(n_objects):
        things[i, :] = np.array([i * 0.5, 4])
    moved_objects, full_width = remove_overlaps(things)
    # None of the objects should change size
    for i in range(things.shape[0]):
        assert moved_objects[i, 1] == things[i, 1]
    assert n_objects * 4 == pytest.approx(moved_objects[-1, 0] - moved_objects[0, 0] + 4, 0.0001)
