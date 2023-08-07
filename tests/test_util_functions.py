import matplotlib.pyplot
import matplotlib.axes
import numpy as np

from scikit_charts.shared import set_visible_deep, auto_limit_axis, AxisEnum, args_in_range, arg_nearest


def test_set_visible_deep_whenArtistWithChildren_allInvisible():
    fig = matplotlib.pyplot.Figure()
    axes = [fig.add_subplot() for i in range(4)]
    ax: matplotlib.axes.Axes

    set_visible_deep(fig, False)

    assert fig.get_visible() is False
    assert all(ax.get_visible() is False for ax in axes)

def test_auto_limit_axis_withValuesForX_setCorrectLimit():
    fig, ax = matplotlib.pyplot.subplots()
    data_y = np.arange(5)
    data_x = [1, 3, 2, 5, 9]
    min_lim = 1 - 8 * 0.2
    max_lim = 9 + 8 * 0.2
    ax.plot(data_x, data_y)

    auto_limit_axis(AxisEnum.X, ax, data_x)

    assert ax.get_xlim() == (min_lim, max_lim)

def test_auto_limit_axis_withValuesForY_setCorrectLimit():
    fig, ax = matplotlib.pyplot.subplots()
    data_y = [1, 3, 2, 5, 9]
    data_x = np.arange(5)
    min_lim = 1 - 8 * 0.2
    max_lim = 9 + 8 * 0.2
    ax.plot(data_x, data_y)

    auto_limit_axis(AxisEnum.Y, ax, data_y)

    assert ax.get_ylim() == (min_lim, max_lim)

def test_args_in_range_withValuesInOrder_returnInRange():
    data = np.asarray([1, 3, 6, 7.4, 7.9, 8, 16, 20])
    v_range = (3.4, 9)
    expected = np.asarray([2, 3, 4, 5])

    result = args_in_range(data, v_range)

    assert np.array_equal(result, expected)

def test_args_in_range_withValuesOutOfOrder_returnInRange():
    data = np.asarray([5, 7, 29, 4.8, 90.1, 231, 5, 6, 3.2])
    v_range = (4.8, 90)
    expected = np.asarray([0, 1, 2, 3, 6, 7])

    result = args_in_range(data, v_range)

    assert np.array_equal(result, expected)

def test_arg_nearest_withIntegerValues_returnNearest():
    data = np.asarray([1, 3, 5, 6, 9, 10])
    val = 7
    expected = 3

    result = arg_nearest(data, val)

    assert result == expected


def test_arg_nearest_withFloatValues_returnNearest():
    data = np.asarray([1.4, 3.2, 5.6, 6.02, 9.0, 10.13])
    val = 4.3
    expected = 1

    result = arg_nearest(data, val)

    assert result == expected

def test_arg_nearest_withSameDistance_returnLower():
    data = np.asarray([1,2])
    val = 1.5
    expected = 0

    result = arg_nearest(data, val)

    assert result == expected
