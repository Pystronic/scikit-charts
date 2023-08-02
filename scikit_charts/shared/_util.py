"""
This module contains general type definitions and functions
which cannot be grouped more specifically.
"""
from enum import IntEnum
from typing import Dict, TypeAlias, List, Tuple, Callable

import PIL.Image
import matplotlib.lines
import matplotlib.artist
import matplotlib.axes
import matplotlib.figure
import numpy as np
from matplotlib.widgets import Button, Slider, CheckButtons

from scikit_charts.metrics import MetricEnum

MetricPlotMap: TypeAlias = Dict[
    MetricEnum,
    List[matplotlib.artist.Artist]
]
"""
Dictionary which maps a MetricEnum to the instances
of plotted data of an axes or figure object.
"""


class AxisEnum(IntEnum):
    """
    Simple enum to differentiate between X and Y axis.
    """
    X = 0
    Y = 1


def set_visible_deep(artist: matplotlib.artist.Artist, visible: bool):
    """
    Sets the visibility of the artist and its direct children.
    """
    artist.set_visible(visible)
    for child in artist.get_children():
        child.set_visible(visible)


def auto_limit_axis(
        axis: AxisEnum,
        ax: matplotlib.axes.Axes,
        data: np.ndarray[float]
) -> Tuple[float, float]:
    """
    Automatically update the limit of the
    Axis to fit the given data.

    :param axis: identifies the Axis to limit
    :param ax:  Axes which owns the Axis
    :param data: data which the limit is based on
    :returns: new limit of the Axis
    """
    data_range = (np.min(data), np.max(data))
    inset = (data_range[0] + data_range[1]) * 0.2
    data_range = (data_range[0] - inset, data_range[1] + inset)

    if axis == AxisEnum.X:
        ax.set_xlim(*data_range)
    else:
        ax.set_ylim(*data_range)
    return data_range


def indices_in_range(
        data: np.ndarray[float], data_range: Tuple[float, float]
) -> np.ndarray[float]:
    """
    Returns the indices of the elements in the array,
    which is within the given range.
    """
    return np.where(
        np.logical_and(
            data >= data_range[0],
            data <= data_range[1]
        )
    )


class AxesButton:
    """
    Utility class for button generation.
    Used to add a button to an Figure instance
    and manage all the required references for
    it to work.
    """
    _parent: matplotlib.figure.Figure
    _container: matplotlib.axes.Axes
    _button: Button
    _on_click: Callable[[], None]

    def __init__(
        self,
        parent: matplotlib.figure.Figure,
        label: str,
        on_click: Callable[[], None],
        position: Tuple[float, float, float, float],
        image: PIL.Image.Image | None = None
    ):
        """
        Initialise a new button on the Figure.

        :param parent: Figure which holds the button
        :param label: label of the button
        :param on_click: function which is called whe the
        button is clicked
        :param position: position within the parent Figure
        in which the button is placed
        :param image: image displayed on the button in
        addition to the label
        """
        self._parent = parent
        self._on_click = on_click

        # init button + container axes
        self._container = self._parent.add_axes(position)
        self._button = Button(self._container, label, image=image)
        self._button.on_clicked(self._on_click)

    def get_button(self) -> Button:
        """
        Return the Button element for customization.
        """
        return self._button

    def get_axes(self) -> matplotlib.axes.Axes:
        """
        Return the axes element which contains the control.
        """
        return self._container


class AxesSlider:
    """
    Utility class for slider generation.
    Used to add a slider to an Figure instance
    and manage all the required references for
    it to work.
    """
    _parent: matplotlib.figure.Figure
    _container: matplotlib.axes.Axes
    _slider: Slider
    _on_changed: Callable[[float], None]

    def __init__(
        self,
        parent: matplotlib.figure.Figure,
        label: str | None,
        val_range: Tuple[float, float],
        initial_val: float,
        on_changed: Callable[[float], None],
        position: Tuple[float, float, float, float]
    ):
        """
        Initialise a new slider on the Figure.

        :param parent: Figure which holds the Slider
        :param label: label of the Slider
        :param val_range: range (min, max) of the Slider
        :param initial_val: initial value of the slider
        :param on_changed: function which is called when the
        Slider position changed
        :param position: position within the parent Figure
        in which the Slider is placed
        """
        self._parent = parent
        self._on_changed = on_changed

        # init slider + container axes
        self._container = self._parent.add_axes(position)
        self._container.set_title(label, loc="left")

        self._slider = Slider(
            self._container,
            "",
            *val_range,
            initial_val,
            valfmt='%0.1f',
            valstep=0.1
        )
        self._slider.on_changed(self._on_changed)

    def set_slider_value(self, value: float):
        """
        Sets the current slider value, without
        triggering the callback,
        """
        self._slider.set_val(value)

    def get_slider(self) -> Slider:
        """
        Return the Slider element for customization.
        """
        return self._slider

    def get_axes(self) -> matplotlib.axes.Axes:
        """
        Return the axes element which contains the control.
        """
        return self._container


class AxesCheckboxes:
    """
    Utility class for checkbox generation.
    Used to add a checkbox to a Figure instance
    and manage all the required references for
    it to work.
    """
    _parent: matplotlib.figure.Figure
    _container: matplotlib.axes.Axes

    _check_buttons: CheckButtons
    _checked_dic: Dict[str, bool]
    _on_check_callback: Callable[[str, bool], None]

    def __init__(
        self,
        parent: matplotlib.figure.Figure,
        labels: List[str],
        title: str | None,
        on_check_changed: Callable[[str, bool], None],
        position: Tuple[float, float, float, float],
        active_labels: Tuple[str] = (),
    ):
        """
        Initialise new CheckButtons on the Figure.

        :param parent: Figure which holds the Slider
        :param labels: labels of the CheckButtons
        :param title: title for the container axes
        :param on_check_changed: function which is called when a
        CheckButton is changed
        :type on_check_changed: Callable[[label, check_state], [None]]
        :param position: position within the parent Figure
        in which the CheckButtons are placed
        :param active_labels: which labels are active on the start
        """
        self._parent = parent
        self._on_check_callback = on_check_changed

        # init checked dic
        self._checked_dic = {}
        for label in labels:
            self._checked_dic[label] = label in active_labels

        # init slider + container axes
        self._container = self._parent.add_axes(position)
        self._container.set_title(title)

        self._check_buttons = CheckButtons(
            self._container,
            labels,
            [self._checked_dic[label] for label in labels]
        )
        self._check_buttons.on_clicked(
            lambda label: self._on_check_changed(label)
        )

    def _on_check_changed(self, label: str):
        check_state = not self._checked_dic[label]
        self._checked_dic[label] = check_state
        self._on_check_callback(label, check_state)

    def get_check_buttons(self) -> CheckButtons:
        """
        Return the CheckButtons element for customization.
        """
        return self._check_buttons

    def get_axes(self) -> matplotlib.axes.Axes:
        """
        Return the axes element which contains the control.
        """
        return self._container

