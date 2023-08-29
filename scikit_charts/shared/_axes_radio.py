"""
The file contains code for a basic ui element
of a radio select box. Parameters like labels
and a function to react on changes can be
customized on creation.
"""
from typing import Tuple, List, Callable

import matplotlib.figure
import matplotlib.axes
from matplotlib.widgets import RadioButtons


class AxesRadio:
    """
    Basic ui element which creates a vertical box containing
    radio buttons. The displayed labels, position and the action
    on a selection change can be customised.
    """
    _parent: matplotlib.figure.Figure
    _container: matplotlib.axes.Axes
    _radio: RadioButtons

    _labels: List[str]
    _on_selection_changed: Callable[[str, int], None]

    def __init__(
            self,
            fig: matplotlib.figure.Figure,
            labels: List[str],
            active_label: int,
            position: Tuple[float, float, float, float],
            on_selection_changed: Callable[[str], None],
            title: str | None = None,
    ):
        """
        Instantiate a new instance of AxesRadio.

        :param fig: figure which contains the select
        :param labels: labels which can be selected from
        :param active_label: index of the label which is active at the start
        :param position: position of the select within the figure
        :type position: Tuple(left, bottom, width, height)
        :param on_selection_changed: function which is called when the
        selected label changes
        :param title: title displayed over the select or None
        """
        # init data
        self._parent = fig
        self._labels = labels
        self._on_selection_changed = on_selection_changed

        # prepare ui
        self._container = self._parent.add_axes(position)
        self._container.set_title(title)
        self._radio = RadioButtons(self._container, labels, active=active_label)
        self._radio.on_clicked(self._on_selection_changed)

    def set_selected_label(self, label: str):
        """
        Sets the currently selected label of the
        ui element.
        """

        if label in self._labels:
            self._radio.set_active(self._labels.index(label))

    def get_axes(self) -> matplotlib.axes.Axes:
        """
        Return the axes element which contains the control.
        """
        return self._container
