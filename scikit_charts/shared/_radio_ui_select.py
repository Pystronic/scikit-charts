"""
This file the implementation of a UI element select,
based on RadioButtons. Use this class to allow the use
to select which ui element is shown from a list.
Only the selected element is displayed, at the time.
"""
from typing import List, Tuple, Dict

import matplotlib.figure
import matplotlib.artist

from scikit_charts.shared import set_visible_deep
from scikit_charts.shared import AxesRadio


class RadioUISelect:
    """
    UI element which allows the user to select
    the currently displayed Artist from a list
    of elements. Only the selected element is
    displayed, while the rest is hidden.
    """
    _parent: matplotlib.figure.Figure
    _select: AxesRadio

    _tabs: List[List[matplotlib.artist.Artist]]
    _active_tab: int
    _label_dic: Dict[str, int]

    def __init__(
            self,
            fig: matplotlib.figure.Figure,
            tabs: List[List[matplotlib.artist.Artist]],
            labels: List[str],
            position: Tuple[float, float, float, float],
            title: str | None = None
    ):
        """
        Initializes a new instance of the UI select.

        :param fig: figure which contains the select and other elements
        :param tabs: 2D list of Artist groups which can be selected for display
        :param labels: labels which describe the selectable Artists
        :param position: position of the select within the figure
        :type position: Tuple(left, bottom, width, height)
        :param title: title displayed for the select
        :raises IndexError: thrown if the tabs and labels are not the same length
        """

        self._parent = fig
        self._tabs = tabs
        self._active_tab = 0
        self._label_dic = {}

        if len(tabs) != len(labels):
            raise IndexError("tab and label lists have to be the same length.")

        # map labels
        for i in range(len(labels)):
            self._label_dic[labels[i]] = i

        # prepare axes
        for i in range(1, len(self._tabs)):
            for artist in self._tabs[i]:
                set_visible_deep(artist, False)

        # init ui elements
        self._select = AxesRadio(
            self._parent,
            labels,
            self._active_tab,
            position,
            lambda label: self._on_selection_change(label),
            title=title
        )

    def _on_selection_change(self, label: str):
        """
        Called when the selected element changes. Hides the
        currently shown Artist and displays the new selection.
        :param label: label which is now selected
        """
        if label in self._label_dic:
            selection = self._label_dic[label]

            for artist in self._tabs[self._active_tab]:
                set_visible_deep(artist, False)
            for artist in self._tabs[selection]:
                set_visible_deep(artist, True)

            self._active_tab = selection
            self._parent.canvas.draw_idle()

    def get_axes(self) -> matplotlib.axes.Axes:
        """
        Return the axes element which contains the control.
        """
        return self._select.get_axes()
