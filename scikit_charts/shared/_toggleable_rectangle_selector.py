"""
This file contains a utility implementation of
RectangleSelector. It can be toggled on and
off by a connected CheckButton.
"""
from typing import Callable, Tuple

import matplotlib.figure
import matplotlib.axes
import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import RectangleSelector

from scikit_charts.shared import AxesCheckboxes


class ToggleableRectangleSelector:
    """
    Utility implementation of RectangleSelector
    which is connected to a CheckButton. The
    CheckButton controls if the selector is
    active and if the selection-area is shown.
    """
    _parent: matplotlib.figure.Figure
    _target: matplotlib.axes.Axes
    _check_boxes: AxesCheckboxes
    _selector: RectangleSelector

    _on_select_callback: Callable[[Tuple[float, float], Tuple[float, float]], None]
    _on_active_changed: Callable[[bool], None] | None

    def __init__(
            self,
            fig: matplotlib.figure.Figure,
            ax: matplotlib.axes.Axes,
            select_callback: Callable[[Tuple[float, float], Tuple[float, float]], None],
            check_label: str,
            check_pos: Tuple[float, float, float, float],
            on_active_changed: Callable[[bool], None] | None = None
    ):
        """
        Instantiates a new ToggleableRectangleSelector.
        It can be toggled on and off by a connected CheckButton.

        :param fig: Figure which own the CheckButton
        :param ax: Axes which is targeted by the Selector
        :param select_callback: called each time the selection
        area is changed
        :param check_label: label of the connected CheckButton
        :param check_pos: position of the connected CheckButton
        :param on_active_changed: called when the selection
        state is toggled
        """
        # initialise data
        self._parent = fig
        self._target = ax
        self._on_select_callback = select_callback
        self._on_active_changed = on_active_changed

        # initialise ui
        self._check_boxes = AxesCheckboxes(
            self._parent,
            [check_label],
            None,
            lambda label, state: self._toggle_selection(state),
            check_pos
        )

        # init selector
        self._selector = RectangleSelector(
            self._target,
            lambda e_click, e_release: self._on_select(e_click, e_release),
            interactive=True,
            use_data_coordinates=True
        )
        self._selector.set_active(False)

    def _toggle_selection(self, new_state: bool):
        self._selector.set_active(new_state)

        if not new_state:
            self._selector.clear()

        if self._on_active_changed is not None:
            self._on_active_changed(new_state)

    def _on_select(self, e_click: MouseEvent, e_release: MouseEvent):
        min_pos = (e_click.xdata, e_click.ydata)
        max_pos = (e_release.xdata, e_release.ydata)

        if np.sum(min_pos) > np.sum(max_pos):
            tmp = min_pos
            min_pos = max_pos
            max_pos = tmp

        self._on_select_callback(min_pos, max_pos)

    def is_active(self) -> bool:
        """
        Return if the selection is currently active.
        """
        return self._selector.get_active()

    def get_button_axes(self) -> matplotlib.axes.Axes:
        """
        Return the axes element which contains the control.
        """
        return self._check_boxes.get_axes()
