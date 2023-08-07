"""
This file contains the implementation for
a draggable axvline. The utility class
automatically setups the callbacks and
allows others to react to the
end of the drag.
"""
from typing import  Callable

import matplotlib.axes
import matplotlib.lines
from matplotlib.offsetbox import DraggableBase
from matplotlib.backend_bases import KeyEvent, MouseEvent
from numpy.typing import ArrayLike

from scikit_charts.shared import arg_nearest


class DraggableAxvline(DraggableBase):
    """
    Utility class for a draggable implementation
    of axes.axvline. The line is customizable, and
    it is possible to react to the end of the drag
    event.

    Notice: Since the pick_event does not work
    for multiple axes in a figure, this class
    also contains a workaround using the
    button press event.
    """
    # ui elements
    _owner: matplotlib.axes.Axes
    _line: matplotlib.lines.Line2D
    _pick_workaround: bool
    _on_drag_finish: Callable[[float, int | None], None] | None

    # metric data
    _current_x: float
    _y_min: float
    _y_max: float
    _snapping: ArrayLike | None

    # pixel data
    _axes_width: int = 1
    _last_x_pos: int = 0

    def __init__(
            self,
            axes: matplotlib.axes.Axes,
            x: float = 0,
            ymin: float = 0,
            ymax: float = 1,
            on_drag_finish: Callable[[float, int | None], None] | None = None,
            snapping: ArrayLike | None = None,
            use_blit=True,
            pick_workaround=False,
            **kwargs
    ):
        """
        New instance of a draggable axvline implementation.

        :param axes: axes on which the line is drawn
        :param x: x position of the line
        :param ymin: bottom start of the line between (0, 1)
        :param ymax: top end of the line between (0, 1)
        :param on_drag_finish: callback which receives the
        new x value and optional snapping index once the drag
        finishes
        :param snapping: list of x values, to which the
        line should snap at the end of dragging
        :param use_blit: if blitting should be activated
        for the line
        :param pick_workaround: if the workaround for
        pick events should be used. Has to be set to
        support Figures with multiple Axes
        :param kwargs: valid keyword arguments are `.Line2D`
        properties, except for 'transform'
        """
        # save attributes
        self._owner = axes
        self._current_x = x
        self._y_min = ymin
        self._y_max = ymax
        self._snapping = snapping
        self._pick_workaround = pick_workaround
        self._on_drag_finish = on_drag_finish

        # init baseline
        self._line = self._owner.axvline(
            self._current_x,
            self._y_min,
            self._y_max,
            **kwargs
        )
        self._line.set_picker(10)

        super().__init__(self._line, use_blit=use_blit)

        # register workaround
        if self._pick_workaround:
            self._owner.figure.canvas.mpl_connect(
                "button_press_event",
                lambda event: self._on_button_press(event)
            )

    def _on_button_press(self, event: KeyEvent):
        """
        Handle key events for the pick workaround.
        """
        if event.inaxes is not self._owner:
            return
        if not issubclass(type(event), MouseEvent):
            return
        if not self._line.contains(event):
            return

        # forward pick event
        self._line.pick(event)

    def save_offset(self):
        """
        Called when the object is picked for dragging; should save the
        reference position of the artist.
        """
        bbox = self._owner.get_window_extent().transformed(
            self._owner.figure.dpi_scale_trans.inverted()
        )
        self._axes_width = int(bbox.width * self._owner.figure.dpi)

        # get relative pixel position from selected x
        x_lim = self._owner.get_xlim()
        relative_x = self._current_x / (abs(x_lim[1] - x_lim[0]))

        self._last_x_pos = int(relative_x * self._axes_width)

    def update_offset(self, dx, dy):
        """
        Called during the dragging; (*dx*, *dy*) is the pixel offset from
        the point where the mouse drag started.
        """
        # get relative pixel position
        current_pos = self._last_x_pos + dx
        relative_pos = current_pos / self._axes_width
        x_lim = self._owner.get_xlim()

        # convert pixel pos to data value
        self._current_x = x_lim[0] + relative_pos * abs(x_lim[1] - x_lim[0])
        # noinspection PyTypeChecker
        self._line.set_xdata([self._current_x])

    def finalize_offset(self):
        """Called when the mouse is released."""
        i_snap: int | None = None
        if self._snapping is not None:
            i_snap = arg_nearest(self._snapping, self._current_x)
            self.set_current_x(
                self._snapping[i_snap]
            )

        if self._on_drag_finish is not None:
            self._on_drag_finish(self._current_x, i_snap)

    def get_current_x(self):
        """
        Get x data value of the current line position.
        """
        return self._current_x

    def set_current_x(self, value: float):
        """
        Set x data value of the current line position
        and redraw.
        """
        self._current_x = value
        # noinspection PyTypeChecker
        self._line.set_xdata([self._current_x])
        self._owner.figure.canvas.draw_idle()
