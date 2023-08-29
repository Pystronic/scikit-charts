"""
This file contains the implementation of the bubble chart.
Call bubble_chart() to create a new instance.

This chart is used to compare multiple metrics at once.
Metrics can be applied dynamically to both axis. It
is also possible to adjust the bubble size and change
the bubble color.
"""
from pathlib import Path
from typing import Union, List, Final

import matplotlib.figure
import matplotlib.collections
import matplotlib.axes
import matplotlib.text
import matplotlib.colors
import matplotlib.cm
import PIL.Image

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame

from scikit_charts.metrics import PredictFunction, create_metrics, MetricEnum, METRIC_DTYPE
from scikit_charts.shared import (RadioUISelect, AxesRadio, AxesButton, AxesSlider,
                                  auto_limit_axis, DEFAULT_PLOT_COLOR, AxisEnum)

############## Constants ###################

CONSTANT_BUBBLE_LABEL: Final[str] = "Constant size"
CONSTANT_BUBBLE_SIZE: Final[float] = matplotlib.rcParams['lines.markersize'] ** 2
INDEX_DATA_LABEL: Final[str] = "index"

PADDING: Final[float] = 0.05
UI_SIDEBAR_WIDTH: Final[float] = 0.24
# (left, bottom, width, height) in %
UI_AREA = (PADDING, PADDING, UI_SIDEBAR_WIDTH, 0.65)
SELECTION_AREA = (
    PADDING, UI_AREA[3] + PADDING * 2,
    UI_SIDEBAR_WIDTH, 1 - (UI_AREA[3] + PADDING * 3)
)

BTN_COLOR_IMAGE_X: Final[PIL.Image.Image] = \
    PIL.Image.open(Path(__file__).parent.joinpath("img/color_btn_x.png"))
BTN_COLOR_IMAGE_Y: Final[PIL.Image.Image] = \
    PIL.Image.open(Path(__file__).parent.joinpath("img/color_btn_y.png"))


#############################################

class BubbleChart:
    """
    Class which implements the bubble chart. It should not be used
    directly. Instantiation should be done by calling bubble_chart().
    """

    # base chart data
    _fig: matplotlib.figure.Figure
    _ax: matplotlib.axes.Axes
    _metrics: DataFrame

    # ui axes
    _tab_select: RadioUISelect
    _x_radio: AxesRadio
    _y_radio: AxesRadio
    _bubble_radio: AxesRadio
    _additional_options: matplotlib.axes.Axes

    # additional options ui elements
    _x_color_btn: AxesButton
    _y_color_btn: AxesButton
    _alpha_slider: AxesSlider
    _reset_btn: AxesButton

    # selected data labels
    _x_label: MetricEnum
    _y_label: MetricEnum
    _bubble_label: str

    # coloring options data
    _color_data: np.ndarray[float]
    _color_map: matplotlib.colors.Colormap | None = None
    _color_norm: matplotlib.colors.Normalize | None = None
    _transparency: float = 1

    # plotted data
    _x_data: np.ndarray[float]
    _y_data: np.ndarray[float]
    _bubble_size: np.ndarray[float]

    _plot_data: matplotlib.collections.PathCollection | None = None

    def __init__(self, metric_frame: DataFrame):
        self._metrics = metric_frame
        self._fig, self._ax = plt.subplots()
        self._fig.subplots_adjust(left=UI_SIDEBAR_WIDTH + PADDING * 2 + 0.04)

        # set axis labels
        self._ax.set_title("bubble chart")
        self._ax.grid(visible=True, linewidth=0.5)

        # init ui elements
        self._init_data_selection()
        self._init_additional_options()

        # init ui selection area
        self._tab_select = RadioUISelect(
            self._fig,
            [
                [self._x_radio.get_axes()],
                [self._y_radio.get_axes()],
                [self._bubble_radio.get_axes()],
                [
                    self._x_color_btn.get_axes(),
                    self._y_color_btn.get_axes(),
                    self._alpha_slider.get_axes(),
                    self._reset_btn.get_axes(),
                    self._additional_options
                ]
            ],
            ["X data", "Y data", "Bubble size", "Other options"],
            SELECTION_AREA,
            "Display options for"
        )

        # plot initial data
        self._x_label = None
        self._y_label = None
        self._bubble_label = None
        self._update_plot_data(
            MetricEnum.PREDICTION,
            MetricEnum.TARGET,
            CONSTANT_BUBBLE_LABEL,
            first_draw=True
        )

    def _init_data_selection(self):
        """
        initialize the data selection ui areas for
        x-axis, y-axis and bubble size.
        """
        column_list: List[str] = [INDEX_DATA_LABEL] + list(self._metrics.columns)
        self._x_radio = AxesRadio(
            self._fig,
            column_list,
            column_list.index(MetricEnum.PREDICTION),
            UI_AREA,
            lambda label: self._update_plot_data(x_label=label),
            "X data"
        )

        self._y_radio = AxesRadio(
            self._fig,
            column_list,
            column_list.index(MetricEnum.TARGET),
            UI_AREA,
            lambda label: self._update_plot_data(y_label=label),
            "Y data"
        )

        self._bubble_radio = AxesRadio(
            self._fig,
            [CONSTANT_BUBBLE_LABEL] + column_list,
            0,
            UI_AREA,
            lambda label: self._update_plot_data(bubble_label=label),
            "Bubble size"
        )

    def _init_additional_options(self):
        """
        initialize the ui areas containing
        additional options for interaction.
        """
        # base axes
        self._additional_options = self._fig.add_axes(UI_AREA)
        self._additional_options.set_title("Other options")
        self._additional_options.tick_params(
            which='both', bottom=False, left=False,
            labelbottom=False, labelleft=False
        )

        # coloring buttons
        self._x_color_btn = AxesButton(
            self._fig,
            "",
            lambda event: self.color_axis(AxisEnum.X),
            (PADDING, 0.58, UI_SIDEBAR_WIDTH / 2, 0.1),
            image=BTN_COLOR_IMAGE_X
        )
        self._y_color_btn = AxesButton(
            self._fig,
            "",
            lambda event: self.color_axis(AxisEnum.Y),
            (PADDING + UI_SIDEBAR_WIDTH / 2, 0.58, UI_SIDEBAR_WIDTH / 2, 0.1),
            image=BTN_COLOR_IMAGE_Y
        )

        # alpha slider
        def on_transparency_changed(new_value: float):
            self._transparency = new_value
            self._update_plot_data(update_color=True)

        self._transparency = 0
        self._alpha_slider = AxesSlider(
            self._fig,
            "Transparency:",
            (0, 1),
            self._transparency,
            on_transparency_changed,
            (PADDING + 0.01, 0.40, UI_SIDEBAR_WIDTH - PADDING, 0.1),
        )

        # reset button
        self._reset_btn = AxesButton(
            self._fig,
            "Reset chart",
            lambda event: self.reset(),
            (PADDING, PADDING, UI_SIDEBAR_WIDTH, 0.1)
        )

    def _update_x_data(self, x_label: MetricEnum | None) -> bool:
        if x_label is not None:
            self._x_label = x_label
            self._ax.set_xlabel(x_label)

            if x_label == INDEX_DATA_LABEL:
                self._x_data = self._metrics.index
            else:
                self._x_data = self._metrics[x_label]

            auto_limit_axis(AxisEnum.X, self._ax, self._x_data)

            return True
        return False

    def _update_y_data(self, y_label: MetricEnum | None) -> bool:
        if y_label is not None:
            self._y_label = y_label
            self._ax.set_ylabel(y_label)

            if y_label == INDEX_DATA_LABEL:
                self._y_data = self._metrics.index
            else:
                self._y_data = self._metrics[y_label]

            auto_limit_axis(AxisEnum.Y, self._ax, self._y_data)

            return True
        return False

    def _update_bubble_size(self, bubble_label: MetricEnum | None) -> bool:
        if bubble_label is not None:
            self._bubble_label = bubble_label
            if bubble_label == CONSTANT_BUBBLE_LABEL:
                self._bubble_size = np.full(
                    len(self._y_data),
                    fill_value=CONSTANT_BUBBLE_SIZE,
                    dtype=METRIC_DTYPE
                )
            else:
                data: np.ndarray
                if bubble_label == INDEX_DATA_LABEL:
                    data = self._metrics.index.astype(METRIC_DTYPE)
                else:
                    data = self._metrics[bubble_label]

                # Normalize size of bubbles
                min_value: float = np.min(data)

                if min_value < CONSTANT_BUBBLE_SIZE / 2:
                    data = np.copy(data)
                    data += abs(min_value) + CONSTANT_BUBBLE_SIZE / 2

                self._bubble_size = data

            return True
        return False

    def _update_plot_data(
            self,
            x_label: MetricEnum | None = None,
            y_label: MetricEnum | None = None,
            bubble_label: str | None = None,
            update_color: bool = False,
            first_draw: bool = False
    ):
        """
        Updates the plotted data of the chart, based
        on the given data. The parameters describe
        which data was changed and has to be updated.

        :param x_label: new label of the x-axis data
        :param y_label: new label of th y-axis data
        :param bubble_label: new label of the bubble-size data
        :param update_color: if the color options changed and
        require a redraw
        :param first_draw: if this is the first draw on the chart
        """
        # update attributes
        has_changes = self._update_x_data(x_label)
        has_changes = self._update_y_data(y_label) or has_changes
        has_changes = self._update_bubble_size(bubble_label) or has_changes

        # plot data
        if has_changes or update_color:
            if not first_draw and self._plot_data is not None:
                self._plot_data.remove()
                self._plot_data = None

            color_data: str | np.ndarray[float]
            if self._color_map is not None and self._color_norm is not None:
                color_data = self._color_data
            else:
                color_data = DEFAULT_PLOT_COLOR

            self._plot_data = self._ax.scatter(
                self._x_data,
                self._y_data,
                self._bubble_size,
                color_data,
                cmap=self._color_map,
                norm=self._color_norm,
                alpha=1-self._transparency
            )
            self._fig.canvas.draw_idle()

    def reset(self):
        """
        Resets the chart data and options to the initial values.
        """
        # reset data
        self._color_map = None
        self._color_norm = None
        self._transparency = 1

        # reset interactive ui
        self._x_radio.set_selected_label(MetricEnum.PREDICTION)
        self._y_radio.set_selected_label(MetricEnum.TARGET)
        self._bubble_radio.set_selected_label(CONSTANT_BUBBLE_LABEL)
        self._alpha_slider.set_slider_value(1)

        # redraw plot
        self._update_plot_data(
            MetricEnum.PREDICTION,
            MetricEnum.TARGET,
            CONSTANT_BUBBLE_LABEL
        )

    def color_axis(self, axis: AxisEnum):
        """
        Add colorization to the chart, based on the X or Y axis.
        The color of the individual points will be normalized
        based on the min and max value of the axis-data.

        :param axis: axis based on which the data is colorized
        """
        # set data used tod calculate color
        if axis == AxisEnum.X:
            self._color_data = self._x_data
        elif axis == AxisEnum.Y:
            self._color_data = self._y_data
        else:
            return

        # initialize colorization
        self._color_map = matplotlib.colormaps["gist_rainbow_r"]
        self._color_norm = matplotlib.colors.Normalize()
        self._update_plot_data(update_color=True)

    def get_figure(self) -> matplotlib.figure.Figure:
        """
        :return: matplot figure instance of the chart
        """
        return self._fig


def bubble_chart(
        x: Union[np.ndarray[float], list[list[float]]],
        y: Union[np.ndarray[float], list[float]],
        predict: PredictFunction
) -> matplotlib.figure.Figure:
    """
    Instantiate a new bubble chart instance from the given
    data and prediction. This chart can be used to compare
    multiple metrics at once. The displayed data can be
    changed dynamically. It also improves upon the
    scatter-plot by allowing dynamic bubble sizes based
    on metrics and adding coloring options to the bubbles.

    Use .show() on the returned element to display the chart.

    :param x: 2D-array of feature values, which was used to train the prediction model
    :param y: 1D-array of target values with the same length as x
    :param predict: function reference to the predict-function of the trained model
    :return: matplotlib.figure.Figure instance used to display the chart
    """
    # preparation
    metrics = create_metrics(x, y, predict)
    plot = BubbleChart(metrics)

    return plot.get_figure()
