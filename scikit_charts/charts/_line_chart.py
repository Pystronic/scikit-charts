"""
This file contains the implementation of the line chart.
Call line_chart() to create a new instance.

This chart is used to find patterns within the
target / prediction metrics, based on the original
order of the data.
"""

from typing import Union

import matplotlib.figure
import matplotlib.axes
import matplotlib.text

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backend_bases import PickEvent
from matplotlib.ticker import MaxNLocator
from pandas import DataFrame

from scikit_charts.metrics import PredictFunction, create_metrics, MetricEnum
from scikit_charts.shared import DataPlotMap, PickableLegend


class LineChart:
    """
    Class which implements the line chart. It should not be used
    directly. Instantiation should be done by calling line_chart().
    """
    _metrics: DataFrame
    _fig: matplotlib.figure.Figure
    _ax: matplotlib.axes.Axes

    _legend: PickableLegend
    _lines: DataPlotMap

    def __init__(self, metric_frame: DataFrame):
        self._metrics = metric_frame
        self._fig, self._ax = plt.subplots()

        # plot data
        self._lines = {
            MetricEnum.TARGET:
                self._ax.plot(
                    self._metrics.index,
                    self._metrics[MetricEnum.TARGET],
                    label=MetricEnum.TARGET.value
                ),
            MetricEnum.PREDICTION:
                self._ax.plot(
                    self._metrics.index,
                    self._metrics[MetricEnum.PREDICTION],
                    label=MetricEnum.PREDICTION.value
                )
        }

        # set axis labels
        self._ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self._ax.set_xlabel("index")
        self._ax.set_ylabel("target / prediction")
        self._ax.set_title("line chart")

        # init legend
        self._legend = PickableLegend(self._fig, loc="outside upper right")

        # connect events
        self._fig.canvas.mpl_connect("pick_event", lambda event: self._on_pick(event))

    def _on_pick(self, event: PickEvent):
        has_changes = self._legend.on_legend_pick(event, self._lines)

        if has_changes:
            self._fig.canvas.draw_idle()

    def get_figure(self) -> matplotlib.figure.Figure:
        """
        :return: matplot figure instance of the chart
        """
        return self._fig


def line_chart(
        x: Union[np.ndarray[float], list[list[float]]],
        y: Union[np.ndarray[float], list[float]],
        predict: PredictFunction
) -> matplotlib.figure.Figure:
    """
    Instantiate a new line chart instance from the given
    data and prediction. This chart should be used to find
    patterns in the target and prediction of the data, based
    on original order of the data.

    Use .show() on the returned element to display the chart.

    :param x: 2D-array of feature values, which was used to train the prediction model
    :param y: 1D-array of target values with the same length as x
    :param predict: function reference to the predict-function of the trained model
    :return: matplotlib.figure.Figure instance used to display the chart
    """
    # preparation
    metrics = create_metrics(x, y, predict)
    chart = LineChart(metrics)

    return chart.get_figure()
