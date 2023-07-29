"""
This file contains the implementation of the scatter plot.
Call scatter_plot() to create a new instance.

This chart is used to compare the results of the prediction
to the actual target values. This can be used to find
systematic errors within a model. The close the points
are to the 45-degree line, the better is the prediction.
"""

from typing import Union

import matplotlib.figure
import matplotlib.collections
import matplotlib.axes
import matplotlib.text

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame

from scikit_charts.metrics import PredictFunction, create_metrics, MetricEnum


class ScatterPlot:
    """
    Class which implements the scatter plot. It should not be used
    directly. Instantiation should be done by calling scatter_plot().
    """
    _metrics: DataFrame
    _fig: matplotlib.figure.Figure
    _ax: matplotlib.axes.Axes

    _plot_data: matplotlib.collections.PathCollection

    def __init__(self, metric_frame: DataFrame):
        self._metrics = metric_frame
        self._fig, self._ax = plt.subplots()

        # set axis labels
        self._ax.set_xlabel("target")
        self._ax.set_ylabel("prediction")
        self._ax.set_title("scatter plot")
        self._ax.grid(visible=True, linewidth=0.5)
        self._ax.axline((0, 0), (1, 1), color="darkgrey")

        # plot data
        self._plot_data = self._ax.scatter(
            self._metrics[MetricEnum.TARGET],
            self._metrics[MetricEnum.PREDICTION],
        )

    def get_figure(self) -> matplotlib.figure.Figure:
        """
        :return: matplot figure instance of the chart
        """
        return self._fig


def scatter_plot(
        x: Union[np.ndarray[float], list[list[float]]],
        y: Union[np.ndarray[float], list[float]],
        predict: PredictFunction
) -> matplotlib.figure.Figure:
    """
    Instantiate a new scatter plot instance from the given
    data and prediction. This chart should be used to check
    the prediction for systematic errors. It also shows how
    close the prediction is to the targeted values.

    Use .show() on the returned element to display the chart.

    :param x: 2D-array of feature values, which was used to train the prediction model
    :param y: 1D-array of target values with the same length as x
    :param predict: function reference to the predict-function of the trained model
    :return: matplotlib.figure.Figure instance used to display the chart
    """
    # preparation
    metrics = create_metrics(x, y, predict)
    plot = ScatterPlot(metrics)

    return plot.get_figure()
