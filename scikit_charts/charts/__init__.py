"""
The :mod:`scikit_charts.charts` module contains functions and implementations for the charts
within this library. Each chart is instantiated by a single function call and
displayed by calling the plot-function of the returned object.
"""

from ._line_chart import line_chart
from ._scatter_plot import scatter_plot

__all__ = [
    "line_chart",
    "scatter_plot"
]
