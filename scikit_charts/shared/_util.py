"""
This module contains general type definitions and functions
which cannot be grouped more specifically.
"""
from typing import Dict, TypeAlias, List

import matplotlib.lines

from scikit_charts.metrics import MetricEnum

MetricPlotMap: TypeAlias = Dict[
    MetricEnum,
    List[matplotlib.lines.Line2D]
]
"""
Dictionary which maps a MetricEnum to the instances
of plotted data of an axes or figure object.
"""
