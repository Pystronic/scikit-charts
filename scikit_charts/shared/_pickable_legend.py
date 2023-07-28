"""
Generic implementation of a legend with pickable entries for charts.
Toggle if the displayed data is shown by clicking the entry
in the legend.

This requires a setup of the pick-event: fig.canvas.mpl_connect("pick_event", on_pick)
"""
from typing import TypeAlias, Dict, Tuple, Union

import matplotlib.figure
import matplotlib.legend
import matplotlib.lines
import matplotlib.text
from matplotlib.backend_bases import PickEvent

from scikit_charts.metrics import MetricEnum
from ._util import MetricPlotMap

MetricLegendMap: TypeAlias = Dict[
    MetricEnum,
    Tuple[matplotlib.lines.Line2D, matplotlib.text.Text]
]
"""
Dictionary which maps a MetricEnum of the plotted data to
the corresponding line + label in the legend.
"""
PICKER_OFFSET: int = 6
"""
Offset interval for pickable entries in the legend.
"""


def init_pickable_legend(
        owner: Union[matplotlib.figure.Figure, matplotlib.axes.Axes],
        loc: str | None = None
) -> MetricLegendMap:
    """
    Initialise a legend with pickable entries for
    the given figure instance. The returned map
    should be stored for responding to the pick event.
    Labels need to be consistent between legend and plotted data.

    :param owner: figure / axes which owns the legend and data
    :param loc: placement of the legend; same as legend(loc)
    :return: map which is used in the pick event to identify the
    legend entries by metric
    """
    # configure legend
    legend: matplotlib.legend.Legend = owner.legend(loc=loc)

    line: matplotlib.lines.Line2D
    text: matplotlib.text.Text
    legend_dict: MetricLegendMap = {}

    # Map plotted lines + texts to legend handlers
    for line in legend.get_lines():
        line.set_picker(PICKER_OFFSET)
        legend_dict[line.get_label()] = [line]
    for text in legend.get_texts():
        text.set_picker(PICKER_OFFSET)
        legend_dict[text.get_text()].append(text)

    return legend_dict


def on_legend_pick(
    event: PickEvent,
    plot_map: MetricPlotMap,
    leg_map: MetricLegendMap
) -> bool:
    """
    Should be called within on_pick to react to the
    pick-events of legend entries.

    :param event: PickEvent which was passed to on_pick
    :param plot_map: mapping from MetricEnum to plotted data
    :param leg_map: mapping from MetricEnum to corresponding
    legend entry
    :return: if the ui changed and requires a redraw
    """
    event_target = event.artist
    label: MetricEnum

    if isinstance(event_target, matplotlib.lines.Line2D):
        label = event_target.get_label()
    elif isinstance(event_target, matplotlib.text.Text):
        label = event_target.get_text()
    else:
        return False

    # Label does not exist for some reason
    if label not in plot_map:
        return False

    picked_lines = plot_map[label]
    visible = False
    for line in picked_lines:
        visible = visible or not line.get_visible()
        line.set_visible(visible)

    # Change the alpha on the line in the legend, so we can see what lines
    # have been toggled.
    for entry in leg_map[label]:
        entry.set_alpha(1.0 if visible else 0.2)

    return True
