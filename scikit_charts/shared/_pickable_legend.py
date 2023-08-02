"""
Generic implementation of a legend with pickable entries for charts.
Toggle if the displayed data is shown by clicking the entry
in the legend.

This requires a setup of the pick-event: fig.canvas.mpl_connect("pick_event", on_pick)
"""
from typing import Dict, Union, List, Tuple

import matplotlib.artist
import matplotlib.figure
import matplotlib.legend
import matplotlib.lines
import matplotlib.patches
import matplotlib.text
from matplotlib.backend_bases import PickEvent

from scikit_charts.metrics import MetricEnum
from scikit_charts.shared import MetricPlotMap

PICKER_OFFSET: int = 6
"""
Offset interval for pickable entries in the legend.
"""

class PickableLegend:
    """
    Legend implementation which allows to
    toggle the display of data points by clicking
    the label in the legend.
    """
    _owner: Union[matplotlib.figure.Figure, matplotlib.axes.Axes]
    _legend: matplotlib.legend.Legend

    _handles: List[matplotlib.artist.Artist]
    _texts: List[matplotlib.text.Text]
    _label_handler_map: Dict[str, Tuple[matplotlib.text.Text, matplotlib.artist.Artist]]

    def __init__(
            self,
            owner: Union[matplotlib.figure.Figure, matplotlib.axes.Axes],
            loc: str | None = None
    ):
        """
        Initialise a Legend with pickable entries for
        the given Figure / Axes instance.

        :param owner: figure / axes which owns the legend and data
        :param loc: placement of the legend; same as legend(loc)
        """
        # initializes data
        self._owner = owner
        self._legend = owner.legend(loc=loc)
        self._handles = self._legend.legend_handles
        self._texts = self._legend.get_texts()

        # init mapping
        self._label_handler_map = {}
        text: matplotlib.text.Text
        handler: matplotlib.artist.Artist
        for text, handler in zip(self._texts, self._handles):
            text.set_picker(PICKER_OFFSET)
            handler.set_picker(PICKER_OFFSET)
            self._label_handler_map[text.get_text()] = (text, handler)

    def on_legend_pick(self, event: PickEvent, plot_map: MetricPlotMap) -> bool:
        """
        Should be called within on_pick to react to the
        pick-events of legend entries.

        :param event: PickEvent which was passed to on_pick
        :param plot_map: mapping from MetricEnum to plotted data
        :return: if the ui changed and requires a redraw
        """
        event_target = event.artist
        label: MetricEnum

        if isinstance(event_target, matplotlib.text.Text):
            label = event_target.get_text()
        elif isinstance(event_target, matplotlib.artist.Artist):
            label = event_target.get_label()
        else:
            return False

        # Label does not exist for some reason
        if label not in plot_map:
            return False

        picked_artists = plot_map[label]
        visible = False
        artist: matplotlib.artist.Artist
        for artist in picked_artists:
            visible = visible or not artist.get_visible()
            artist.set_visible(visible)

        # Change the alpha on the entry in the legend, so we can see what lines
        # have been toggled.
        for entry in self._label_handler_map[label]:
            entry.set_alpha(1.0 if visible else 0.2)

        return True
