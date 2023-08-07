"""
This file contains the implementation of the partial
dependency plot.
Call partial_dependency_plot() to create a new instance.

This chart is used to compare the impact of multiple
features on the generated prediction value. The impact
of each feature is plotted in relation to the other
feature values.
The feature value can be dynamically selected.
A prediction is also shown, based on the selected
values.
"""
from typing import Union, List, Final, Literal, Protocol, Any, Dict, Callable

import matplotlib.figure
import matplotlib.axes
import matplotlib.lines
import matplotlib.patches
import matplotlib.widgets
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt

from sklearn.base import is_regressor
from sklearn.inspection import partial_dependence

import numpy as np

from sklearn.utils.validation import check_is_fitted
from scikit_charts.metrics import METRIC_DTYPE
from scikit_charts.shared import DraggableAxvline, arg_nearest, DEFAULT_PLOT_COLOR, AxesButton

############## Constants ###################

DRAW_STEPS: Final[int] = 1000
LIMIT_COLOR: Final[str] = "#fae0d7"
TXT_ERROR_COLOR: Final[str] = "#fae0d7"
TXT_DEFAULT_COLOR: Final[str] = ".95"

PADDING: Final[float] = 0.04
UI_SIDEBAR_WIDTH: Final[float] = 0.2


#############################################

class SklearnPredictor(Protocol):
    """
    Class which defines the structure of a scikit-learn
    predictor. Used to improve typing within the
    library.
    """

    def fit(self, X, y, sample_weight=None, **fit_params):
        ...

    def predict(self, X: List[List]) -> Any:
        ...


class FeatureAxes:
    """
    A helper class which manages a single plotted feature
    for a partial dependency plot. Supports interactivity
    to select an x value to preview the calculated predictions.
    """
    # layout data
    _parent: matplotlib.figure.Figure
    _parent_spec: gridspec.SubplotSpec
    _internal_spec: gridspec.GridSpec
    _on_data_selected: Callable[[float, int], None]

    _main_axes: matplotlib.axes.Axes
    _histo_axes: matplotlib.axes.Axes
    _axvline: DraggableAxvline = None

    # data points
    _label: str
    _x_origin: np.ndarray
    _x_pdp: np.ndarray[float]
    _y_pdp: np.ndarray[float]

    _x_selected: float
    _i_selected: int

    # plotted values
    _ice_plot: List[matplotlib.lines.Line2D] = None
    _lower_limit_plot: matplotlib.patches.Polygon = None
    _upper_limit_plot: matplotlib.patches.Polygon = None

    def __init__(
            self,
            fig: matplotlib.figure.Figure,
            label: str,
            x_origin: np.ndarray,
            x_pdp: np.ndarray[float],
            y_pdp: np.ndarray[float],
            sub_spec: gridspec.SubplotSpec,
            on_data_selected: Callable[[float, int], None]
    ):
        """
        Initialise a new instance of FeatureAxes.
        This class manages a single plotted feature
        for a partial dependency plot.

        Also supports selecting an x value to display
        the interaction with other variables.

        :param fig: Figure which owns the axes
        :param label: label of the feature in the title
        :param x_origin: original x values of the feature
        for the histogram and limits
        :param x_pdp: calculated x values for the partial
        dependency on the plotted feature
        :param y_pdp: calculated y values for the partial
        dependency of the plotted feature
        :param sub_spec: a SubplotGridSpec which defines
        the position of the axes in the Figure
        :param on_data_selected: callback when the
        user selects an x value interactively
        """
        if len(x_pdp) != len(y_pdp):
            raise IndexError("Length of x_ice and y_ice data is not the same!")

        # initialise data
        self._parent = fig
        self._parent_spec = sub_spec
        self._label = label
        self._x_origin = x_origin
        self._y_pdp = y_pdp
        self._x_pdp = x_pdp
        self._on_data_selected = on_data_selected

        # initialise axes from gridspec
        self._internal_spec = self._parent_spec.subgridspec(10, 1, hspace=0)
        self._main_axes = self._parent.add_subplot(
            self._internal_spec[1:9, 0]
        )

        self._histo_axes = self._parent.add_subplot(
            self._internal_spec[0, 0],
            sharex=self._main_axes
        )

        # configure axes style
        if not self._parent_spec.is_first_col():
            self._main_axes.yaxis.set_visible(False)
        self._main_axes.margins(x=0.02)
        self._main_axes.set_xlim(np.min(self._x_pdp), np.max(self._x_pdp))

        self._histo_axes.set_axis_off()

        # plot data
        self._plot_data()

        self._histo_axes.hist(
            self._x_origin,
            bins=100,
            histtype="step",
        )

        # init axvline
        self.set_selected_value("median")
        self._axvline = DraggableAxvline(
            self._main_axes,
            self._x_selected,
            snapping=self._x_pdp,
            on_drag_finish=lambda val, i: self._on_axvline_changed(val, i),
            pick_workaround=True,
            color="red"
        )

    def _on_axvline_changed(self, new_val: float, i: int):
        self._x_selected = new_val
        self._i_selected = i
        self._update_title()

        if self._on_data_selected is not None:
            self._on_data_selected(new_val, i)

    def _update_title(self):
        self._histo_axes.set_title(f"{self._label}: {self._x_selected:.2f}")

    def _plot_data(self):
        # clear old data
        if self._ice_plot is not None:
            for item in self._ice_plot:
                item.remove()
            self._lower_limit_plot.remove()
            self._upper_limit_plot.remove()

        self._ice_plot = self._main_axes.plot(
            self._x_pdp,
            self._y_pdp,
            color=DEFAULT_PLOT_COLOR
        )

        # draw lower limit / upper limit
        x_min = self._x_origin.min()
        x_max = self._x_origin.max()
        xlim = self._main_axes.get_xbound()

        self._lower_limit_plot = self._main_axes.axvspan(
            xmin=xlim[0],
            xmax=x_min,
            color=LIMIT_COLOR
        )
        self._upper_limit_plot = self._main_axes.axvspan(
            xmin=x_max,
            xmax=xlim[1],
            color=LIMIT_COLOR
        )

    def reset(self):
        """
        Reset the plot and selection of the FeatureAxes
        to the initial value.
        """
        self._main_axes.set_xlim(np.min(self._x_pdp), np.max(self._x_pdp))

        self.set_selected_value("median")
        self._plot_data()

    def set_selected_value(
            self,
            selector: Literal["median", "average"] | int | float
    ):
        """
        Update the currently selected value of the FeatureAxes.
        Possible values are the median, average-value, a index
        value or a custom float.

        The correlation value which is closest to the selection
        will be chosen and selected.
        """
        value: float

        if selector == "median":
            i = int(len(self._x_origin) / 2)
            value = self._x_origin[i]
        elif selector == "average":
            value = np.average(self._x_origin)
        elif isinstance(selector, int):
            value = self._x_origin[selector]
        elif isinstance(selector, float):
            value = selector
        else:
            raise TypeError("Type of value does not match 'Literal[\"median\", \"average\"] | int'")

        # snap to nearest possible value
        self._i_selected = arg_nearest(self._x_pdp, value)
        self._x_selected = self._x_pdp[self._i_selected]

        self._update_title()
        if self._axvline is not None:
            self._axvline.set_current_x(self._x_selected)

    def set_prediction_values(self, y_pdp: np.ndarray[float]):
        """
        Set the prediction values for the feature
        of this FeatureAxes. Usually called if another
        feature index changes.
        :param y_pdp: new list of y values for the feature
        """
        self._y_pdp = y_pdp
        self._plot_data()

    def get_selected_value(self):
        """
        Return the currently selected feature
        value of the FeatureAxes.
        """
        return self._x_selected

    def get_selected_pdp_index(self):
        """
        Returns the index of the selected data,
        from the initial pdp array of x values.
        """
        return self._i_selected

    def get_main_axes(self) -> matplotlib.axes.Axes:
        """
        Return the main axes of the FeatureAxes.
        The main axes contains the plot of the
        feature correlation.
        """
        return self._main_axes


class PartialDependencyPlot:
    """
    Class which implements the partial dependency plot. It should not
    be used directly. Instantiation should be done by calling
    partial_dependency_plot().
    """

    # base chart data
    _fig: matplotlib.figure.Figure
    _grid_spec: gridspec.GridSpec
    _predictor: SklearnPredictor

    # ui element
    _reset_btn: AxesButton

    _median_btn: AxesButton
    _average_btn: AxesButton

    _limit_btn: AxesButton
    _y_lower_txt: matplotlib.widgets.TextBox
    _y_upper_txt: matplotlib.widgets.TextBox

    # data points
    _x: np.ndarray[float]
    _y: np.ndarray[float]
    _features: List[int]

    _pdp_dict: Dict[int, np.ndarray[float]]
    _pdp_prediction: np.ndarray[float]
    _pdp_density: int

    # axes classes
    _feature_axes_dict: Dict[int, FeatureAxes]
    _first_column: List[FeatureAxes]

    def __init__(
            self,
            x: np.ndarray[float],
            y: np.ndarray[float],
            features: List[int],
            predictor: SklearnPredictor,
            columns: int,
            density: int,
            hide_options: bool
    ):
        if columns < 1:
            raise ValueError("The value of columns has to be 1 or more!")

        self._x = x
        self._y = y
        self._predictor = predictor
        self._features = features
        self._pdp_density = density

        self._fig = plt.figure()
        self._fig.suptitle("partial dependency plot")
        if not hide_options:
            self._fig.subplots_adjust(left=UI_SIDEBAR_WIDTH + PADDING * 2 + 0.04)

        # calculate individual conditional expectation
        pdp = partial_dependence(
            self._predictor,
            self._x,
            # Generate for all features
            self._features,
            kind="average",
            grid_resolution=self._pdp_density
        )
        feature_values = pdp["grid_values"]
        self._pdp_prediction = pdp["average"][0]

        # initialise pdp dictionary and initial x values
        self._pdp_dict = {}
        for i_feature, i_value in zip(self._features, range(len(feature_values))):
            self._pdp_dict[i_feature] = feature_values[i_value]

        # initialise axes
        rows = int(1 + len(self._features) / columns)
        self._grid_spec = gridspec.GridSpec(
            rows, columns, hspace=0.05, figure=self._fig
        )
        self._init_axes(rows, columns)
        self._update_row_labels()

        if not hide_options:
            self._init_interactive()

    def _init_axes(self, rows: int, columns: int):
        # generate initial axes values
        i_median = int(self._pdp_density / 2)
        initial_indices: List[int] = []

        for feature in self._features:
            initial_indices.append(
                np.argsort(self._pdp_dict[feature])[i_median]
            )

        y_min = np.min(self._pdp_prediction)
        y_max = np.max(self._pdp_prediction)

        # initialise sub axes
        self._feature_axes_dict = {}
        self._first_column = []
        feature_index = 0

        for row in range(rows):
            for col in range(columns):
                if feature_index >= len(self._features):
                    break

                feature = self._features[feature_index]
                pdp_predict = self._pdp_predict_for_feature(
                    self._pdp_prediction,
                    initial_indices,
                    feature_index
                )

                axes = FeatureAxes(
                    self._fig,
                    f"x{feature}",
                    self._x[feature_index],
                    self._pdp_dict[feature],
                    pdp_predict,
                    self._grid_spec[row, col],
                    lambda val, i: self._feature_index_changed(i)
                )
                axes.get_main_axes().set_ylim(y_min, y_max)

                self._feature_axes_dict[feature_index] = axes

                if col == 0:
                    self._first_column.append(axes)

                feature_index += 1

    def _init_interactive(self):
        # btn
        self._reset_btn = AxesButton(
            self._fig,
            "Reset",
            lambda e: self.reset(),
            (PADDING, 0.85, UI_SIDEBAR_WIDTH, 0.1),
        )

        self._median_btn = AxesButton(
            self._fig,
            "Select Median",
            lambda e: self._set_axes_values("median"),
            (PADDING, 0.7, UI_SIDEBAR_WIDTH, 0.1),
        )
        self._average_btn = AxesButton(
            self._fig,
            "Select Average",
            lambda e: self._set_axes_values("average"),
            (PADDING, 0.6, UI_SIDEBAR_WIDTH, 0.1),
        )

        # limit selection
        y_min = np.min(self._pdp_prediction)
        y_max = np.max(self._pdp_prediction)

        self._limit_btn = AxesButton(
            self._fig,
            "Set y limits",
            lambda e: self._on_y_limit_changed(),
            (PADDING, 0.45, UI_SIDEBAR_WIDTH, 0.1)
        )
        self._y_lower_txt = matplotlib.widgets.TextBox(
            self._fig.add_axes((PADDING + 0.05, 0.35, UI_SIDEBAR_WIDTH - 0.05, 0.1)),
            "Lower:",
            initial=f"{y_min:.2f}"
        )
        self._y_upper_txt = matplotlib.widgets.TextBox(
            self._fig.add_axes((PADDING + 0.05, 0.25, UI_SIDEBAR_WIDTH - 0.05, 0.1)),
            "Upper:",
            initial=f"{y_max:.2f}"
        )

    def _on_y_limit_changed(self):
        # parse and validate data
        y_min: float | None = None
        y_max: float | None = None

        min_str = self._y_lower_txt.text
        max_str = self._y_upper_txt.text

        try:
            y_min = float(min_str)
            self._y_lower_txt.color = TXT_DEFAULT_COLOR
        except ValueError:
            y_min = None
            self._y_lower_txt.color = TXT_ERROR_COLOR

        try:
            y_max = float(max_str)
            self._y_upper_txt.color = TXT_DEFAULT_COLOR
        except ValueError:
            y_max = None
            self._y_upper_txt.color = TXT_ERROR_COLOR

        if y_min is None or y_max is None:
            return

        # set limits
        for axes in self._feature_axes_dict.values():
            axes.get_main_axes().set_ylim(y_min, y_max)

    def _set_axes_values(self, selector: Literal["median", "average"] | int | float):
        for axes in self._feature_axes_dict.values():
            axes.set_selected_value(selector)

        self._feature_index_changed(-1)
        self._update_row_labels()

    @staticmethod
    def _pdp_predict_for_feature(
            pdp: np.ndarray[float],
            indices: List[int],
            feature_index: int
    ) -> np.ndarray[float]:
        """
        Select the predicted result, based
        on the selected Feature indices.
        :param pdp: result generated by partial_dependence
        :param indices: selected indices for each feature variable
        :param feature_index: current index, for which the prediction
        values are selected
        :return: list of prediction values, for the values of the
        feature
        """
        left = indices[:feature_index]
        right = []
        if feature_index + 1 < len(indices):
            right = indices[feature_index + 1:]

        return pdp[*left, :, *right]

    def _feature_index_changed(self, i_changed: int):
        # collect selected indices
        selected_indices: List[int] = []
        axes: FeatureAxes
        for axes in self._feature_axes_dict.values():
            selected_indices.append(axes.get_selected_pdp_index())

        # update y values
        for i_axes, axes in self._feature_axes_dict.items():
            if i_axes == i_changed:
                continue

            axes.set_prediction_values(
                self._pdp_predict_for_feature(
                    self._pdp_prediction,
                    selected_indices,
                    i_axes
                )
            )

        self._update_row_labels()

    def _update_row_labels(self):
        axes: FeatureAxes

        # collect data
        prediction: float | np.ndarray[float] = self._pdp_prediction
        for axes in self._feature_axes_dict.values():
            prediction = prediction[axes.get_selected_pdp_index()]

        for axes in self._first_column:
            axes.get_main_axes().set_ylabel(f"prediction: {prediction:.2f}")

    def reset(self):
        """
        Resets the chart data and options to the initial values.
        """
        y_min = np.min(self._pdp_prediction)
        y_max = np.max(self._pdp_prediction)

        # reset plots
        axes: FeatureAxes
        for axes in self._feature_axes_dict.values():
            axes.reset()
            axes.get_main_axes().set_ylim(y_min, y_max)

        self._feature_index_changed(-1)
        self._update_row_labels()

        # reset interaction
        self._y_lower_txt.set_val(f"{y_min:.2f}")
        self._y_upper_txt.set_val(f"{y_max:.2f}")

    def get_figure(self) -> matplotlib.figure.Figure:
        """
        :return: matplot figure instance of the chart
        """
        return self._fig


def partial_dependency_plot(
        x: Union[np.ndarray[float], list[list[float]]],
        y: Union[np.ndarray[float], list[float]],
        predictor: SklearnPredictor,
        features: List[int],
        columns: int = 3,
        density: int = 50,
        hide_options = False
) -> matplotlib.figure.Figure:
    """
    Instantiate a new partial dependency plot instance from the
    given features and target values.

    This chart is used to compare the impact of multiple
    features on the generated prediction value. The impact
    of each feature is plotted in relation to the other
    feature values.
    The feature value can be dynamically selected and impact
    the other plots and the generated prediction.

    Use .show() on the returned element to display the chart.

    :param x: 2D-array of feature values, which was used to train the prediction model
    :param y: 1D-array of target values with the same length as x
    :param predictor: the trained model which implements the functions defined
    in SklearnPredictor
    :param features: indices of x columns, which describes the features that
    should be plotted
    :param columns: amount of columns for the grid of feature Axes
    :param density: amount of values generated for each feature to
    plot the partial dependency. NOTICE: a high value can cause heavy
    performance impacts
    :param hide_options: if the configuration options should be
    hidden to gain more space
    :return: matplotlib.figure.Figure instance used to display the chart
    :raises TypeError: raised if the predictor is not a regressor, based on
    sklearn.base.is_regressor
    :raises NotFittedError: raised if fit was not called on the predictor
    :raises ValueError: raised if x or y have the wrong shape or are empty
    """

    # validate predictor
    if not is_regressor(predictor):
        raise TypeError("is_regressor(predictor) has to return true!")
    check_is_fitted(predictor)

    # Force copy to ensure datatype is correct
    x = np.array(x, dtype=METRIC_DTYPE, copy=False)
    y = np.array(y, dtype=METRIC_DTYPE, copy=False)

    # validate arguments
    if x.ndim != 2:
        raise ValueError(f"x-array has the wrong shape. Expected (n, n) got {x.shape}!")
    if len(x) == 0:
        raise ValueError("x-array is empty!")
    if y.ndim != 1:
        raise ValueError(f"y-array has the wrong shape. Expected (n) got {y.shape}")
    if len(y) == 0:
        raise ValueError("y-array is empty!")
    if len(y) != len(x):
        raise ValueError(f"Different array length between x and y. x: {len(x)}, y: {len(y)}")

    plot = PartialDependencyPlot(x, y, features, predictor, columns, density, hide_options)

    return plot.get_figure()
