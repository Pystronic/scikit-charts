import pytest
import numpy as np
from pandas import DataFrame

from scikit_charts.metrics import create_metrics, PredictionException, METRIC_DTYPE, MetricEnum

#                         x,                                  y
create_whenWrongShape = [([[[1], [2], [3]], [[1], [2], [3]]], [1, 2, 3, 4]),
                         ([[1, 2, 3], [1, 2, 3]], [[1], [2], [3]])]
create_whenWrongShape_label = [f"x={i[0]}, y={i[1]}" for i in create_whenWrongShape]

#                         x, y
create_whenEmptyArrays = [([], [1, 2, 3, 4]),
                          ([[1, 2, 3], [1, 2, 3]], [])]
create_whenEmptyArrays_label = [f"x={i[0]}, y={i[1]}" for i in create_whenEmptyArrays]


@pytest.mark.parametrize("x,y", create_whenWrongShape, ids=create_whenWrongShape_label)
def test_create_metrics_whenArraysWrongShape_raiseValueError(x: np.ndarray, y: np.ndarray):
    with pytest.raises(ValueError):
        create_metrics(x, y, lambda x: 1.0)


@pytest.mark.parametrize("x,y", create_whenEmptyArrays, ids=create_whenEmptyArrays_label)
def test_create_metrics_whenArraysEmpty_raiseValueError(x: np.ndarray, y: np.ndarray):
    with pytest.raises(ValueError):
        create_metrics(x, y, lambda x: 1.0)


def test_create_metrics_whenArraysLengthMismatch_raiseValueError():
    x = [[1, 2, 3], [1, 2, 3]]
    y = [1]
    with pytest.raises(ValueError):
        create_metrics(x, y, lambda x: 1.0)


def predict_raise_error() -> float:
    raise RuntimeError("Prediction error")


def test_create_metrics_whenPredictRaisesError_raisePredictionException():
    x = [[1, 2, 3], [3, 4, 5]]
    y = [3, 6]
    with pytest.raises(PredictionException):
        create_metrics(x, y, predict_raise_error)


def predict_average_rounded(x: tuple[float]) -> float:
    return round(np.average(x))


def test_create_metrics_whenSingleFeature_returnCorrectDataframe():
    x = [[2], [4]]
    y = [3, 5]
    expected_frame = DataFrame(
        data=list(zip([2, 4], y, [2, 4], [1, 1], [1/y[0], 1/y[1]])),
        columns=["x0", MetricEnum.TARGET, MetricEnum.PREDICTION, MetricEnum.RESIDUAL, MetricEnum.RELATIVE_ERROR],
        dtype=METRIC_DTYPE
    )

    actual_frame = create_metrics(x, y, predict_average_rounded)

    assert expected_frame.shape == actual_frame.shape
    assert np.array_equal(expected_frame.index, actual_frame.index)
    for column in expected_frame.columns:
        assert np.array_equal(expected_frame[column], actual_frame[column])


def test_create_metrics_whenMultipleFeatures_returnCorrectDataframe():
    x = [[1, 2, 3], [3, 4, 5]]
    y = [3, 5]
    expected_frame = DataFrame(
        data=list(zip([1, 3], [2, 4], [3, 5], y, [2, 4], [1, 1], [1/y[0], 1/y[1]])),
        columns=[
            "x0", "x1", "x2",
            MetricEnum.TARGET, MetricEnum.PREDICTION, MetricEnum.RESIDUAL, MetricEnum.RELATIVE_ERROR
        ],
        dtype=METRIC_DTYPE
    )

    actual_frame = create_metrics(x, y, predict_average_rounded)

    assert expected_frame.shape == actual_frame.shape
    assert np.array_equal(expected_frame.columns, actual_frame.columns)
    for column in expected_frame.columns:
        assert np.array_equal(expected_frame[column], actual_frame[column])


def test_get_feature_columns_whenMultipleFeatures_returnInOrder():
    x = [[1, 2, 3], [3, 4, 5]]
    y = [3, 5]
    expected = ["x0", "x1", "x2"]

    frame = create_metrics(x, y, predict_average_rounded)
    actual = MetricEnum.get_feature_columns(frame)

    assert np.array_equal(expected, actual)

