import numpy as np
import pytest

from qxmt.evaluation.defaults_regression import (
    MeanAbsoluteError,
    R2Score,
    RootMeanSquaredError,
)


class TestMeanAbsoluteError:
    @pytest.mark.parametrize(
        ["actual", "predicted", "expected"],
        [
            pytest.param(np.array([1.0, 1.5, 0.4]), np.array([0.8, 1.7, 0.2]), 0.2, id="case1"),
            pytest.param(np.array([0.0, 1.0, 0.4]), np.array([-0.8, 1.2, 0.2]), 0.4, id="case2"),
            pytest.param(np.array([5.0, 0.0, 2.0]), np.array([5.0, 0.0, 2.0]), 0.00, id="all_correct"),
        ],
    )
    def test_evaluate(self, actual: np.ndarray, predicted: np.ndarray, expected: float) -> None:
        mae = MeanAbsoluteError()
        score = mae.evaluate(actual, predicted)

        assert round(score, 2) == expected


class TestRootMeanSquaredError:
    @pytest.mark.parametrize(
        ["actual", "predicted", "expected"],
        [
            pytest.param(np.array([1.0, 1.5, 0.4]), np.array([0.8, 1.7, 0.2]), 0.2, id="case1"),
            pytest.param(np.array([0.0, 1.0, 0.4]), np.array([-0.8, 1.2, 0.2]), 0.49, id="case2"),
            pytest.param(np.array([5.0, 0.0, 2.0]), np.array([5.0, 0.0, 2.0]), 0.00, id="all_correct"),
        ],
    )
    def test_evaluate(self, actual: np.ndarray, predicted: np.ndarray, expected: float) -> None:
        rmse = RootMeanSquaredError()
        score = rmse.evaluate(actual, predicted)

        assert round(score, 2) == expected


class TestR2Score:
    @pytest.mark.parametrize(
        ["actual", "predicted", "expected"],
        [
            pytest.param(np.array([1.0, 1.5, 0.4]), np.array([0.8, 1.7, 0.2]), 0.8, id="case1"),
            pytest.param(np.array([0.0, 1.0, 0.4]), np.array([-0.8, 1.2, 0.2]), -0.42, id="case2"),
            pytest.param(np.array([5.0, 0.0, 2.0]), np.array([5.0, 0.0, 2.0]), 1.00, id="all_correct"),
        ],
    )
    def test_evaluate(self, actual: np.ndarray, predicted: np.ndarray, expected: float) -> None:
        r2 = R2Score()
        score = r2.evaluate(actual, predicted)

        assert round(score, 2) == expected
