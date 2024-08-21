import numpy as np
import pytest

from qxmt.evaluation import Accuracy, F1Score, Precision, Recall


class TestAccuracy:
    @pytest.mark.parametrize(
        ["actual", "predicted", "expected"],
        [
            pytest.param(np.array([0, 1, 1, 0, 1]), np.array([0, 1, 0, 0, 1]), 0.80, id="case1"),
            pytest.param(np.array([0, 1, 1, 0, 1]), np.array([0, 1, 1, 0, 1]), 1.00, id="all_correct"),
            pytest.param(np.array([0, 1, 1, 0, 1]), np.array([1, 0, 0, 1, 0]), 0.00, id="all_wrong"),
        ],
    )
    def test_evaluate(self, actual: np.ndarray, predicted: np.ndarray, expected: float) -> None:
        accuracy = Accuracy()
        score = accuracy.evaluate(actual, predicted)

        assert score == expected


class TestPrecision:
    @pytest.mark.parametrize(
        ["actual", "predicted", "expected"],
        [
            pytest.param(np.array([0, 1, 1, 0, 1]), np.array([1, 1, 0, 1, 1]), 0.50, id="case1"),
            pytest.param(np.array([0, 1, 1, 0, 1]), np.array([0, 1, 1, 0, 0]), 1.00, id="all_correct"),
            pytest.param(np.array([0, 1, 1, 0, 1]), np.array([1, 0, 0, 1, 0]), 0.00, id="all_wrong"),
        ],
    )
    def test_evaluate(self, actual: np.ndarray, predicted: np.ndarray, expected: float) -> None:
        precision = Precision()
        score = precision.evaluate(actual, predicted)

        assert round(score, 2) == expected


class TestRecall:
    @pytest.mark.parametrize(
        ["actual", "predicted", "expected"],
        [
            pytest.param(np.array([0, 1, 1, 0, 1]), np.array([1, 1, 0, 1, 0]), 0.33, id="case1"),
            pytest.param(np.array([0, 1, 1, 0, 1]), np.array([0, 1, 1, 1, 1]), 1.00, id="all_correct"),
            pytest.param(np.array([0, 1, 1, 0, 1]), np.array([1, 0, 0, 1, 0]), 0.00, id="all_wrong"),
        ],
    )
    def test_evaluate(self, actual: np.ndarray, predicted: np.ndarray, expected: float) -> None:
        recall = Recall()
        score = recall.evaluate(actual, predicted)

        assert round(score, 2) == expected


class TestF1Score:
    @pytest.mark.parametrize(
        ["actual", "predicted", "expected"],
        [
            pytest.param(np.array([0, 1, 1, 0, 1]), np.array([1, 1, 1, 1, 0]), 0.57, id="case1"),
            pytest.param(np.array([0, 1, 1, 0, 1]), np.array([0, 1, 1, 0, 1]), 1.00, id="all_correct"),
            pytest.param(np.array([0, 1, 1, 0, 1]), np.array([1, 0, 0, 1, 0]), 0.00, id="all_wrong"),
        ],
    )
    def test_evaluate(self, actual: np.ndarray, predicted: np.ndarray, expected: float) -> None:
        f1_score = F1Score()
        score = f1_score.evaluate(actual, predicted)

        assert round(score, 2) == expected
