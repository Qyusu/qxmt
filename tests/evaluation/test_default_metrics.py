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
            pytest.param(np.array([0, 8, 8, 0, 8]), np.array([0, 8, 0, 0, 8]), 0.80, id="pos_is_not_1"),
            pytest.param(np.array([0, 1, 2, 2, 1]), np.array([0, 1, 2, 2, 0]), 0.80, id="multi_class_case1"),
            pytest.param(np.array([0, 1, 2, 2, 1]), np.array([0, 1, 2, 2, 1]), 1.00, id="multi_class_all_correct"),
            pytest.param(np.array([0, 1, 2, 2, 1]), np.array([1, 0, 0, 1, 0]), 0.00, id="multi_class_all_wrong"),
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
            pytest.param(np.array([0, 5, 5, 0, 5]), np.array([5, 5, 0, 5, 5]), 0.50, id="pos_is_not_1"),
            pytest.param(np.array([0, 1, 2, 2, 1]), np.array([0, 1, 2, 2, 0]), 0.67, id="multi_class_case1"),
            pytest.param(np.array([0, 1, 2, 2, 1]), np.array([0, 1, 2, 2, 1]), 1.00, id="multi_class_all_correct"),
            pytest.param(np.array([0, 1, 2, 2, 1]), np.array([1, 0, 0, 1, 0]), 0.00, id="multi_class_all_wrong"),
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
            pytest.param(np.array([2, 5, 5, 2, 5]), np.array([5, 5, 2, 5, 2]), 0.33, id="pos_is_not_1"),
            pytest.param(np.array([0, 1, 2, 2, 1]), np.array([0, 1, 2, 2, 0]), 0.83, id="multi_class_case1"),
            pytest.param(np.array([0, 1, 2, 2, 1]), np.array([0, 1, 2, 2, 1]), 1.00, id="multi_class_all_correct"),
            pytest.param(np.array([0, 1, 2, 2, 1]), np.array([1, 0, 0, 1, 0]), 0.00, id="multi_class_all_wrong"),
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
            pytest.param(np.array([0, 3, 3, 0, 3]), np.array([3, 3, 3, 3, 0]), 0.57, id="pos_is_not_1"),
            pytest.param(np.array([0, 1, 2, 2, 1]), np.array([0, 1, 2, 2, 0]), 0.78, id="multi_class_case1"),
            pytest.param(np.array([0, 1, 2, 2, 1]), np.array([0, 1, 2, 2, 1]), 1.00, id="multi_class_all_correct"),
            pytest.param(np.array([0, 1, 2, 2, 1]), np.array([1, 0, 0, 1, 0]), 0.00, id="multi_class_all_wrong"),
        ],
    )
    def test_evaluate(self, actual: np.ndarray, predicted: np.ndarray, expected: float) -> None:
        f1_score = F1Score()
        score = f1_score.evaluate(actual, predicted)

        assert round(score, 2) == expected
