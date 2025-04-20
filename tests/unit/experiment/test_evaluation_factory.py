import numpy as np
import pytest

from qxmt.evaluation.metrics.base import BaseMetric
from qxmt.experiment.evaluation_factory import EvaluationFactory


class CustomMetric(BaseMetric):
    def __init__(self, name: str = "custom") -> None:
        super().__init__(name)

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray) -> float:
        score = actual[0] + predicted[0]

        return float(score)


class TestEvaluationFactory:
    def test_evaluate_qkernel_classification(self) -> None:
        params = {"actual": [0, 1, 0, 1], "predicted": [0, 1, 1, 1]}
        result = EvaluationFactory.evaluate(
            model_type="qkernel", task_type="classification", params=params, default_metrics_name=["accuracy"]
        )

        assert "accuracy" in result
        assert isinstance(result["accuracy"], float)
        assert 0 <= result["accuracy"] <= 1

    def test_evaluate_qkernel_regression(self) -> None:
        params = {"actual": [1.0, 2.0, 3.0], "predicted": [1.1, 2.1, 2.9]}
        result = EvaluationFactory.evaluate(
            model_type="qkernel", task_type="regression", params=params, default_metrics_name=["mean_absolute_error"]
        )

        assert "mean_absolute_error" in result
        assert isinstance(result["mean_absolute_error"], float)
        assert result["mean_absolute_error"] >= 0

    def test_evaluate_vqe(self) -> None:
        cost_history = [10.0, 8.0, 9.5, 4.0, -1.5]
        params = {"cost_history": cost_history}
        result = EvaluationFactory.evaluate(
            model_type="vqe", task_type=None, params=params, default_metrics_name=["final_cost"]
        )

        assert "final_cost" in result
        assert isinstance(result["final_cost"], float)
        assert result["final_cost"] == -1.5

    def test_evaluate_invalid_model_type(self) -> None:
        with pytest.raises(ValueError):
            EvaluationFactory.evaluate(
                model_type="invalid_model", task_type="classification", params={"actual": [0], "predicted": [0]}
            )

    def test_evaluate_invalid_task_type(self) -> None:
        with pytest.raises(ValueError):
            EvaluationFactory.evaluate(
                model_type="qkernel", task_type="invalid_task", params={"actual": [0], "predicted": [0]}
            )

    def test_evaluate_with_custom_metrics(self) -> None:
        # [TODO]: custom metric receive from BaseMetric class not from config file
        pass
