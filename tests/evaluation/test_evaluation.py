from typing import get_args

import numpy as np
import pytest

from qxmt.evaluation import Evaluation
from qxmt.evaluation.defaults import DEFAULT_METRICS_NAME, BaseMetric

DEFAULT_METRICS_NUM = len(get_args(DEFAULT_METRICS_NAME))


class CustomMetric(BaseMetric):
    def __init__(self, name: str = "custom") -> None:
        super().__init__(name)

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray) -> float:
        score = actual[0] + predicted[0]

        return float(score)


class ErrorCustomMetric:
    # not inherit BaseMetric
    def __init__(self, name: str = "error_custom") -> None:
        self.name = name

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray) -> float:
        score = actual[0] + predicted[0]

        return float(score)


class TestEvaluation:
    @pytest.fixture(scope="function")
    def base_evaluation(self) -> Evaluation:
        actual = np.array([0, 1, 1, 0, 1])
        predicted = np.array([0, 1, 0, 1, 0])
        default_metrics_name = ["accuracy", "precision", "recall", "f1_score"]
        custom_metrics = None
        return Evaluation(
            actual=actual,
            predicted=predicted,
            default_metrics_name=default_metrics_name,
            custom_metrics=custom_metrics,
        )

    @pytest.fixture(scope="function")
    def custom_evaluation(self) -> Evaluation:
        actual = np.array([0, 1, 1, 0, 1])
        predicted = np.array([0, 1, 0, 1, 0])
        default_metrics_name = None
        return Evaluation(
            actual=actual,
            predicted=predicted,
            default_metrics_name=default_metrics_name,
            custom_metrics=[{"module_name": __name__, "implement_name": "CustomMetric", "params": {}}],
        )

    def test_init_default_metrics(self, base_evaluation: Evaluation) -> None:
        assert len(base_evaluation.default_metrics) == 4
        for metric in base_evaluation.default_metrics:
            assert metric.score is None

        # default_metrics_name is None
        default_name_none_evaluation = Evaluation(
            actual=np.array([0, 1]), predicted=np.array([0, 1]), default_metrics_name=None
        )
        assert len(default_name_none_evaluation.default_metrics) == DEFAULT_METRICS_NUM

        # value error if metric is not implemented
        with pytest.raises(ValueError):
            Evaluation(actual=np.array([0, 1]), predicted=np.array([0, 1]), default_metrics_name=["not_implemented"])

            Evaluation(actual=np.array([0, 1]), predicted=np.array([0, 1]), default_metrics_name=["not_implemented"])

    def test_init_custom_metrics(self, base_evaluation: Evaluation, custom_evaluation: Evaluation) -> None:
        # custom_metrics is None
        assert base_evaluation.custom_metrics == []

        # custom_metrics is not None
        assert len(custom_evaluation.custom_metrics) == 1
        assert custom_evaluation.custom_metrics[0].score is None
        assert custom_evaluation.custom_metrics_name == ["custom"]

        # value error if custom metrics is not BaseMetric instance
        with pytest.raises(ValueError):
            Evaluation(
                actual=np.array([0, 1]),
                predicted=np.array([0, 1]),
                custom_metrics=[{"module_name": __name__, "implement_name": "ErrorCustomMetric", "params": {}}],
            )

    def test_set_evaluation_result(self, base_evaluation: Evaluation) -> None:
        base_evaluation.set_evaluation_result(base_evaluation.default_metrics)
        acutal_scores = {"accuracy": 0.4, "precision": 0.5, "recall": 0.33, "f1_score": 0.4}
        assert len(base_evaluation.default_metrics) == 4
        for metric in base_evaluation.default_metrics:
            assert metric.score is not None
            assert round(metric.score, 2) == acutal_scores[metric.name]

    def test_evaluate(self, custom_evaluation: Evaluation) -> None:
        custom_evaluation.evaluate()
        acutal_scores = {"accuracy": 0.4, "precision": 0.5, "recall": 0.33, "f1_score": 0.4, "custom": 0.0}
        metrics_list = custom_evaluation.default_metrics + custom_evaluation.custom_metrics
        assert len(metrics_list) == DEFAULT_METRICS_NUM + 1
        for metric in metrics_list:
            assert metric.score is not None
            assert round(metric.score, 2) == acutal_scores[metric.name]

    def test_to_dict(self, base_evaluation: Evaluation, custom_evaluation: Evaluation) -> None:
        # only default metrics
        with pytest.raises(ValueError):
            base_evaluation.to_dict()

        base_evaluation.evaluate()
        result = base_evaluation.to_dict()
        actual_dict = {"accuracy": 0.4, "precision": 0.5, "recall": 0.33, "f1_score": 0.4}
        assert len(result) == 4
        for key, value in result.items():
            assert round(value, 2) == actual_dict[key]

        # default and custom metrics
        with pytest.raises(ValueError):
            custom_evaluation.to_dict()

        custom_evaluation.evaluate()
        result = custom_evaluation.to_dict()
        actual_dict = {"accuracy": 0.4, "precision": 0.5, "recall": 0.33, "f1_score": 0.4, "custom": 0.0}
        assert len(result) == 5
        for key, value in result.items():
            assert round(value, 2) == actual_dict[key]

    def test_to_dataframe(self, base_evaluation: Evaluation, custom_evaluation: Evaluation) -> None:
        base_evaluation.evaluate()
        df = base_evaluation.to_dataframe(id="0", id_columns_name="run_id")
        assert len(df) == 1
        assert len(df.columns) == 5
        assert df["run_id"].values[0] == "0"
        assert round(df["accuracy"].values[0], 2) == 0.4
        assert round(df["precision"].values[0], 2) == 0.5
        assert round(df["recall"].values[0], 2) == 0.33
        assert round(df["f1_score"].values[0], 2) == 0.4

        # default and custom metrics
        custom_evaluation.evaluate()
        df = custom_evaluation.to_dataframe(id="0", id_columns_name="run_id")
        assert len(df) == 1
        assert len(df.columns) == 6
        assert df["run_id"].values[0] == "0"
        assert round(df["accuracy"].values[0], 2) == 0.4
        assert round(df["precision"].values[0], 2) == 0.5
        assert round(df["recall"].values[0], 2) == 0.33
        assert round(df["f1_score"].values[0], 2) == 0.4
        assert round(df["custom"].values[0], 2) == 0.0
