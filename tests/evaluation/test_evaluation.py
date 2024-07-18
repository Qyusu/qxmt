import numpy as np
import pytest

from qk_manager import Evaluation


class TestEvaluation:
    @pytest.fixture(scope="function")
    def base_evaluation(self) -> Evaluation:
        actual = np.array([0, 1, 1, 0, 1])
        predicted = np.array([0, 1, 0, 1, 0])
        default_metrics_name = ["accuracy", "precision", "recall", "f1_score"]
        return Evaluation(
            actual=actual,
            predicted=predicted,
            kernel_matrix=None,
            default_metrics_name=default_metrics_name,
        )

    def test_evaluate_default_metrics(self, base_evaluation) -> None:
        base_evaluation.evaluate_default_metrics()
        acutal_scores = {"accuracy": 0.4, "precision": 0.5, "recall": 0.33, "f1_score": 0.4}
        assert len(base_evaluation.default_metrics) == 4
        for metric in base_evaluation.default_metrics:
            assert round(metric.score, 2) == acutal_scores[metric.name]

    def test_evaluate_custom_metrics(self, base_evaluation) -> None:
        with pytest.raises(NotImplementedError):
            base_evaluation.evaluate_custom_metrics()

    def test_evaluate(self, base_evaluation) -> None:
        base_evaluation.evaluate()
        acutal_scores = {"accuracy": 0.4, "precision": 0.5, "recall": 0.33, "f1_score": 0.4}
        assert len(base_evaluation.default_metrics) == 4
        for metric in base_evaluation.default_metrics:
            assert round(metric.score, 2) == acutal_scores[metric.name]

    def test_to_dict(self, base_evaluation) -> None:
        with pytest.raises(ValueError):
            base_evaluation.to_dict()

        base_evaluation.evaluate()
        result = base_evaluation.to_dict()
        actual_dict = {"accuracy": 0.4, "precision": 0.5, "recall": 0.33, "f1_score": 0.4}
        assert len(result) == 4
        for key, value in result.items():
            assert round(value, 2) == actual_dict[key]

    def test_to_dataframe(self, base_evaluation) -> None:
        base_evaluation.evaluate()
        df = base_evaluation.to_dataframe(id="0", id_columns_name="run_id")
        assert len(df) == 1
        assert len(df.columns) == 5
        assert df["run_id"].values[0] == "0"
        assert round(df["accuracy"].values[0], 2) == 0.4
        assert round(df["precision"].values[0], 2) == 0.5
        assert round(df["recall"].values[0], 2) == 0.33
        assert round(df["f1_score"].values[0], 2) == 0.4
