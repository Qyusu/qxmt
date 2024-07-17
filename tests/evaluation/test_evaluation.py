import numpy as np
import pytest

from qk_manager import Evaluation


class TestEvaluation:
    @pytest.fixture(scope="function")
    def base_evaluation(self) -> Evaluation:
        run_id = 0
        actual = np.array([0, 1, 1, 0, 1])
        predicted = np.array([0, 1, 0, 1, 0])
        default_metrics_name = ["accuracy", "precision", "recall", "f1_score"]
        return Evaluation(
            run_id=run_id,
            actual=actual,
            predicted=predicted,
            kernel_matrix=None,
            id_columns_name="run_id",
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

    def test_plot(self, base_evaluation) -> None:
        with pytest.raises(NotImplementedError):
            base_evaluation.plot()

    def test_to_dataframe(self, base_evaluation) -> None:
        with pytest.raises(ValueError):
            base_evaluation.to_dataframe()

        base_evaluation.evaluate()
        df = base_evaluation.to_dataframe()
        assert len(df) == 1
        assert len(df.columns) == 5
        assert round(df["run_id"].values[0], 2) == 0
        assert round(df["accuracy"].values[0], 2) == 0.4
        assert round(df["precision"].values[0], 2) == 0.5
        assert round(df["recall"].values[0], 2) == 0.33
        assert round(df["f1_score"].values[0], 2) == 0.4
