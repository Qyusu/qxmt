from typing import Optional

import numpy as np
import pandas as pd

from qk_manager.constants import DEFAULT_METRICS_NAME
from qk_manager.evaluation.base_metric import BaseMetric
from qk_manager.evaluation.default_metrics import (
    Accuracy,
    F1Score,
    Precision,
    Recall,
    TargetAlignmet,
)

NAME2METRIC = {
    "accuracy": Accuracy,
    "precision": Precision,
    "recall": Recall,
    "f1_score": F1Score,
    "target_alignmet": TargetAlignmet,
}


class Evaluation:
    def __init__(
        self,
        run_id: int,
        actual: np.ndarray,
        predicted: np.ndarray,
        kernel_matrix: Optional[np.ndarray] = None,
        id_columns_name: str = "run_id",
        default_metrics_name: list[str] = DEFAULT_METRICS_NAME,
    ) -> None:
        self.run_id: int = run_id
        self.actual: np.ndarray = actual
        self.predicted: np.ndarray = predicted
        self.kernel_matrix: Optional[np.ndarray] = kernel_matrix
        self.id_columns_name: str = id_columns_name
        self.default_metrics_name: list[str] = default_metrics_name
        self.default_metrics: list[BaseMetric] = []
        self.custom_metrics: list[BaseMetric] = []

    def evaluate_default_metrics(self) -> None:
        """Evaluate default metrics.

        Raises:
            ValueError: if the metric is not implemented
        """
        for metric_name in self.default_metrics_name:
            if metric_name not in NAME2METRIC:
                raise ValueError(f"{metric_name} is not implemented.")

            metric = NAME2METRIC[metric_name]()
            metric.set_score(self.actual, self.predicted)
            self.default_metrics.append(metric)

    def evaluate_custom_metrics(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> None:
        """Evaluate default and custom metrics."""
        self.evaluate_default_metrics()
        # self.evaluate_custom_metrics()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert evaluation metrics to DataFrame.

        Returns:
            pd.DataFrame: DataFrame of evaluation metrics
        """
        metrics = self.default_metrics + self.custom_metrics
        if len(metrics) == 0:
            raise ValueError("Metrics are not evaluated yet.")

        data = {metric.name: metric.score for metric in metrics}
        df = pd.DataFrame(data, index=[0])
        df.insert(0, self.id_columns_name, self.run_id)

        return df
