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
        actual: np.ndarray,
        predicted: np.ndarray,
        kernel_matrix: Optional[np.ndarray] = None,
        default_metrics_name: list[str] = DEFAULT_METRICS_NAME,
    ) -> None:
        self.actual: np.ndarray = actual
        self.predicted: np.ndarray = predicted
        self.kernel_matrix: Optional[np.ndarray] = kernel_matrix
        self.default_metrics_name: list[str] = default_metrics_name
        self.default_metrics: list[BaseMetric] = []
        self.custom_metrics: list[BaseMetric] = []

    def __repr__(self):
        return (
            f"Evaluation(actual={self.actual.tolist()}, "
            f"predicted={self.predicted.tolist()}, default_metrics_name={self.default_metrics_name})"
        )

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

    def to_dict(self) -> dict:
        """Convert evaluation metrics to dictionary.

        Raises:
            ValueError: if the metrics are not evaluated yet

        Returns:
            dict: dictionary of evaluation metrics
        """
        metrics = self.default_metrics + self.custom_metrics
        if len(metrics) == 0:
            raise ValueError("Metrics are not evaluated yet.")

        data = {metric.name: metric.score for metric in metrics}

        return data

    def to_dataframe(
        self,
        id: Optional[str] = None,
        id_columns_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Convert evaluation metrics to DataFrame.

        Args:
            id (Optional[str], optional): id of the evaluation (ex: run_id). Defaults to None.
            id_columns_name (Optional[str], optional): name of the id column. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame of evaluation metrics
        """
        data = self.to_dict()
        df = pd.DataFrame(data, index=[0])
        if (id is not None) and (id_columns_name is not None):
            df.insert(0, id_columns_name, id)

        return df
