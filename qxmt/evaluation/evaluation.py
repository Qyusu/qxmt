from typing import Optional

import numpy as np
import pandas as pd

from qxmt.constants import DEFAULT_METRICS_NAME
from qxmt.evaluation.defaults import Accuracy, BaseMetric, F1Score, Precision, Recall

NAME2METRIC = {
    "accuracy": Accuracy,
    "precision": Precision,
    "recall": Recall,
    "f1_score": F1Score,
}


class Evaluation:
    def __init__(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        default_metrics_name: Optional[list[str]] = None,
        custom_metrics: Optional[list[BaseMetric]] = None,
    ) -> None:
        self.actual: np.ndarray = actual
        self.predicted: np.ndarray = predicted
        self.default_metrics_name: list[str]
        self.custom_metrics_name: list[str]
        self.default_metrics: list[BaseMetric]
        self.custom_metrics: list[BaseMetric]

        self.init_default_metrics(default_metrics_name)
        self.init_custom_metrics(custom_metrics)

    def __repr__(self) -> str:
        return (
            f"Evaluation(actual={self.actual.tolist()}, "
            f"predicted={self.predicted.tolist()}, "
            f"default_metrics_name={self.default_metrics_name})"
            f"custom_metrics_name={self.custom_metrics_name}"
        )

    def init_default_metrics(self, default_metrics_name: Optional[list[str]]) -> None:
        """Initialize and validate default metrics.

        Raises:
            ValueError: if the metric is not implemented
        """
        if default_metrics_name is not None:
            self.default_metrics_name = default_metrics_name
        else:
            self.default_metrics_name = DEFAULT_METRICS_NAME

        self.default_metrics = []
        for metric_name in self.default_metrics_name:
            if metric_name not in NAME2METRIC:
                raise ValueError(f"{metric_name} is not implemented.")

            metric = NAME2METRIC[metric_name]()
            self.default_metrics.append(metric)

    def init_custom_metrics(self, custom_metrics: Optional[list[BaseMetric]]) -> None:
        """Initialize and validate custom metrics.

        Args:
            custom_metrics (Optional[list[BaseMetric]], optional): list of custom metrics. Defaults to None.

        Raises:
            ValueError: if the metric is not subclass of BaseMetric
        """
        self.custom_metrics = custom_metrics if custom_metrics is not None else []

        self.custom_metrics_name = []
        if len(self.custom_metrics) > 0:
            for metric in self.custom_metrics:
                if not isinstance(metric, BaseMetric):
                    raise ValueError("Custom metrics must be a subclass of BaseMetric.")
                self.custom_metrics_name.append(metric.name)

    def set_evaluation_result(self, metrics: list[BaseMetric]) -> None:
        """Evaluate default metrics.

        Raises:
            ValueError: if the metric is not implemented
        """
        for metric in metrics:
            metric.set_score(self.actual, self.predicted)

    def evaluate(self) -> None:
        """Evaluate default and custom metrics."""
        self.set_evaluation_result(self.default_metrics)
        self.set_evaluation_result(self.custom_metrics)

    def to_dict(self) -> dict:
        """Convert evaluation metrics to dictionary.

        Raises:
            ValueError: if the metrics are not evaluated yet

        Returns:
            dict: dictionary of evaluation metrics
        """
        metrics = self.default_metrics + self.custom_metrics
        for metric in metrics:
            if metric.score is None:
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
