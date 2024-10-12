from typing import Any, Optional, Type, get_args

import numpy as np
import pandas as pd

from qxmt.evaluation.base import BaseMetric
from qxmt.evaluation.defaults_classification import (
    DEFAULT_CLF_METRICS_NAME,
    NAME2CLF_METRIC,
)
from qxmt.evaluation.defaults_regression import (
    DEFAULT_REG_METRICS_NAME,
    NAME2REG_METRIC,
)
from qxmt.utils import load_object_from_yaml


class Evaluation:
    """Evaluation class for model evaluation.
    This class is used to evaluate the model performance by comparing the actual and predicted values.
    Evaluation metrics are defined by default and custom metrics.
    Furthermore, the result can be accessed as a dictionary or DataFrame.

    Examples:
        >>> from qxmt.evaluation.evaluation import Evaluation
        >>> actual = np.array([1, 0, 1])
        >>> predicted = np.array([1, 1, 1])
        >>> evaluation = Evaluation(actual, predicted)
        >>> evaluation.evaluate()
        >>> evaluation.to_dict()
        {'accuracy': 0.6666666666666666, 'precision': 0.6666666666666666, 'recall': 1.0, 'f1_score': 0.8}
        >>> evaluation.to_dataframe()
           accuracy  precision  recall  f1_score
        0  0.666667   0.666667     1.0       0.8
    """

    DEFAULT_METRICS_NAME: list[str] = []
    NAME2METRIC: dict[str, Type[BaseMetric]] = {}

    def __init__(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        default_metrics_name: Optional[list[str]] = None,
        custom_metrics: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Initialize the evaluation class.

        Args:
            actual (np.ndarray): numpy array of actual values
            predicted (np.ndarray): numpy array of predicted values
            default_metrics_name (Optional[list[str]], optional):
                metrics name list defined by default. Defaults to None.
            custom_metrics (Optional[list[dict[str, Any]]], optional):
                metrics name list defined by user custom. Defaults to None.
        """
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
            self.default_metrics_name = self.DEFAULT_METRICS_NAME

        self.default_metrics = []
        for metric_name in self.default_metrics_name:
            if metric_name not in self.NAME2METRIC:
                raise ValueError(f"{metric_name} is not implemented.")

            metric = self.NAME2METRIC[metric_name](name=metric_name)
            self.default_metrics.append(metric)

    def init_custom_metrics(self, custom_metrics: Optional[list[dict[str, Any]]]) -> None:
        """Initialize and validate custom metrics.

        Args:
            custom_metrics (Optional[list[dict[str, Any]]]): list of custom metrics. Defaults to None.

        Raises:
            ValueError: if the metric is not subclass of BaseMetric
        """
        self.custom_metrics_name = []
        self.custom_metrics = []

        if custom_metrics is not None:
            for metric_config in custom_metrics:
                metric = load_object_from_yaml(config=metric_config)
                if not isinstance(metric, BaseMetric):
                    raise ValueError("Custom metrics must be a subclass of BaseMetric.")

                self.custom_metrics.append(metric)
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


class ClassificationEvaluation(Evaluation):
    DEFAULT_METRICS_NAME = list(get_args(DEFAULT_CLF_METRICS_NAME))
    NAME2METRIC = NAME2CLF_METRIC


class RegressionEvaluation(Evaluation):
    DEFAULT_METRICS_NAME = list(get_args(DEFAULT_REG_METRICS_NAME))
    NAME2METRIC = NAME2REG_METRIC
