from typing import Any, Optional, Type, get_args

import pandas as pd

from qxmt.evaluation.metrics.base import BaseMetric
from qxmt.evaluation.metrics.defaults_classification import (
    DEFAULT_CLF_METRICS_NAME,
    NAME2CLF_METRIC,
)
from qxmt.evaluation.metrics.defaults_regression import (
    DEFAULT_REG_METRICS_NAME,
    NAME2REG_METRIC,
)
from qxmt.evaluation.metrics.defaults_vqe import (
    DEFAULT_VQE_METRICS_NAME,
    NAME2VQE_METRIC,
)
from qxmt.utils import load_object_from_yaml


class Evaluation:
    """A class for evaluating model performance using various metrics.

    This class provides functionality to evaluate model predictions against actual values
    using both default and custom metrics. Results can be accessed as dictionaries or pandas DataFrames.

    Attributes:
        DEFAULT_METRICS_NAME (list[str]): List of default metric names
        NAME2METRIC (dict[str, Type[BaseMetric]]): Mapping of metric names to metric classes

    Examples:
        >>> from qxmt.evaluation.evaluation import Evaluation
        >>> params = {"actual": np.array([1, 0, 1]), "predicted": np.array([1, 1, 1])}
        >>> evaluation = Evaluation(params)
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
        params: dict[str, Any],
        default_metrics_name: Optional[list[str]] = None,
        custom_metrics: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Initialize the evaluation class with parameters and metrics.

        Args:
            params (dict[str, Any]): Dictionary containing evaluation parameters
            default_metrics_name (Optional[list[str]], optional):
                List of default metric names to use. If None, uses class default metrics.
            custom_metrics (Optional[list[dict[str, Any]]], optional):
                List of custom metric configurations. Each metric must be a subclass of BaseMetric.
        """
        self.params: dict[str, Any] = params
        self.default_metrics_name: list[str]
        self.custom_metrics_name: list[str]
        self.default_metrics: list[BaseMetric]
        self.custom_metrics: list[BaseMetric]

        self.init_default_metrics(default_metrics_name)
        self.init_custom_metrics(custom_metrics)

    def __repr__(self) -> str:
        return (
            f"Evaluation(params={self.params}, "
            f"default_metrics_name={self.default_metrics_name})"
            f"custom_metrics_name={self.custom_metrics_name}"
        )

    def init_default_metrics(self, default_metrics_name: Optional[list[str]]) -> None:
        """Initialize and validate default metrics.

        Args:
            default_metrics_name (Optional[list[str]]): List of default metric names to use

        Raises:
            ValueError: If any specified metric is not implemented
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
            custom_metrics (Optional[list[dict[str, Any]]]): List of custom metric configurations

        Raises:
            ValueError: If any custom metric is not a subclass of BaseMetric
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
        """Calculate scores for a list of metrics using the evaluation parameters.

        This method evaluates each metric using only the parameters it requires.
        If a metric requires parameters that are not available in self.params,
        a KeyError will be raised.

        Args:
            metrics (list[BaseMetric]): List of metrics to evaluate

        Raises:
            KeyError: If a required parameter is missing from self.params
        """
        for metric in metrics:
            metric.set_score(**self.params)

    def evaluate(self) -> None:
        """Evaluate all default and custom metrics."""
        self.set_evaluation_result(self.default_metrics)
        self.set_evaluation_result(self.custom_metrics)

    def to_dict(self) -> dict:
        """Convert evaluation results to a dictionary.

        Returns:
            dict: Dictionary containing metric names as keys and their scores as values

        Raises:
            ValueError: If metrics have not been evaluated yet
        """
        metrics = self.default_metrics + self.custom_metrics
        for metric in metrics:
            if (not metric.accept_none) and (metric.score is None):
                raise ValueError("Metrics are not evaluated yet.")

        data = {metric.name: metric.score for metric in metrics}
        return data

    def to_dataframe(
        self,
        id: Optional[str] = None,
        id_columns_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Convert evaluation results to a pandas DataFrame.

        Args:
            id (Optional[str], optional): Identifier for the evaluation (e.g., run_id)
            id_columns_name (Optional[str], optional): Name of the ID column in the DataFrame

        Returns:
            pd.DataFrame: DataFrame containing evaluation metrics
        """
        data = self.to_dict()
        df = pd.DataFrame([data])
        if (id is not None) and (id_columns_name is not None):
            df.insert(0, id_columns_name, id)

        return df


class ClassificationEvaluation(Evaluation):
    """Evaluation class specifically for classification tasks.

    Inherits from Evaluation and uses classification-specific metrics.
    """

    DEFAULT_METRICS_NAME = list(get_args(DEFAULT_CLF_METRICS_NAME))
    NAME2METRIC = NAME2CLF_METRIC


class RegressionEvaluation(Evaluation):
    """Evaluation class specifically for regression tasks.

    Inherits from Evaluation and uses regression-specific metrics.
    """

    DEFAULT_METRICS_NAME = list(get_args(DEFAULT_REG_METRICS_NAME))
    NAME2METRIC = NAME2REG_METRIC


class VQEEvaluation(Evaluation):
    """Evaluation class specifically for Variational Quantum Eigensolver (VQE) tasks.

    Inherits from Evaluation and uses VQE-specific metrics.
    """

    DEFAULT_METRICS_NAME = list(get_args(DEFAULT_VQE_METRICS_NAME))
    NAME2METRIC = NAME2VQE_METRIC
