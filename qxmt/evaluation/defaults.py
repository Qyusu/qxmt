from typing import Any, Literal

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from qxmt.evaluation.base import BaseMetric


class Accuracy(BaseMetric):
    """Accuracy metric class.

    Args:
        BaseMetric (_type_): Base metric class

    Examples:
        >>> import numpy as np
        >>> from qxmt.evaluation.defaults import Accuracy
        >>> metric = Accuracy()
        >>> metric.set_score(np.array([1, 0, 1]), np.array([1, 1, 1]))
        >>> metric.output_score()
        accuracy: 0.67
    """

    def __init__(self, name: str = "accuracy") -> None:
        """Initialize the accuracy metric.

        Args:
            name (str, optional): name of accuracy metric. Defaults to "accuracy".
        """
        super().__init__(name)

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray, **kwargs: Any) -> float:
        """Evaluate the accuracy score.

        Args:
            actual (np.ndarray): numpy array of actual values
            predicted (np.ndarray): numpy array of predicted values

        Returns:
            float: accuracy score
        """
        score = accuracy_score(y_true=actual, y_pred=predicted, **kwargs)

        return float(score)


class Recall(BaseMetric):
    """Recall metric class.

    Args:
        BaseMetric (_type_): Base metric class

    Examples:
        >>> import numpy as np
        >>> from qxmt.evaluation.defaults import Recall
        >>> metric = Recall()
        >>> metric.set_score(np.array([1, 0, 1]), np.array([1, 1, 1]))
        >>> metric.output_score()
        recall: 1.0
    """

    def __init__(self, name: str = "recall") -> None:
        """Initialize the recall metric.

        Args:
            name (str, optional): name of recall metric. Defaults to "recall".
        """
        super().__init__(name)

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray, **kwargs: Any) -> float:
        """Evaluate the recall score.

        Args:
            actual (np.ndarray): numpy array of actual values
            predicted (np.ndarray): numpy array of predicted values
            **kwargs (dict): additional keyword arguments. The following options are supported:
                - average (str): define averaging method

        Returns:
            float: recall score
        """
        if kwargs.get("average") is None:
            average = "binary" if len(np.unique(actual)) == 2 else "macro"
            kwargs["average"] = average
        score = recall_score(y_true=actual, y_pred=predicted, **kwargs)

        return float(score)


class Precision(BaseMetric):
    """Precision metric class.

    Args:
        BaseMetric (_type_): Base metric class

    Examples:
        >>> import numpy as np
        >>> from qxmt.evaluation.defaults import Precision
        >>> metric = Precision()
        >>> metric.set_score(np.array([1, 0, 1]), np.array([1, 1, 1]))
        >>> metric.output_score()
        precision: 0.67
    """

    def __init__(self, name: str = "precision") -> None:
        """Initialize the precision metric.

        Args:
            name (str, optional): name of precision metric. Defaults to "precision".
        """
        super().__init__(name)

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray, **kwargs: Any) -> float:
        """Evaluate the precision score.

        Args:
            actual (np.ndarray): numpy array of actual values
            predicted (np.ndarray): numpy array of predicted values
            **kwargs (dict): additional keyword arguments. The following options are supported:
                - average (str): define averaging method

        Returns:
            float: precision score
        """
        if kwargs.get("average") is None:
            average = "binary" if len(np.unique(actual)) == 2 else "macro"
            kwargs["average"] = average
        score = precision_score(y_true=actual, y_pred=predicted, **kwargs)

        return float(score)


class F1Score(BaseMetric):
    """F1 score metric

    Args:
        BaseMetric (_type_): Base metric class

    Examples:
        >>> import numpy as np
        >>> from qxmt.evaluation.defaults import F1Score
        >>> metric = F1Score()
        >>> metric.set_score(np.array([1, 0, 1]), np.array([1, 1, 1]))
        >>> metric.output_score()
        f1_score: 0.8
    """

    def __init__(self, name: str = "f1_score") -> None:
        """Initialize the F1 score metric.

        Args:
            name (str, optional): name of f1-score metric. Defaults to "f1_score".
        """
        super().__init__(name)

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray, **kwargs: Any) -> float:
        """Evaluate the F1 score.

        Args:
            actual (np.ndarray): numpy array of actual values
            predicted (np.ndarray): numpy array of predicted values
            **kwargs (dict): additional keyword arguments. The following options are supported:
                - average (str): define averaging method

        Returns:
            float: F1 score
        """
        if kwargs.get("average") is None:
            average = "binary" if len(np.unique(actual)) == 2 else "macro"
            kwargs["average"] = average
        score = f1_score(y_true=actual, y_pred=predicted, **kwargs)

        return float(score)


class TargetAlignmet(BaseMetric):
    def __init__(self, name: str = "target_alignment") -> None:
        super().__init__(name)

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray, **kwargs: Any) -> float:
        raise NotImplementedError


# set default metrics name list for evaluation
DEFAULT_METRICS_NAME = Literal["accuracy", "precision", "recall", "f1_score"]

NAME2METRIC = {
    "accuracy": Accuracy,
    "precision": Precision,
    "recall": Recall,
    "f1_score": F1Score,
}
