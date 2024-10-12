from typing import Any, Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

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
                - pos_label (int): positive label for binary classification (default: max value)

        Returns:
            float: recall score
        """
        if kwargs.get("average") is None:
            average = "binary" if len(np.unique(actual)) == 2 else "macro"
            kwargs["average"] = average

        # skleran default pos_label is 1.
        # it is handle not enclude 1 label case. ex) [0, 8] -> pos_label=8
        if (kwargs.get("average") == "binary") and (kwargs.get("pos_label") is None):
            pos_label = max(np.unique(actual))
            kwargs["pos_label"] = pos_label

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
                - pos_label (int): positive label for binary classification (default: max value)

        Returns:
            float: precision score
        """
        if kwargs.get("average") is None:
            average = "binary" if len(np.unique(actual)) == 2 else "macro"
            kwargs["average"] = average

        # skleran default pos_label is 1.
        # it is handle not enclude 1 label case. ex) [0, 8] -> pos_label=8
        if (kwargs.get("average") == "binary") and (kwargs.get("pos_label") is None):
            pos_label = max(np.unique(actual))
            kwargs["pos_label"] = pos_label

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
                - pos_label (int): positive label for binary classification (default: max value)

        Returns:
            float: F1 score
        """
        if kwargs.get("average") is None:
            average = "binary" if len(np.unique(actual)) == 2 else "macro"
            kwargs["average"] = average

        # skleran default pos_label is 1.
        # it is handle not enclude 1 label case. ex) [0, 8] -> pos_label=8
        if (kwargs.get("average") == "binary") and (kwargs.get("pos_label") is None):
            pos_label = max(np.unique(actual))
            kwargs["pos_label"] = pos_label

        score = f1_score(y_true=actual, y_pred=predicted, **kwargs)

        return float(score)


class TargetAlignmet(BaseMetric):
    def __init__(self, name: str = "target_alignment") -> None:
        super().__init__(name)

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray, **kwargs: Any) -> float:
        raise NotImplementedError


class MeanAbsoluteError(BaseMetric):
    """Mean absolute error metric class.

    Args:
        BaseMetric (_type_): Base metric class

    Examples:
        >>> import numpy as np
        >>> from qxmt.evaluation.defaults import MeanAbsoluteError
        >>> metric = MeanAbsoluteError()
        >>> metric.set_score(np.array([1, 0, 1]), np.array([1, 1, 1]))
        >>> metric.output_score()
        mean_absolute_error: 0.33
    """

    def __init__(self, name: str = "mean_absolute_error") -> None:
        """Initialize the mean absolute error metric.

        Args:
            name (str, optional): name of mean absolute error metric. Defaults to "mean_absolute_error".
        """
        super().__init__(name)

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray, **kwargs: Any) -> float:
        """Evaluate the mean absolute error.

        Args:
            actual (np.ndarray): numpy array of actual values
            predicted (np.ndarray): numpy array of predicted values

        Returns:
            float: mean absolute error score
        """
        score = mean_absolute_error(y_true=actual, y_pred=predicted, **kwargs)
        return float(score)


class RootMeanSquaredError(BaseMetric):
    """Root mean squared error metric class.

    Args:
        BaseMetric (_type_): Base metric class

    Examples:
        >>> import numpy as np
        >>> from qxmt.evaluation.defaults import RootMeanSquaredError
        >>> metric = RootMeanSquaredError()
        >>> metric.set_score(np.array([1, 0, 1]), np.array([1, 1, 1]))
        >>> metric.output_score()
        root_mean_squared_error: 0.58
    """

    def __init__(self, name: str = "root_mean_squared_error") -> None:
        """Initialize the root mean squared error metric.

        Args:
            name (str, optional): name of root mean squared error metric. Defaults to "root_mean_squared_error".
        """
        super().__init__(name)

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray, **kwargs: Any) -> float:
        """Evaluate the root mean squared error.

        Args:
            actual (np.ndarray): numpy array of actual values
            predicted (np.ndarray): numpy array of predicted values

        Returns:
            float: root mean squared error score
        """
        score = root_mean_squared_error(y_true=actual, y_pred=predicted, **kwargs)
        return float(score)


class R2Score(BaseMetric):
    """R2 score metric class.

    Args:
        BaseMetric (_type_): Base metric class

    Examples:
        >>> import numpy as np
        >>> from qxmt.evaluation.defaults import R2Score
        >>> metric = R2Score()
        >>> metric.set_score(np.array([1, 0, 1]), np.array([1, 1, 1]))
        >>> metric.output_score()
        r2_score: -0.5
    """

    def __init__(self, name: str = "r2_score") -> None:
        """Initialize the R2 score metric.

        Args:
            name (str, optional): name of R2 score metric. Defaults to "r2_score".
        """
        super().__init__(name)

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray, **kwargs: Any) -> float:
        """Evaluate the R2 score.

        Args:
            actual (np.ndarray): numpy array of actual values
            predicted (np.ndarray): numpy array of predicted values

        Returns:
            float: R2 score
        """
        score = r2_score(y_true=actual, y_pred=predicted, **kwargs)
        return float(score)


# set default metrics name list for evaluation
DEFAULT_METRICS_NAME = Literal[
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "mean_absolute_error",
    "root_mean_squared_error",
    "r2_score",
]

NAME2METRIC = {
    "accuracy": Accuracy,
    "precision": Precision,
    "recall": Recall,
    "f1_score": F1Score,
    "mean_absolute_error": MeanAbsoluteError,
    "root_mean_squared_error": RootMeanSquaredError,
    "r2_score": R2Score,
}
