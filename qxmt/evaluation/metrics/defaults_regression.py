from typing import Any, Literal, Type

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from qxmt.evaluation.metrics.base import BaseMetric


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


# set default regression metrics name list for evaluation
DEFAULT_REG_METRICS_NAME = Literal[
    "mean_absolute_error",
    "root_mean_squared_error",
    "r2_score",
]

NAME2REG_METRIC: dict[str, Type[BaseMetric]] = {
    "mean_absolute_error": MeanAbsoluteError,
    "root_mean_squared_error": RootMeanSquaredError,
    "r2_score": R2Score,
}
