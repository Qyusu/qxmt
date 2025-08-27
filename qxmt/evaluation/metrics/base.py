from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Optional

from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


class BaseMetric(ABC):
    """Base class for evaluation metric.
    This class is used to define the evaluation metric for the model and visualization.
    Provide a common interface within the QXMT library by absorbing differences between metrics.

    Examples:
        >>> import numpy as np
        >>> from qxmt.evaluation.base import BaseMetric
        >>> class CustomMetric(BaseMetric):
        ...     def __init__(self, name: str) -> None:
        ...         super().__init__(name)
        ...
        ...     @staticmethod
        ...     def evaluate(actual: np.ndarray, predicted: np.ndarray, **kwargs: Any) -> float:
        ...         return np.mean(np.abs(actual - predicted))
        ...
        >>> metric = CustomMetric("mean_absolute_error")
        >>> metric.set_score(np.array([1, 3, 3]), np.array([1, 2, 3]))
        >>> metric.output_score()
        mean_absolute_error: 0.33
    """

    def __init__(self, name: str) -> None:
        """Base class for evaluation metric.

        Args:
            name (str): name of the metric. It is used for the column name of the output and DataFrame.
        """
        self.name: str = name
        self.score: Optional[float] = None
        self.accept_none: bool = False

    @staticmethod
    @abstractmethod
    def evaluate(**kwargs: Any) -> float:
        """define evaluation method for each metric.

        Args:
            **kwargs (dict): additional arguments

        Returns:
            float: evaluated score
        """
        pass

    def set_score(self, **kwargs: Any) -> None:
        """Evaluated the score and set it to the score attribute.

        Args:
            **kwargs (dict): additional arguments
        """
        self.score = self.evaluate(**kwargs)

    def output_score(self, logger: Logger = LOGGER) -> None:
        """Output the evaluated score on standard output.

        Args:
            logger (Logger, optional): logger object. Defaults to LOGGER.

        Raises:
            ValueError: if the score is not evaluated yet
        """
        if self.score is None:
            raise ValueError("Score is not evaluated yet.")

        logger.info(f"{self.name}: {self.score:.2f}")
