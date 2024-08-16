from abc import ABC, abstractmethod
from logging import Logger
from typing import Optional

import numpy as np

from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


class BaseMetric(ABC):
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.score: Optional[float] = None

    @staticmethod
    @abstractmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray, **kwargs: dict) -> float:
        """define evaluation method for each metric.

        Args:
            actual (np.ndarray): array of actual value
            predicted (np.ndarray): array of predicted value
            **kwargs (dict): additional arguments

        Returns:
            float: evaluated score
        """
        pass

    def set_score(self, actual: np.ndarray, predicted: np.ndarray, **kwargs: dict) -> None:
        """Evaluated the score and set it to the score attribute.

        Args:
            actual (np.ndarray): array of actual value
            predicted (np.ndarray): array of predicted value
            **kwargs (dict): additional arguments
        """
        self.score = self.evaluate(actual, predicted, **kwargs)

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
