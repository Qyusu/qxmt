from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from qxmt.evaluation.base import BaseMetric


class Accuracy(BaseMetric):
    def __init__(self, name: str = "accuracy") -> None:
        super().__init__(name)

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray, **kwargs: Any) -> float:
        score = accuracy_score(y_true=actual, y_pred=predicted, **kwargs)

        return float(score)


class Recall(BaseMetric):
    def __init__(self, name: str = "recall") -> None:
        super().__init__(name)

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray, **kwargs: Any) -> float:
        score = recall_score(y_true=actual, y_pred=predicted, **kwargs)

        return float(score)


class Precision(BaseMetric):
    def __init__(self, name: str = "precision") -> None:
        super().__init__(name)

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray, **kwargs: Any) -> float:
        score = precision_score(y_true=actual, y_pred=predicted, **kwargs)

        return float(score)


class F1Score(BaseMetric):
    def __init__(self, name: str = "f1_score") -> None:
        super().__init__(name)

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray, **kwargs: Any) -> float:
        score = f1_score(y_true=actual, y_pred=predicted, **kwargs)

        return float(score)


class TargetAlignmet(BaseMetric):
    def __init__(self, name: str = "target_alignment") -> None:
        super().__init__(name)

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray, **kwargs: Any) -> float:
        raise NotImplementedError
