import numpy as np
from sklearn.metrics import accuracy_score

from qk_manager.evaluation.base_metric import BaseMetric


class Accuracy(BaseMetric):
    def __init__(self, name: str = "accuracy") -> None:
        super().__init__(name)

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray) -> float:
        score = accuracy_score(
            y_true=actual,
            y_pred=predicted,
            normalize=True,
            sample_weight=None,
        )

        return float(score)
