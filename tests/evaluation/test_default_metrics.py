import numpy as np

from qk_manager import Accuracy


class TestAccuracy:
    def test_evaluate(self) -> None:
        actual = np.array([0, 1, 1, 0, 1])
        predicted = np.array([0, 1, 0, 0, 1])

        accuracy = Accuracy()
        score = accuracy.evaluate(actual, predicted)

        assert score == 0.8
