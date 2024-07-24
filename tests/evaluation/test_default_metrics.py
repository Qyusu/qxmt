import numpy as np

from quri import Accuracy, F1Score, Precision, Recall


class TestAccuracy:
    def test_evaluate(self) -> None:
        actual = np.array([0, 1, 1, 0, 1])
        predicted = np.array([0, 1, 0, 0, 1])

        accuracy = Accuracy()
        score = accuracy.evaluate(actual, predicted)

        assert score == 0.8


class TestPrecision:
    def test_evaluate(self) -> None:
        actual = np.array([0, 1, 1, 0, 1])
        predicted = np.array([0, 1, 0, 0, 1])

        precision = Precision()
        score = precision.evaluate(actual, predicted)

        assert round(score, 2) == 1.0


class TestRecall:
    def test_evaluate(self) -> None:
        actual = np.array([0, 1, 1, 0, 1])
        predicted = np.array([0, 1, 0, 0, 1])

        recall = Recall()
        score = recall.evaluate(actual, predicted)

        assert round(score, 2) == 0.67


class TestF1Score:
    def test_evaluate(self) -> None:
        actual = np.array([0, 1, 1, 0, 1])
        predicted = np.array([0, 1, 0, 0, 1])

        f1_score = F1Score()
        score = f1_score.evaluate(actual, predicted)

        assert round(score, 2) == 0.8
