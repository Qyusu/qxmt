from typing import Any

import numpy as np
import pytest
from _pytest.capture import CaptureFixture

from qxmt.evaluation.base import BaseMetric


class DummyMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__("dummy")

    @staticmethod
    def evaluate(actual: np.ndarray, predicted: np.ndarray, **kwargs: Any) -> float:
        return np.sum(np.add(actual, predicted))


@pytest.fixture(scope="module")
def dummy_values() -> tuple[np.ndarray, np.ndarray]:
    return np.array([0, 1, 1, 0, 1]), np.array([0, 1, 0, 0, 1])


class TestBaseMetric:
    def test_init(self) -> None:
        metric = DummyMetric()

        assert metric.name == "dummy"
        assert metric.score is None

    def test_set_score(self, dummy_values: tuple[np.ndarray, np.ndarray]) -> None:
        metric = DummyMetric()
        metric.set_score(dummy_values[0], dummy_values[1])

        assert metric.score == 5.0

    def test_print_score(self, dummy_values: tuple[np.ndarray, np.ndarray], capsys: CaptureFixture) -> None:
        metric = DummyMetric()

        with pytest.raises(ValueError):
            metric.print_score()

        metric.set_score(dummy_values[0], dummy_values[1])
        metric.print_score()
        captured = capsys.readouterr()
        assert captured.out == "dummy: 5.00\n"