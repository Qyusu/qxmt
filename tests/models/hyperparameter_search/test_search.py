import numpy as np
import pytest
from sklearn.svm import SVC

from qxmt.models.hyperparameter_search.search import HyperParameterSearch


@pytest.fixture(scope="function")
def model() -> SVC:
    return SVC()


@pytest.fixture(scope="function")
def dataset() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    return X, y


class TestHyperParameterSearch:
    def test_search(self, model: SVC, dataset: tuple[np.ndarray, np.ndarray]) -> None:
        search_space = {"C": [0.1, 1, 10]}
        search_args = {
            "cv": 5,
            "n_jobs": -1,
            "refit": True,
        }

        # invalid search type
        search_type = "invalid"
        with pytest.raises(ValueError):
            searcher = HyperParameterSearch(model, search_type, search_space, search_args, X=dataset[0], y=dataset[1])
            searcher.search()
