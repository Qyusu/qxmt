import numpy as np
import pytest
from optuna import Trial
from optuna.samplers import RandomSampler, TPESampler
from pytest_mock import MockFixture
from sklearn.svm import SVC

from qxmt.models.hyperparameter_search.search import HyperParameterSearch


class TestHyperParameterSearch:
    def test_init(self) -> None:
        search_args = {"sampler": RandomSampler()}

        # match sampler of search_args and search_type
        searcher = HyperParameterSearch(np.array([[1, 2], [3, 4]]), np.array([0, 1]), SVC(), "random", {}, search_args)
        assert searcher

        # not match sampler of search_args and search_type
        with pytest.raises(ValueError):
            searcher = HyperParameterSearch(np.array([[1, 2], [3, 4]]), np.array([0, 1]), SVC(), "tpe", {}, search_args)

    @pytest.mark.parametrize(
        ["search_args", "expected"],
        [
            pytest.param(
                {
                    "scoring": "neg_mean_squared_error",
                },
                "minimize",
                id="minizing scoring",
            ),
            pytest.param(
                {
                    "scoring": "accuracy",
                },
                "maximize",
                id="maximizing scoring",
            ),
            pytest.param(
                {
                    "scoring": None,
                },
                "maximize",
                id="default scoring and direction",
            ),
        ],
    )
    def test_get_direction(self, search_args: dict, expected: str) -> None:
        search_space = {"C": [0.1, 1.0], "gamma": [0.01, 0.1]}
        searcher = HyperParameterSearch(
            np.array([[1, 2], [3, 4]]), np.array([0, 1]), SVC(), "random", search_space, search_args
        )
        direction = searcher._get_direction()
        assert direction == expected

    @pytest.mark.parametrize(
        ["search_type", "search_args", "expected"],
        [
            pytest.param(
                "tpe",
                {"sampler": TPESampler(), "n_jobs": -1, "direction": "maximize"},
                {"sampler": TPESampler(), "direction": "maximize"},
                id="sampler and direction are provided by search_args",
            ),
            pytest.param(
                "random",
                {"cv": 5, "n_jobs": -1},
                {"sampler": RandomSampler(), "direction": "maximize"},
                id="paramter update by defualt logic",
            ),
        ],
    )
    def test_get_study_args(self, search_type: str, search_args: dict, expected: dict) -> None:
        searcher = HyperParameterSearch(
            np.array([[1, 2], [3, 4]]), np.array([0, 1]), SVC(), search_type, {}, search_args
        )
        study_args = searcher._get_study_args()
        for k, v in expected.items():
            if hasattr(v, "__class__"):
                assert isinstance(study_args[k], type(v))
            else:
                assert study_args[k] == v

    def test_format_params_to_optuna(self, mocker: MockFixture) -> None:
        search_space = {"C": [1, 10], "gamma": [0.01, 0.1], "kernel": ["rbf", "linear"]}
        searcher = HyperParameterSearch(np.array([[1, 2], [3, 4]]), np.array([0, 1]), SVC(), "random", search_space, {})
        trial_mock = mocker.MagicMock()
        params = searcher._format_params_to_optuna(trial_mock)

        assert params["C"] == trial_mock.suggest_int("C", 1, 10)
        assert params["gamma"] == trial_mock.suggest_float("gamma", 0.01, 0.1)
        assert params["kernel"] == trial_mock.suggest_categorical("kernel", ["rbf", "linear"])
