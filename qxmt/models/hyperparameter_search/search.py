import inspect
from logging import Logger
from typing import Any, Callable, Optional

import numpy as np
import optuna
from optuna import Trial
from optuna.samplers import GridSampler, RandomSampler, TPESampler
from sklearn.model_selection import cross_val_score

from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


class HyperParameterSearch:
    """Hyperparameter search class for machine models using optuna.
    This class provides grid search , random search, and TPE search for hyperparameter optimization.
    The search space is defined by the user, and the search arguments can be customized.
    Reference: https://optuna.readthedocs.io/en/stable/reference/index.html

    Example:
        >>> from sklearn.svm import SVC
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from qxmt.models.hyperparameter_search.search import HyperParameterSearch
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>> model = SVC()
        >>> sampler_type = "tpe"
        >>> search_space = {
        ...     "C": [0.1, 1.0],
        ...     "gamma": [0.01, 0.1]
        ... }
        >>> search_args = {
        ...     "cv": 5,
        ...     "direction": "maximize",
        ...     "n_jobs": -1,
        ... }
        >>> searcher = HyperParameterSearch(X_train, y_train, model, sampler_type, search_space, search_args)
        >>> best_params = searcher.search()
        >>> best_params
        {'C': 0.8526745595533768, 'gamma': 0.01217052619278743}
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any,
        sampler_type: str,
        search_space: dict[str, list[Any]],
        search_args: Optional[dict[str, Any]],
        objective: Optional[Callable] = None,
        logger: Logger = LOGGER,
    ) -> None:
        """Initialize the hyperparameter search class.

        Args:
            X (np.ndarray): dataset for search
            y (np.ndarray): target values for search
            model (Any): model instance for hyperparameter search
            sampler_type (str): sampler type for hyperparameter search (random, grid, tpe)
            search_space (dict[str, list[Any]]): search space for hyperparameter search
            search_args (Optional[dict[str, Any]]): search arguments for hyperparameter search
            objective (Optional[Callable], optional): objective function for hyperparameter search. Defaults to None.
            logger (Logger, optional): logger instance. Defaults to LOGGER.

        Raises:
            ValueError: Sampler of search_args not matching search type
        """
        self.X = X
        self.y = y
        self.model = model
        self.sampler_type = sampler_type
        self.search_space = search_space
        self.search_args = search_args or {}
        self.objective = objective
        self.logger = logger

        self.scoring = self.search_args.get("scoring")

        sampler_from_type = self._get_sampler()
        if (search_args is not None) and (
            not isinstance(search_args.get("sampler", sampler_from_type), type(sampler_from_type))
        ):
            raise ValueError(
                f"""Sampler '{search_args['sampler']}' specified in search_args is
                 not matching sampler_type '{self.sampler_type}'."""
            )

    def _get_sampler(self) -> RandomSampler | GridSampler | TPESampler:
        """Get sampler instance based on sampler_type. If not specified, default to TPE sampler.

        Returns:
            Any: sampler instance
        """
        match self.sampler_type.lower():
            case "random":
                return RandomSampler()
            case "grid":
                return GridSampler(self.search_space)
            case "tpe":
                return TPESampler()
            case _:
                self.logger.warning(f"Unknown sampler_type '{self.sampler_type}', defaulting to TPE sampler.")
                return TPESampler()

    def _get_direction(self) -> str:
        """Get direction based on scoring. If not specified, default to 'maximize'.

        Returns:
            str: direction of optimization
        """
        direction = self.search_args.get("direction", None)
        if direction is None:
            if self.scoring in ["neg_mean_squared_error", "neg_log_loss"]:
                direction = "minimize"
            else:
                direction = "maximize"
            self.logger.info(f"Direction not specified, defaulting to '{direction}' based on scoring '{self.scoring}'.")

        return direction

    def _get_study_args(self) -> dict[str, Any]:
        """Get study arguments for optuna study for search_args.

        Returns:
            dict[str, Any]: optuna study arguments
        """
        study_args = {}
        if self.search_args is not None:
            study_args = {
                key: value
                for key, value in self.search_args.items()
                if key in inspect.signature(optuna.create_study).parameters.keys()
            }

        if study_args.get("sampler") is None:
            study_args["sampler"] = self._get_sampler()

        if study_args.get("direction") is None:
            study_args["direction"] = self._get_direction()

        return study_args

    def _get_optimize_args(self) -> dict[str, Any]:
        """Get optimize arguments for optuna study for search_args.

        Returns:
            dict[str, Any]: optuna optimize arguments
        """
        optimize_args = {}
        if self.search_args is not None:
            optimize_args = {
                key: value
                for key, value in self.search_args.items()
                if key in inspect.signature(optuna.study.Study.optimize).parameters.keys()
            }
        return optimize_args

    def search(self) -> dict[str, Any]:
        """Search the best hyperparameters for the model.

        Returns:
            dict[str, Any]: best hyperparameters found by search
        """
        study_args = self._get_study_args()
        study = optuna.create_study(**study_args)

        if self.objective is None:
            self.objective = self.default_objective

        optimize_args = self._get_optimize_args()
        study.optimize(self.objective, **optimize_args)

        self.logger.info(f"Best parameters: {study.best_params}")
        self.logger.info(f"Best value: {study.best_value}")

        return study.best_params

    def _format_params_to_optuna(self, trial: Trial) -> dict[str, Any]:
        """Set the parameters in the format of optuna.

        Args:
            trial (Trial): optuna trial instance

        Returns:
            dict[str, Any]: parameters in the format of optuna
        """
        params = {}
        for param_name, param_values in self.search_space.items():
            if len(param_values) == 2 and all(isinstance(v, int) for v in param_values):
                # define range of integers
                low, high = param_values
                params[param_name] = trial.suggest_int(param_name, low, high)
            elif len(param_values) == 2 and all(isinstance(v, float) for v in param_values):
                # define range of floats
                low, high = param_values
                params[param_name] = trial.suggest_float(param_name, low, high)
            else:
                # define categorical values
                params[param_name] = trial.suggest_categorical(param_name, param_values)

        return params

    def default_objective(self, trial: Trial) -> float:
        """Default objective function for hyperparameter search.

        Args:
            trial (Trial): optuna trial instance

        Returns:
            float: score of the model
        """
        params = self._format_params_to_optuna(trial)
        model = self.model.set_params(**params)

        # evaluate the model by mean score of cross validation
        cv = self.search_args.get("cv", 3)
        score = cross_val_score(model, self.X, self.y, cv=cv, scoring=self.scoring)

        return score.mean()
