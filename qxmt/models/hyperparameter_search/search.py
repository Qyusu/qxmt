from logging import Logger
from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


class HyperParameterSearch:
    """Hyperparameter search class for machine models.
    This class provides grid search and random search for hyperparameters.

    Example:
        >>> from sklearn.svm import SVC
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from qxmt.models.hyperparameter_search.search import HyperParameterSearch
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>> model = SVC()
        >>> search_type = "grid"
        >>> search_space = {
        ...     "C": [0.1, 1, 10]
        ... }
        >>> search_args = {
        ...     "cv": 5,
        ...     "n_jobs": -1,
        ...     "refit": True,
        ... }
        >>> searcher = HyperParameterSearch(model, search_type, search_space, search_args, X_train, y_train)
        >>> best_params = searcher.search()
        >>> best_params
        {'C': 0.1}
    """

    def __init__(
        self,
        model: Any,
        search_type: str,
        search_space: dict[str, list[Any]],
        search_args: Any,
        X: np.ndarray,
        y: np.ndarray,
        logger: Logger = LOGGER,
    ) -> None:
        """Initialize the hyperparameter search class.

        Args:
            model (Any): model instance for hyperparameter search
            search_type (str): search type for hyperparameter search (grid or random)
            search_space (dict[str, list[Any]]): search space for hyperparameter search
            search_args (Any): search arguments for hyperparameter search
            X_train (np.ndarray): dataset for search
            y_train (np.ndarray): target values for search
        """
        self.model = model
        self.search_type = search_type
        self.search_space = search_space
        self.search_args = search_args
        self.X = X
        self.y = y
        self.logger = logger

    def search(self) -> dict[str, Any]:
        """Search the best hyperparameters for the model.

        Raises:
            ValueError: Not supported search type

        Returns:
            dict[str, Any]: best hyperparameters
        """
        if not self.search_args.get("refit"):
            self.logger.info("refit parameter is set to False. Model will not be refitted by the best parameters.")

        match self.search_type.lower():
            case "grid":
                return self.grid_search()
            case "random":
                return self.random_search()
            case _:
                raise ValueError(f"Not supported search type: {self.search_type}")

    def grid_search(self) -> dict[str, Any]:
        """Grid search for hyperparameters.

        Returns:
            dict[str, Any]: best hyperparameters
        """
        searcher = GridSearchCV(self.model, self.search_space, **self.search_args)
        searcher.fit(self.X, self.y)
        return searcher.best_params_

    def random_search(self) -> dict[str, Any]:
        """Random search for hyperparameters.

        Returns:
            dict[str, Any]: best hyperparameters
        """
        if self.search_args.get("random_state") is None:
            self.logger.info("random_state is not set. Random search may not be reproducible.")

        searcher = RandomizedSearchCV(self.model, self.search_space, **self.search_args)
        searcher.fit(self.X, self.y)
        return searcher.best_params_
