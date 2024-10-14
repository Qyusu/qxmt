import copy
from pathlib import Path
from typing import Any, Callable, Optional

import dill
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR

from qxmt.constants import DEFAULT_N_JOBS
from qxmt.kernels.base import BaseKernel
from qxmt.models.base import BaseKernelModel
from qxmt.models.hyperparameter_search.search import HyperParameterSearch


class QSVR(BaseKernelModel):
    """Quantum Support Vector Regression (QSVR) model.
    This class wraps the sklearn.svm.SVR class to provide a QSVR model.
    Then, many methods use the same interface as the sklearn.svm.SVR class.
    References: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

    Examples:
        >>> from qxmt.models.qsvc import QSVR
        >>> from qxmt.kernels.pennylane import FidelityKernel
        >>> from qxmt.feature_maps.pennylane.defaults import ZZFeatureMap
        >>> from qxmt.configs import DeviceConfig
        >>> from qxmt.devices.builder import DeviceBuilder
        >>> config = DeviceConfig(
        ...     platform="pennylane",
        ...     name="default.qubit",
        ...     n_qubits=2,
        ...     shots=1000,
        >>> )
        >>> device = DeviceBuilder(config).build()
        >>> feature_map = ZZFeatureMap(2, 2)
        >>> kernel = FidelityKernel(device, feature_map)
        >>> model = QSVR(kernel=kernel)
        >>> model.fit(X_train, y_train)
        >>> model.predict(X_test)
        np.array([-0.5441202 , -0.19483975, -0.19773099, -0.24928479,  0.12222858])
    """

    def __init__(self, kernel: BaseKernel, **kwargs: Any) -> None:
        """Initialize the QSVR model.

        Args:
            kernel (BaseKernel): kernel instance of BaseKernel class
        """
        super().__init__(kernel)
        self.model = SVR(kernel=self.kernel.compute_matrix, **kwargs)

    def cross_val_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_jobs: int = DEFAULT_N_JOBS,
        **kwargs: Any,
    ) -> np.ndarray:
        """Cross validation score of the QSVR model.
        Default to use the R^2 score.

        Args:
            X (np.ndarray): numpy array of features
            y (np.ndarray): numpy array of target values

        Returns:
            np.ndarray: numpy array of scores
        """
        scores = cross_val_score(estimator=self.model, X=X, y=y, n_jobs=n_jobs, **kwargs)

        return scores

    def hyperparameter_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        search_type: str,
        search_space: dict[str, list[Any]],
        search_args: dict[str, Any],
        objective: Optional[Callable] = None,
        refit: bool = True,
    ) -> dict[str, Any]:
        """Search the best hyperparameters for the QSVR model.

        Args:
            X (np.ndarray): dataset for search
            y (np.ndarray): target values for search
            search_type (str): search type for hyperparameter search (grid, random, tpe)
            search_space (dict[str, list[Any]]): search space for hyperparameter search
            search_args (dict[str, Any]): search arguments for hyperparameter search
            objective (Optional[Callable], optional): objective function for search. Defaults to None.
            refit (bool, optional): refit the model with best hyperparameters. Defaults to True.

        Raises:
            ValueError: Not supported search type

        Returns:
            dict[str, Any]: best hyperparameters
        """
        search_model = copy.deepcopy(self.model)

        if "scoring" not in search_args.keys():
            search_args["scoring"] = "r2"

        searcher = HyperParameterSearch(
            X=X,
            y=y,
            model=search_model,
            search_type=search_type,
            search_space=search_space,
            search_args=search_args,
            objective=objective,
        )
        best_params = searcher.search()

        if refit:
            self.model.set_params(**best_params)
            self.model.fit(X, y)

        return best_params

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        """Fit the model with given features and target values.

        Args:
            X (np.ndarray): numpy array of features
            y (np.ndarray): numpy array of target values
        """
        self.model.fit(X, y, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target value with given features.

        Args:
            X (np.ndarray): numpy array of features

        Returns:
            np.ndarray: numpy array of predicted values
        """
        return self.model.predict(X)

    def save(self, path: str | Path) -> None:
        """Save the model to the given path.

        Args:
            path (str | Path): path to save the model
        """
        # [TODO] Use pickle of joblib
        # AttributeError: Can't pickle local object 'BaseKernel._to_fm_instance.<locals>.CustomFeatureMap'
        dill.dump(self.model, open(path, "wb"))

    def load(self, path: str | Path) -> "QSVR":
        """Load the trained model from the given path.

        Args:
            path (str | Path): path to load the model

        Returns:
            QSVR: loaded QSVC model
        """
        # [TODO] Use pickle of joblib
        return dill.load(open(path, "rb"))

    def get_params(self) -> dict:
        """Get the parameters of the model."""
        return self.model.get_params()

    def set_params(self, params: dict) -> None:
        """Set the parameters of the model."""
        self.model.set_params(**params)
