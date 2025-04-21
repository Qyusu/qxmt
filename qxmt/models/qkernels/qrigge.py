import copy
from pathlib import Path
from typing import Any, Optional, cast

import dill
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score

from qxmt.constants import DEFAULT_N_JOBS
from qxmt.kernels import BaseKernel
from qxmt.models.hyperparameter_search import HyperParameterSearch
from qxmt.models.qkernels import BaseKernelModel


class QRiggeRegressor(BaseKernelModel):
    """Quantum Rigge Regressor model.

    Detail of Kernel Rigge Regressor model is refered to the sklearn.kernel_ridge.KernelRidge class.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html

    Args:
        BaseKernelModel (_type_): kernel instance of BaseKernel class
    """

    def __init__(
        self,
        kernel: BaseKernel,
        n_jobs: int = DEFAULT_N_JOBS,
        show_progress: bool = True,
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the QRiggeRegressor model.

        Args:
            kernel (BaseKernel): kernel instance of BaseKernel class
            n_jobs (int): number of jobs for parallel computation
            show_progress (bool): flag for showing progress bar
            alpha (float): Regularization strength; must be a positive float
        """
        super().__init__(kernel, n_jobs, show_progress)
        self.fit_X: Optional[np.ndarray] = None
        self.model = KernelRidge(alpha=alpha, kernel="precomputed", **kwargs)

    def __getattr__(self, name: str) -> Any:
        # if the attribute is in the model, return it
        if hasattr(self.model, name):
            return getattr(self.model, name)
        # if the attribute is not in the model, raise AttributeError
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def cross_val_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Cross validation score of the Quantum Kernel Ridge Regressor model.
        Default to use the R^2 score.

        Args:
            X (np.ndarray): numpy array of input data
            y (np.ndarray): numpy array of target data
            **kwargs (dict): additional arguments

        Returns:
            np.ndarray: array of scores of the estimator for each run of the cross validation
        """
        kernel_X, _ = self.kernel.compute_matrix(
            X, X, return_shots_resutls=False, n_jobs=self.n_jobs, show_progress=self.show_progress
        )
        scores = cross_val_score(estimator=self.model, X=kernel_X, y=y, n_jobs=self.n_jobs, **kwargs)
        return scores

    def hyperparameter_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampler_type: str,
        search_space: dict[str, list[Any]],
        search_args: dict[str, Any],
        objective: Optional[Any] = None,
        refit: bool = True,
    ) -> dict[str, Any]:
        """Search the best hyperparameters for the Quantum Kernel Ridge Regressor model.

        Args:
            X (np.ndarray): dataset for search
            y (np.ndarray): target values for search
            sampler_type (str): sampler type for hyperparameter search (grid, random, tpe)
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
        X_kernel, _ = self.kernel.compute_matrix(
            X, X, return_shots_resutls=False, n_jobs=self.n_jobs, show_progress=self.show_progress
        )

        if "scoring" not in search_args.keys():
            search_args["scoring"] = "r2"

        searcher = HyperParameterSearch(
            X=X_kernel,
            y=y,
            model=search_model,
            sampler_type=sampler_type,
            search_space=search_space,
            search_args=search_args,
            objective=objective,
        )
        best_params = searcher.search()

        if refit:
            self.model.set_params(**best_params)
            self.model.fit(X_kernel, y)

        return best_params

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        save_shots_path: Optional[Path | str] = None,
    ) -> None:
        """Fit the model with given input data and target data.

        Args:
            X (np.ndarray): numpy array of input data
            y (np.ndarray): numpy array of target data
            save_shots_path (Optional[Path | str], optional): save path for the shot results. Defaults to None.
        """
        self.fit_X = X
        if save_shots_path is not None:
            kernel_train_X, shots_matrix = self.kernel.compute_matrix(
                self.fit_X,
                self.fit_X,
                return_shots_resutls=True,
                n_jobs=self.n_jobs,
                bar_label="Train",
                show_progress=self.show_progress,
            )
            if shots_matrix is not None:
                self.kernel.save_shots_results(shots_matrix, save_shots_path)
        else:
            kernel_train_X, _ = self.kernel.compute_matrix(
                self.fit_X,
                self.fit_X,
                return_shots_resutls=False,
                n_jobs=self.n_jobs,
                bar_label="Train",
                show_progress=self.show_progress,
            )
        self.model.fit(kernel_train_X, y)

    def predict(self, X: np.ndarray, bar_label: str = "") -> np.ndarray:
        """Predict target values of input data.

        Args:
            X (np.ndarray): numpy array of input data
            bar_label (str): label for progress bar. Defaults to "".

        Returns:
            np.ndarray: numpy array of predicted target values
        """
        if self.fit_X is None:
            raise ValueError("The model is not trained yet.")
        else:
            kernel_pred_X, _ = self.kernel.compute_matrix(
                X,
                self.fit_X,
                return_shots_resutls=False,
                n_jobs=self.n_jobs,
                bar_label=bar_label,
                show_progress=self.show_progress,
            )
        return self.model.predict(kernel_pred_X)

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
        """Return the coefficient of determination R^2 of the prediction.

        Args:
            X (np.ndarray): numpy array of input data
            y (np.ndarray): numpy array of target data

        Returns:
            float: R^2 of the prediction
        """
        if self.fit_X is None:
            raise ValueError("The model is not trained yet.")
        else:
            kernel_pred_X, _ = self.kernel.compute_matrix(
                X, self.fit_X, return_shots_resutls=False, n_jobs=self.n_jobs, show_progress=self.show_progress
            )
        return cast(float, self.model.score(kernel_pred_X, y, sample_weight=sample_weight))

    def save(self, path: str | Path) -> None:
        """Save the model to the given path.

        Args:
            path (str | Path): path to save the model
        """
        dill.dump(self.model, open(path, "wb"))

    def load(self, path: str | Path) -> "QRiggeRegressor":
        """Load the trained model from the given path.

        Args:
            path (str | Path): path to load the model

        Returns:
            QRiggeRegressor: loaded QRiggeRegressor model
        """
        return dill.load(open(path, "rb"))

    def get_params(self) -> dict:
        """Get the parameters of the model."""
        return self.model.get_params()

    def set_params(self, params: dict) -> None:
        """Set the parameters of the model."""
        self.model.set_params(**params)
