from pathlib import Path
from typing import Any, Optional, cast

import dill
import numpy as np
from sklearn.kernel_ridge import KernelRidge

from qxmt.kernels.base import BaseKernel
from qxmt.models.base import BaseKernelModel


class QRiggeRegressor(BaseKernelModel):
    """Quantum Rigge Regressor model.

    Detail of Kernel Rigge Regressor model is refered to the sklearn.kernel_ridge.KernelRidge class.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html

    Args:
        BaseKernelModel (_type_): kernel instance of BaseKernel class
    """

    def __init__(self, kernel: BaseKernel, alpha: float = 1.0, **kwargs: Any) -> None:
        """Initialize the QRiggeRegressor model.

        Args:
            kernel (BaseKernel): kernel instance of BaseKernel class
            alpha (float): Regularization strength; must be a positive float
        """
        super().__init__(kernel)
        self.fit_X: Optional[np.ndarray] = None
        self.model = KernelRidge(alpha=alpha, kernel="precomputed", **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model with given input data and target data.

        Args:
            X (np.ndarray): numpy array of input data
            y (np.ndarray): numpy array of target data
        """
        self.fit_X = X
        kernel_train_X = self.kernel.compute_matrix(self.fit_X, self.fit_X)
        self.model.fit(kernel_train_X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values of input data.

        Args:
            X (np.ndarray): numpy array of input data

        Returns:
            np.ndarray: numpy array of predicted target values
        """
        if self.fit_X is None:
            raise ValueError("The model is not trained yet.")
        else:
            kernel_pred_X = self.kernel.compute_matrix(X, self.fit_X)
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
            kernel_pred_X = self.kernel.compute_matrix(X, self.fit_X)
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
