from pathlib import Path
from typing import Any

import dill
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from qxmt.constants import DEFAULT_N_JOBS
from qxmt.kernels.base import BaseKernel
from qxmt.models.base import BaseKernelModel


class QSVM(BaseKernelModel):
    """Quantum Support Vector Machine (QSVM) model.
    This class wraps the sklearn.svm.SVC class to provide a QSVM model.
    Then, many methods use the same interface as the sklearn.svm.SVC class.

    Examples:
        >>> from qxmt.models.qsvm import QSVM
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
        >>> model = QSVM(kernel=kernel)
        >>> model.fit(X_train, y_train)
        >>> model.predict(X_test)
        np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    """

    def __init__(self, kernel: BaseKernel, **kwargs: Any) -> None:
        """Initialize the QSVM model.

        Args:
            kernel (BaseKernel): kernel instance of BaseKernel class
        """
        super().__init__(kernel)
        self.model = SVC(kernel=self.kernel.compute_matrix, **kwargs)

    def cross_val_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_jobs: int = DEFAULT_N_JOBS,
        **kwargs: Any,
    ) -> np.ndarray:
        """Cross validation score of the model.

        Args:
            X (np.ndarray): numpy array of features
            y (np.ndarray): numpy array of target values

        Returns:
            np.ndarray: numpy array of scores
        """
        scores = cross_val_score(estimator=self.model, X=X, y=y, n_jobs=n_jobs, **kwargs)

        return scores

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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict the probability of target value with given features.
        This method is only available for SVC with probability=True.

        Args:
            X (np.ndarray): numpy array of features

        Returns:
            np.ndarray: numpy array of predicted probabilities
        """
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        """Save the model to the given path.

        Args:
            path (str | Path): path to save the model
        """
        # [TODO] Use pickle of joblib
        # AttributeError: Can't pickle local object 'BaseKernel._to_fm_instance.<locals>.CustomFeatureMap'
        dill.dump(self.model, open(path, "wb"))

    def load(self, path: str | Path) -> "QSVM":
        """Load the trained model from the given path.

        Args:
            path (str | Path): path to load the model

        Returns:
            QSVM: loaded QSVM model
        """
        # [TODO] Use pickle of joblib
        return dill.load(open(path, "rb"))

    def get_params(self) -> dict:
        """Get the parameters of the model."""
        return self.model.get_params()

    def set_params(self, params: dict) -> None:
        """Set the parameters of the model."""
        self.model.set_params(**params)
