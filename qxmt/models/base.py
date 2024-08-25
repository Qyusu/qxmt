from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from qxmt.constants import DEFAULT_N_JOBS
from qxmt.kernels.base import BaseKernel


class BaseMLModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: dict) -> None:
        """Fit the model with given data.

        Args:
            X (np.ndarray): array of features
            y (np.ndarray): array of target value
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target value with given features.

        Args:
            X (np.ndarray): array of features

        Returns:
            np.ndarray: array of predicted value
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save the model to the given path.

        Args:
            path (str | Path): path to save the model
        """
        pass

    @abstractmethod
    def load(self, path: str | Path) -> "BaseMLModel":
        """Load the model from the given path.

        Args:
            path (str | Path): path to load the model
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Get the parameters of the model.

        Returns:
            dict: dictionary of parameters
        """
        pass

    @abstractmethod
    def set_params(self, params: dict) -> None:
        """Set the parameters of the model.

        Args:
            params (dict): dictionary of parameters
        """
        pass


class BaseKernelModel(BaseMLModel):
    def __init__(self, kernel: BaseKernel) -> None:
        super().__init__()
        self.kernel = kernel

    def get_kernel_matrix(
        self,
        x_array_1: np.ndarray,
        x_array_2: np.ndarray,
        n_jobs: int = DEFAULT_N_JOBS,
    ) -> np.ndarray:
        """Get the kernel matrix of the given data.
        This method is alias of kernel.compute_matrix().

        Args:
            x_array_1 (np.ndarray): array of samples (ex: training data)
            x_array_2 (np.ndarray): array of samples (ex: test data)

        Returns:
            np.ndarray: kernel matrix
        """
        return self.kernel.compute_matrix(x_array_1, x_array_2, n_jobs=n_jobs)

    def plot_kernel_matrix(
        self,
        x_array_1: np.ndarray,
        x_array_2: np.ndarray,
        n_jobs: int = DEFAULT_N_JOBS,
    ) -> None:
        """Plot the kernel matrix of the given data.
        This method is alias of kernel.plot_matrix().

        Args:
            x_array_1 (np.ndarray): array of samples (ex: training data)
            x_array_2 (np.ndarray): array of samples (ex: test data)
        """
        self.kernel.plot_matrix(x_array_1, x_array_2, n_jobs=n_jobs)

    def plot_train_test_kernel_matrix(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        n_jobs: int = DEFAULT_N_JOBS,
    ) -> None:
        """Plot the kernel matrix of training and testing data.
        This method is alias of kernel.plot_train_test_matrix().

        Args:
            x_train (np.ndarray): array of training samples
            x_test (np.ndarray): array of testing samples
        """
        self.kernel.plot_train_test_matrix(x_train, x_test, n_jobs=n_jobs)
