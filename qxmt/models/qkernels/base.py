from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from qxmt.constants import DEFAULT_N_JOBS
from qxmt.feature_maps import BaseFeatureMap
from qxmt.kernels import BaseKernel


class BaseMLModel(ABC):
    """Base class for quantum machine learning models.
    This class is an abstract class for qunatum machine learning models.
    If you want to implement a new quantum machine learning model, you should inherit this class.
    For compatibility with QXMT framework, you should implement some methods such as fit,
    predict, save, load, get_params, and set_params.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        """Fit the model with given data.

        Args:
            X (np.ndarray): array of features
            y (np.ndarray): array of target value
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, bar_label: str = "") -> np.ndarray:
        """Predict the target value with given features.

        Args:
            X (np.ndarray): array of features
            bar_label (str): label for progress bar. Defaults to "".

        Returns:
            np.ndarray: array of predicted value
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save the model to the given path.
        [TODO]: standardize the save/load method using same library

        Args:
            path (str | Path): path to save the model
        """
        pass

    @abstractmethod
    def load(self, path: str | Path) -> "BaseMLModel":
        """Load the model from the given path.
        [TODO]: standardize the save/load method using same library

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
    """Base class for kernel-based quantum machine learning models.
    This class is an abstract class for kernel-based quantum machine learning models.
    If you want to implement a new kernel-based quantum machine learning model, you should inherit this class.
    This class requires a kernel instance of BaseKernel class.
    The user can use any Feature Map or Kernel to be used, as long as it follows the interface of the BaseKernel class.
    """

    def __init__(self, kernel: BaseKernel, n_jobs: int = DEFAULT_N_JOBS, show_progress: bool = True) -> None:
        """Initialize the kernel model.

        Args:
            kernel (BaseKernel): kernel instance of BaseKernel class
            n_jobs (int): number of parallel jobs. Defaults to DEFAULT_N_JOBS.
            show_progress (bool): flag for showing progress bar. Defaults to True.
        """
        super().__init__()
        self.kernel = kernel
        self.n_jobs = n_jobs
        self.show_progress = show_progress

    def get_feature_map(self) -> BaseFeatureMap:
        """Get the feature map of the model.

        Returns:
            BaseFeatureMap: feature map instance
        """
        return self.kernel.feature_map

    def get_kernel_matrix(
        self,
        x_array_1: np.ndarray,
        x_array_2: np.ndarray,
    ) -> np.ndarray:
        """Get the kernel matrix of the given data.
        This method is alias of kernel.compute_matrix().

        Args:
            x_array_1 (np.ndarray): array of samples (ex: training data)
            x_array_2 (np.ndarray): array of samples (ex: test data)

        Returns:
            np.ndarray: kernel matrix
        """
        kernel_matrix, _ = self.kernel.compute_matrix(
            x_array_1,
            x_array_2,
            return_shots_resutls=False,
            n_jobs=self.n_jobs,
            show_progress=self.show_progress,
        )

        return kernel_matrix

    def plot_kernel_matrix(
        self,
        x_array_1: np.ndarray,
        x_array_2: np.ndarray,
    ) -> None:
        """Plot the kernel matrix of the given data.
        This method is alias of kernel.plot_matrix().

        Args:
            x_array_1 (np.ndarray): array of samples (ex: training data)
            x_array_2 (np.ndarray): array of samples (ex: test data)
        """
        self.kernel.plot_matrix(x_array_1, x_array_2, n_jobs=self.n_jobs, show_progress=self.show_progress)

    def plot_train_test_kernel_matrix(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
    ) -> None:
        """Plot the kernel matrix of training and testing data.
        This method is alias of kernel.plot_train_test_matrix().

        Args:
            x_train (np.ndarray): array of training samples
            x_test (np.ndarray): array of testing samples
        """
        self.kernel.plot_train_test_matrix(x_train, x_test, n_jobs=self.n_jobs, show_progress=self.show_progress)
