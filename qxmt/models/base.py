from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseModel(ABC):
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
    def load(self, path: str | Path) -> None:
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


class BaseKernelModel(BaseModel):
    @abstractmethod
    def get_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """Get the kernel matrix of the given data.

        Args:
            X (np.ndarray): array of features

        Returns:
            np.ndarray: kernel matrix
        """
        pass
