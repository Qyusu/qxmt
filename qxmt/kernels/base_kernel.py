from abc import ABC, abstractmethod

import numpy as np
import pennylane as qml

from qxmt.feature_maps.base_feature_map import BaseFeatureMap


class BaseKernel(ABC):
    def __init__(self, device: qml.Device, feature_map: BaseFeatureMap) -> None:
        self.device = device
        self.feature_map = feature_map

    @abstractmethod
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel value between two samples.

        Args:
            x1 (np.ndarray): array of one sample
            x2 (np.ndarray): array of one sample

        Returns:
            float: computed kernel value
        """
        pass

    def compute_matrix(self, x_array_1: np.ndarray, x_array_2: np.ndarray) -> np.ndarray:
        """Default implementation of kernel matrix computation.

        Args:
            x_array_1 (np.ndarray): array of samples (ex: training data)
            x_array_2 (np.ndarray): array of samples (ex: test data)

        Returns:
            np.ndarray: computed kernel matrix
        """
        n_samples_1 = len(x_array_1)
        n_samples_2 = len(x_array_2)
        kernel_matrix = np.zeros((n_samples_1, n_samples_2))

        for i in range(n_samples_1):
            for j in range(n_samples_2):
                kernel_matrix[i, j] = self.compute(x_array_1[i], x_array_2[j])

        return kernel_matrix
