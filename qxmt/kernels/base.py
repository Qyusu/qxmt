from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from mpl_toolkits.axes_grid1 import make_axes_locatable

from qxmt.feature_maps.base import BaseFeatureMap


class BaseKernel(ABC):
    def __init__(self, device: qml.Device, feature_map: Optional[BaseFeatureMap] = None) -> None:
        self.device: qml.Device = device
        self.feature_map: Optional[BaseFeatureMap] = feature_map
        self.n_qubits: int = len(self.device.wires)

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

    def plot_matrix(
        self,
        x_array_1: np.ndarray,
        x_array_2: np.ndarray,
        save_path: Optional[str | Path] = None,
    ) -> None:
        """Plot kernel matrix.

        Args:
            x_array_1 (np.ndarray): array of samples (ex: training data)
            x_array_2 (np.ndarray): array of samples (ex: test data)
        """

        kernel_matrix = self.compute_matrix(x_array_1, x_array_2)
        plt.imshow(np.asmatrix(kernel_matrix), interpolation="nearest", origin="upper", cmap="viridis")
        plt.colorbar()
        plt.title("Kernel matrix")
        if save_path is not None:
            plt.savefig(save_path)

        plt.show()

    def plot_train_test_matrix(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        save_path: Optional[str | Path] = None,
    ) -> None:
        """Plot kernel matrix for training and testing data.

        Args:
            x_train (np.ndarray): array of training samples
            x_test (np.ndarray): array of testing samples
        """
        train_kernel = self.compute_matrix(x_train, x_train)
        test_kernel = self.compute_matrix(x_test, x_train)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        train_axs = axs[0]  # type: ignore
        im_train = train_axs.imshow(np.asmatrix(train_kernel), interpolation="nearest", origin="upper", cmap="Blues")
        train_axs.set_title("Training kernel matrix")
        divider_train = make_axes_locatable(train_axs)
        cax_train = divider_train.append_axes("right", size="5%", pad=0.2)
        fig.colorbar(im_train, cax=cax_train)

        test_axs = axs[1]  # type: ignore
        im_test = test_axs.imshow(np.asmatrix(test_kernel), interpolation="nearest", origin="upper", cmap="Reds")
        test_axs.set_title("Testing kernel matrix")
        divider_test = make_axes_locatable(test_axs)
        cax_test = divider_test.append_axes("right", size="5%", pad=0.2)
        fig.colorbar(im_test, cax=cax_test)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)

        plt.show()
