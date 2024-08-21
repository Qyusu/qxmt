from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1 import make_axes_locatable

from qxmt.constants import DEFAULT_N_JOBS
from qxmt.feature_maps.base import BaseFeatureMap
from qxmt.kernels.utils import get_number_of_qubits, get_platform_from_device
from qxmt.types import QuantumDeviceType


class BaseKernel(ABC):
    def __init__(
        self,
        device: QuantumDeviceType,
        feature_map: BaseFeatureMap | Callable[[np.ndarray], None],
    ) -> None:
        self.device: QuantumDeviceType = device
        self.platform: str = get_platform_from_device(self.device)
        self.n_qubits: int = get_number_of_qubits(self.device)
        if callable(feature_map):
            feature_map = self._to_fm_instance(feature_map)
        self.feature_map = feature_map

    def _to_fm_instance(self, feature_map: Callable[[np.ndarray], None]) -> BaseFeatureMap:
        """Convert a feature map function to a BaseFeatureMap instance.

        Args:
            feature_map (Callable[[np.ndarray], None]): function that defines the feature map circuit

        Returns:
            BaseFeatureMap: instance of BaseFeatureMap
        """

        class CustomFeatureMap(BaseFeatureMap):
            def __init__(self, platform: str, n_qubits: int) -> None:
                super().__init__(platform, n_qubits)

            def feature_map(self, x: np.ndarray) -> None:
                self.check_input_shape(x)
                feature_map(x)

        return CustomFeatureMap(self.platform, self.n_qubits)

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

    def compute_matrix(self, x_array_1: np.ndarray, x_array_2: np.ndarray, n_jobs: int = DEFAULT_N_JOBS) -> np.ndarray:
        """Default implementation of kernel matrix computation.

        Args:
            x_array_1 (np.ndarray): array of samples (ex: training data)
            x_array_2 (np.ndarray): array of samples (ex: test data)
            n_jobs (int, optional): number of jobs for parallel computation. Defaults to DEFAULT_N_JOBS.

        Returns:
            np.ndarray: computed kernel matrix
        """

        def _compute_entry(i: int, j: int) -> tuple[int, int, float]:
            return i, j, self.compute(x_array_1[i], x_array_2[j])

        # compute each entry of the kernel matrix in parallel
        n_samples_1 = len(x_array_1)
        n_samples_2 = len(x_array_2)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_entry)(i, j) for i in range(n_samples_1) for j in range(n_samples_2)
        )

        kernel_matrix = np.zeros((n_samples_1, n_samples_2))
        for i, j, value in results:  # type: ignore
            kernel_matrix[i, j] = value

        return kernel_matrix

    def plot_matrix(
        self,
        x_array_1: np.ndarray,
        x_array_2: np.ndarray,
        save_path: Optional[str | Path] = None,
        n_jobs: int = DEFAULT_N_JOBS,
    ) -> None:
        """Plot kernel matrix.

        Args:
            x_array_1 (np.ndarray): array of samples (ex: training data)
            x_array_2 (np.ndarray): array of samples (ex: test data)
            save_path (Optional[str | Path], optional): save path for the plot. Defaults to None.
            n_jobs (int, optional): number of jobs for parallel computation. Defaults to DEFAULT_N_JOBS.
        """

        kernel_matrix = self.compute_matrix(x_array_1, x_array_2, n_jobs=n_jobs)
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
        n_jobs: int = DEFAULT_N_JOBS,
    ) -> None:
        """Plot kernel matrix for training and testing data.

        Args:
            x_train (np.ndarray): array of training samples
            x_test (np.ndarray): array of testing samples
            save_path (Optional[str | Path], optional): save path for the plot. Defaults to None.
            n_jobs (int, optional): number of jobs for parallel computation. Defaults to DEFAULT_N_JOBS.
        """
        train_kernel = self.compute_matrix(x_train, x_train, n_jobs=n_jobs)
        test_kernel = self.compute_matrix(x_test, x_train, n_jobs=n_jobs)

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
