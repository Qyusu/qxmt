from abc import ABC, abstractmethod
from itertools import product
from pathlib import Path
from types import FunctionType
from typing import Callable, Optional, cast

import h5py
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1 import make_axes_locatable

from qxmt.constants import DEFAULT_N_JOBS
from qxmt.devices.base import BaseDevice
from qxmt.exceptions import DeviceSettingError
from qxmt.feature_maps.base import BaseFeatureMap


class BaseKernel(ABC):
    """
    Base kernel class for quantum kernel computation.
    This class is used to compute the kernel value between two samples.
    If defining a custom kernel, inherit this class and implement the `compute()` method.
    The feature map used to compute the kernel value is defined in the constructor.
    It is possible to use a feature map instance or a function that defines the feature map circuit.

    Examples:
        >>> import numpy as np
        >>> from typing import Callable
        >>> from qxmt.kernels.base import BaseKernel
        >>> from qxmt.feature_maps.pennylane.defaults import ZZFeatureMap
        >>> from qxmt.configs import DeviceConfig
        >>> from qxmt.devices.base import BaseDevice
        >>> from qxmt.devices.builder import DeviceBuilder
        >>> config = DeviceConfig(
        ...     platform="pennylane",
        ...     name="default.qubit",
        ...     n_qubits=2,
        ...     shots=1000,
        >>> )
        >>> device = DeviceBuilder(config).build()
        >>> feature_map = ZZFeatureMap(2, 2)
        >>> class CustomKernel(BaseKernel):
        ...     def __init__(self, device: BaseDevice, feature_map: Callable[[np.ndarray], None]) -> None:
        ...         super().__init__(device, feature_map)
        ...
        ...     def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        ...         return np.dot(x1, x2)
        >>> kernel = CustomKernel(device, feature_map)
        >>> x1 = np.random.rand(2)
        >>> x2 = np.random.rand(2)
        >>> kernel.compute(x1, x2)
        0.28
    """

    def __init__(
        self,
        device: BaseDevice,
        feature_map: BaseFeatureMap | Callable[[np.ndarray], None],
    ) -> None:
        """Initialize the kernel class.

        Args:
            device (BaseDevice): device instance for quantum computation
            feature_map (BaseFeatureMap | Callable[[np.ndarray], None]): feature map instance or function
        """
        self.device: BaseDevice = device
        self.platform: str = self.device.platform
        self.n_qubits: int = self.device.n_qubits
        self.feature_map: BaseFeatureMap = self._set_feature_map(feature_map)
        self.is_sampling: bool = (self.device.shots is not None) and (self.device.shots > 0)

    def _set_feature_map(self, feature_map: BaseFeatureMap | Callable[[np.ndarray], None]) -> BaseFeatureMap:
        """Set the feature map instance of the BaseFeatureMap.

        Args:
            feature_map (BaseFeatureMap | Callable[[np.ndarray], None]): feature map instance or function

        Returns:
            BaseFeatureMap: feature map instance
        """
        if isinstance(feature_map, FunctionType):
            return self._to_fm_instance(feature_map)
        else:
            return cast(BaseFeatureMap, feature_map)

    def _to_fm_instance(self, feature_map_func: Callable[[np.ndarray], None]) -> BaseFeatureMap:
        """Convert a feature map function to a BaseFeatureMap instance.

        Args:
            feature_map_func (Callable[[np.ndarray], None]): function that defines the feature map circuit.
                if the function needs some parameters, it should be defined in the function as a default value.

        Returns:
            BaseFeatureMap: instance of BaseFeatureMap
        """

        class CustomFeatureMap(BaseFeatureMap):
            def __init__(self, platform: str, n_qubits: int) -> None:
                super().__init__(platform, n_qubits)

            def feature_map(self, x: np.ndarray) -> None:
                feature_map_func(x)

        return CustomFeatureMap(self.platform, self.n_qubits)

    def _validate_sampling_values(self, sampling_result: np.ndarray, valid_values: list[int] = [0, 1]) -> None:
        """Validate the sampling resutls of each shots.

        Args:
            sampling_result (np.ndarray): array of sampling results
            valid_values (list[int], optional): valid value of quantum state. Defaults to [0, 1].

        Raises:
            DeviceSettingError: qunatum device is not in sampling mode
            ValueError: invalid values in the sampling results
        """
        if not self.is_sampling:
            raise DeviceSettingError("Shots must be set to a positive integer value to use sampling.")

        if not np.all(np.isin(sampling_result, valid_values)):
            unique_values = np.unique(sampling_result)
            invalid_values = unique_values[~np.isin(unique_values, valid_values)]
            raise ValueError(f"The input array contains values other than 0 and 1. (invalid values: {invalid_values})")

    def _generate_all_observable_states(self, state_pattern: str = "01") -> list[str]:
        """Generate all possible observable states for the given number of qubits.

        Returns:
            list[str]: list of all possible observable states
        """
        return ["".join(bits) for bits in product(state_pattern, repeat=self.n_qubits)]

    @abstractmethod
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute kernel value between two samples.

        Args:
            x1 (np.ndarray): array of one sample
            x2 (np.ndarray): array of one sample

        Returns:
            tuple[float, np.ndarray]: kernel value and probability distribution
        """
        pass

    def compute_matrix(
        self,
        x_array_1: np.ndarray,
        x_array_2: np.ndarray,
        return_shots_resutls: bool = False,
        n_jobs: int = DEFAULT_N_JOBS,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Default implementation of kernel matrix computation.

        Args:
            x_array_1 (np.ndarray): array of samples (ex: training data)
            x_array_2 (np.ndarray): array of samples (ex: test data)
            return_shots_resutls (bool, optional): return the shot results. Defaults to False.
            n_jobs (int, optional): number of jobs for parallel computation. Defaults to DEFAULT_N_JOBS.

        Returns:
            np.ndarray: computed kernel matrix
        """

        def _compute_entry(i: int, j: int) -> tuple[int, int, tuple[float, np.ndarray]]:
            return i, j, self.compute(x_array_1[i], x_array_2[j])

        # compute each entry of the kernel matrix in parallel
        n_samples_1 = len(x_array_1)
        n_samples_2 = len(x_array_2)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_entry)(i, j) for i in range(n_samples_1) for j in range(n_samples_2)
        )

        # initialize the shots results matrix when return_shots_resutls is True and sampling is enabled
        if self.is_sampling and return_shots_resutls:
            num_state = 2**self.device.n_qubits
            shots_matrix = np.zeros((n_samples_1, n_samples_2, num_state))
        else:
            shots_matrix = None

        kernel_matrix = np.zeros((n_samples_1, n_samples_2))
        for i, j, result in results:  # type: ignore
            kernel_matrix[i, j] = result[0]

            if shots_matrix is not None:
                shots_matrix[i, j] = result[1]

        return kernel_matrix, shots_matrix

    def save_shots_results(self, probs_matrix: np.ndarray, save_path: str | Path) -> None:
        """Save the shot results to a file.
        probs_matrix contains the probability distribution of the observable states for each sample.
        expected shape of probs_matrix: (n_samples_1, n_samples_2, num_state)

        Args:
            probs_matrix (np.ndarray): probability distribution of the observable states
            save_path (str | Path): save path for the shot results

        Raises:
            DeviceSettingError: not sampling mode
            ValueError: save path extension not ".h5"
            ValueError: state labels and probs length mismatch
        """
        if not self.is_sampling:
            raise DeviceSettingError("Shots must be set to a positive integer value to save the shot results.")

        if Path(save_path).suffix != ".h5":
            raise ValueError("The save path must be a .h5 file.")

        state_labels = self._generate_all_observable_states(state_pattern="01")
        if probs_matrix.shape[2] != len(state_labels):
            raise ValueError(
                "The length of the state of probability distribution must be equal to the number of observable states."
            )

        with h5py.File(save_path, "w") as f:
            probs_dataset = f.create_dataset("probs", data=probs_matrix, compression="gzip", dtype="float32")

            metadata_group = f.create_group("metadata")
            metadata_group.attrs["platform"] = self.platform
            metadata_group.attrs["n_qubits"] = self.n_qubits
            metadata_group.attrs["shots"] = self.device.shots
            metadata_group.attrs["state_labels"] = state_labels
            metadata_group.attrs["n_samples"] = probs_dataset.shape[0] * probs_dataset.shape[1]

    def plot_matrix(
        self,
        x_array_1: np.ndarray,
        x_array_2: np.ndarray,
        save_path: Optional[str | Path] = None,
        n_jobs: int = DEFAULT_N_JOBS,
    ) -> None:
        """Plot kernel matrix for given samples.
        Caluculation of kernel values is performed in parallel.

        Args:
            x_array_1 (np.ndarray): array of samples (ex: training data)
            x_array_2 (np.ndarray): array of samples (ex: test data)
            save_path (Optional[str | Path], optional): save path for the plot. Defaults to None.
            n_jobs (int, optional): number of jobs for parallel computation. Defaults to DEFAULT_N_JOBS.
        """

        kernel_matrix, _ = self.compute_matrix(x_array_1, x_array_2, return_shots_resutls=False, n_jobs=n_jobs)
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
        Caluculation of kernel values is performed in parallel.

        Args:
            x_train (np.ndarray): array of training samples
            x_test (np.ndarray): array of testing samples
            save_path (Optional[str | Path], optional): save path for the plot. Defaults to None.
            n_jobs (int, optional): number of jobs for parallel computation. Defaults to DEFAULT_N_JOBS.
        """
        train_kernel, _ = self.compute_matrix(x_train, x_train, return_shots_resutls=False, n_jobs=n_jobs)
        test_kernel, _ = self.compute_matrix(x_test, x_train, return_shots_resutls=False, n_jobs=n_jobs)

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
