import multiprocessing as mp
from abc import ABC, abstractmethod
from pathlib import Path
from types import FunctionType
from typing import Callable, Optional, cast

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rich.progress import Progress

from qxmt.constants import DEFAULT_N_JOBS
from qxmt.devices.base import BaseDevice
from qxmt.exceptions import DeviceSettingError
from qxmt.feature_maps.base import BaseFeatureMap, FeatureMapFromFunc
from qxmt.kernels.sampling import generate_all_observable_states

MAX_IBMQ_REQUEST_NUM = 3
STATE_VECTOR_BLOCK_SIZE = 2048


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
            return FeatureMapFromFunc(self.platform, self.n_qubits, feature_map)
        else:
            return cast(BaseFeatureMap, feature_map)

    @abstractmethod
    def _compute_matrix_by_state_vector(
        self,
        x1_array: np.ndarray,
        x2_array: np.ndarray,
        bar_label: str = "",
        show_progress: bool = True,
        block_size: int = STATE_VECTOR_BLOCK_SIZE,
    ) -> np.ndarray:
        """Compute kernel value between two samples.

        Args:
            x1_array (np.ndarray): array of all sample
            x2_array (np.ndarray): array of all sample
            bar_label (str, optional): label for the progress bar. Defaults to empty string ("").
            show_progress (bool, optional): whether to show progress bar. Defaults to True.
            block_size (int, optional): block size for the batch computation. Defaults to 2048.

        Returns:
            np.ndarray: kernel value and probability distribution
        """
        pass

    @abstractmethod
    def _compute_by_sampling(self, x1: np.ndarray, x2: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute kernel value between two samples.

        Args:
            x1 (np.ndarray): array of one sample
            x2 (np.ndarray): array of one sample

        Returns:
            tuple[float, np.ndarray]: kernel value and probability distribution
        """
        pass

    def _compute_entry_by_sampling(
        self, i: int, j: int, x_array_1: np.ndarray, x_array_2: np.ndarray, progress_queue: Optional[mp.Queue]
    ) -> tuple[int, int, tuple[float, np.ndarray] | Exception]:
        """Compute each entry of the kernel matrix.
        This method is used for parallel computation of the kernel matrix.
        If an error occurs in the self.compute() method, it is handled and returned as an exception.

        Args:
            i (int): row index of kernel matrix
            j (int): column index of kernel matrix
            x_array_1 (np.ndarray): input array 1 that index i for kernel computation
            x_array_2 (np.ndarray): input array 2 that index j for kernel computation
            progress_queue (Optional[mp.Queue]): progress queue for tracking the progress

        Returns:
            tuple[int, int, tuple[float, np.ndarray] | Exception]:
                row index, column index, kernel value and probability distribution

        Raises:
            Exception: error in the self.compute() method
        """
        try:
            result = self._compute_by_sampling(x_array_1, x_array_2)
            if progress_queue is not None:
                progress_queue.put(1)
            return i, j, result
        except Exception as e:
            if progress_queue is not None:
                progress_queue.put(1)
            return i, j, e

    def _compute_matrix_by_sampling(
        self,
        x_array_1: np.ndarray,
        x_array_2: np.ndarray,
        return_shots_resutls: bool,
        n_jobs: int = DEFAULT_N_JOBS,
        bar_label: str = "",
        show_progress: bool = True,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Default implementation of kernel matrix computation.
        Due to the parallel computation, raise an error is delayed until the end of the computation.

        Args:
            x_array_1 (np.ndarray): array of samples (ex: training data)
            x_array_2 (np.ndarray): array of samples (ex: test data)
            return_shots_resutls (bool): return the shot results.
            n_jobs (int, optional): number of jobs for parallel computation. Defaults to DEFAULT_N_JOBS.
            bar_label (str, optional): label for the progress bar. Defaults to empty string ("").
            show_progress (bool, optional): whether to show progress bar. Defaults to True.

        Returns:
            np.ndarray: computed kernel matrix

        Raises:
            Exception: error in the self.compute() method
        """

        # compute each entry of the kernel matrix in parallel
        n_samples_1 = len(x_array_1)
        n_samples_2 = len(x_array_2)

        # parallel computation for each entry of the kernel matrix
        tasks = [(i, j, x_array_1[i], x_array_2[j]) for i in range(n_samples_1) for j in range(n_samples_2)]

        if show_progress:
            with mp.Manager() as manager:
                progress_queue = manager.Queue()
                with Progress() as progress:
                    bar_label = f" ({bar_label})" if bar_label else ""
                    task_progress = progress.add_task(f"Computing Kernel Matrix{bar_label}", total=len(tasks))

                    with mp.Pool(processes=n_jobs) as pool:
                        results = pool.starmap_async(
                            self._compute_entry_by_sampling,
                            [(i, j, x_array_1, x_array_2, progress_queue) for (i, j, x_array_1, x_array_2) in tasks],
                        )

                        # track progress
                        completed = 0
                        while not progress.finished:
                            progress_queue.get()
                            completed += 1
                            progress.update(task_progress, completed=completed)

                        # get all process results
                        results.wait()
                        final_results = results.get()

                        # finalize progress bar
                        progress.update(task_progress, completed=len(tasks))
                        progress.stop_task(task_progress)
                        progress.refresh()
        else:
            with mp.Pool(processes=n_jobs) as pool:
                results = pool.starmap(
                    self._compute_entry_by_sampling,
                    [(i, j, x_array_1, x_array_2, None) for (i, j, x_array_1, x_array_2) in tasks],
                )
                final_results = results

        # initialize the shots results matrix when return_shots_resutls is True and sampling is enabled
        if self.is_sampling and return_shots_resutls:
            num_state = 2**self.device.n_qubits
            shots_matrix = np.zeros((n_samples_1, n_samples_2, num_state))
        else:
            shots_matrix = None

        kernel_matrix = np.zeros((n_samples_1, n_samples_2))
        for i, j, result in final_results:
            if isinstance(result, Exception):
                # raise error in self.compute() method
                raise result
            else:
                # success to compute the kernel value
                kernel_matrix[i, j] = result[0]

                if shots_matrix is not None:
                    shots_matrix[i, j] = result[1]

        return kernel_matrix, shots_matrix

    def _compute_entry_by_ibmq(
        self,
        i: int,
        j: int,
        x_array_1: np.ndarray,
        x_array_2: np.ndarray,
    ) -> tuple[int, int, tuple[float | list[float], np.ndarray]]:
        result = self._compute_by_sampling(x_array_1, x_array_2)
        return i, j, result

    def _compute_matrix_by_ibmq(
        self,
        x_array_1: np.ndarray,
        x_array_2: np.ndarray,
        return_shots_resutls: bool,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute the kernel matrix for given samples by IBM Quantum real device.

        Args:
            x_array_1 (np.ndarray): array of samples (ex: training data)
            x_array_2 (np.ndarray): array of samples (ex: test data)
            return_shots_resutls (bool): return the shot results.

        Returns:
            tuple[np.ndarray, Optional[np.ndarray]]: computed kernel matrix and shot results
        """
        # compute each entry of the kernel matrix in parallel
        n_samples_1 = len(x_array_1)
        n_samples_2 = len(x_array_2)

        tasks = [(i, j, x_array_1[i], x_array_2[j]) for i in range(n_samples_1) for j in range(n_samples_2)]
        with mp.Pool(processes=MAX_IBMQ_REQUEST_NUM) as pool:
            results = pool.starmap_async(
                self._compute_entry_by_ibmq,
                [(i, j, x_array_1, x_array_2) for (i, j, x_array_1, x_array_2) in tasks],
            )

            # get all process results
            results.wait()
            final_results = results.get()

        # initialize the shots results matrix when return_shots_resutls is True
        if self.is_sampling and return_shots_resutls:
            num_state = 2**self.device.n_qubits
            shots_matrix = np.zeros((n_samples_1, n_samples_2, num_state))
        else:
            shots_matrix = None

        kernel_matrix = np.zeros((n_samples_1, n_samples_2))
        for i, j, result in final_results:
            kernel_matrix[i, j] = result[0]

            if shots_matrix is not None:
                shots_matrix[i, j] = result[1]

        return kernel_matrix, shots_matrix

    def compute_matrix(
        self,
        x_array_1: np.ndarray,
        x_array_2: np.ndarray,
        return_shots_resutls: bool = False,
        n_jobs: int = DEFAULT_N_JOBS,
        bar_label: str = "",
        show_progress: bool = True,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute the kernel matrix for given samples.
        The kernel matrix is computed by simulator or IBM Quantum real device.

        Args:
            x_array_1 (np.ndarray): array of samples (ex: training data)
            x_array_2 (np.ndarray): array of samples (ex: test data)
            return_shots_resutls (bool, optional): return the shot results. Defaults to False.
            n_jobs (int, optional): parallel computation for each entry of the kernel matrix.
                This value only valid for simulator mode. Defaults to DEFAULT_N_JOBS.
            bar_label (str, optional): label for the progress bar. Defaults to empty string ("").
            show_progress (bool, optional): whether to show progress bar. Defaults to True.

        Returns:
            tuple[np.ndarray, Optional[np.ndarray]]: computed kernel matrix and shot results
        """
        if self.device.is_simulator() and not self.is_sampling:
            kernel_matrix = self._compute_matrix_by_state_vector(x_array_1, x_array_2, bar_label, show_progress)
            return kernel_matrix, None
        elif self.device.is_simulator():
            return self._compute_matrix_by_sampling(
                x_array_1, x_array_2, return_shots_resutls, n_jobs, bar_label, show_progress
            )
        else:
            return self._compute_matrix_by_ibmq(x_array_1, x_array_2, return_shots_resutls)

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

        state_labels = generate_all_observable_states(self.n_qubits, state_pattern="01")
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
            metadata_group.attrs["n_elements"] = probs_dataset.shape[0] * probs_dataset.shape[1]

    def plot_matrix(
        self,
        x_array_1: np.ndarray,
        x_array_2: np.ndarray,
        save_path: Optional[str | Path] = None,
        n_jobs: int = DEFAULT_N_JOBS,
        show_progress: bool = True,
    ) -> None:
        """Plot kernel matrix for given samples.
        Caluculation of kernel values is performed in parallel.

        Args:
            x_array_1 (np.ndarray): array of samples (ex: training data)
            x_array_2 (np.ndarray): array of samples (ex: test data)
            save_path (Optional[str | Path], optional): save path for the plot. Defaults to None.
            n_jobs (int, optional): number of jobs for parallel computation. Defaults to DEFAULT_N_JOBS.
            show_progress (bool, optional): whether to show progress bar. Defaults to True.
        """

        kernel_matrix, _ = self.compute_matrix(
            x_array_1, x_array_2, return_shots_resutls=False, n_jobs=n_jobs, show_progress=show_progress
        )
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
        show_progress: bool = True,
    ) -> None:
        """Plot kernel matrix for training and testing data.
        Caluculation of kernel values is performed in parallel.

        Args:
            x_train (np.ndarray): array of training samples
            x_test (np.ndarray): array of testing samples
            save_path (Optional[str | Path], optional): save path for the plot. Defaults to None.
            n_jobs (int, optional): number of jobs for parallel computation. Defaults to DEFAULT_N_JOBS.
            show_progress (bool, optional): whether to show progress bar. Defaults to True.
        """
        train_kernel, _ = self.compute_matrix(
            x_train, x_train, return_shots_resutls=False, n_jobs=n_jobs, bar_label="Train", show_progress=show_progress
        )
        test_kernel, _ = self.compute_matrix(
            x_test, x_train, return_shots_resutls=False, n_jobs=n_jobs, bar_label="Test", show_progress=show_progress
        )

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
