from typing import Callable

import numpy as np
import pennylane as qml
from pennylane.measurements.sample import SampleMP
from pennylane.measurements.state import StateMP

from qxmt.devices import BaseDevice
from qxmt.feature_maps import BaseFeatureMap
from qxmt.kernels.pennylane.base import PennyLaneBaseKernel


class FidelityKernel(PennyLaneBaseKernel):
    """Fidelity kernel class.
    The fidelity kernel is a quantum kernel that computes the kernel value based on the fidelity
    between two quantum states.

    Args:
        BaseKernel (_type_): base class of kernel

    Examples:
        >>> import numpy as np
        >>> from qxmt.kernels.pennylane.fidelity_kernel import FidelityKernel
        >>> from qxmt.feature_maps.pennylane.defaults import ZZFeatureMap
        >>> from qxmt.configs import DeviceConfig
        >>> from qxmt.devices.builder import DeviceBuilder
        >>> config = DeviceConfig(
        ...     platform="pennylane",
        ...     name="default.qubit",
        ...     n_qubits=2,
        ...     shots=1024,
        >>> )
        >>> device = DeviceBuilder(config).build()
        >>> feature_map = ZZFeatureMap(2, 2)
        >>> kernel = FidelityKernel(device, feature_map)
        >>> x1 = np.random.rand(2)
        >>> x2 = np.random.rand(2)
        >>> kernel.compute(x1, x2)
        0.14
    """

    def __init__(
        self,
        device: BaseDevice,
        feature_map: BaseFeatureMap | Callable[[np.ndarray], None],
    ) -> None:
        """Initialize the FidelityKernel class.

        Args:
            device (BaseDevice): device instance for quantum computation
            feature_map (BaseFeatureMap | Callable[[np.ndarray], None]): feature map instance or function
        """
        super().__init__(device, feature_map)

    def _circuit_for_sampling(self, *args: np.ndarray) -> SampleMP | list[SampleMP]:
        self._validate_circuit_args(args, 2, "FidelityKernel._circuit_for_sampling")
        x1, x2 = args
        self.feature_map(x1)
        qml.adjoint(self.feature_map)(x2)  # type: ignore

        return self._get_sampling_measurement()

    def _circuit_for_state_vector(self, *args: np.ndarray) -> StateMP:
        self._validate_circuit_args(args, 1, "FidelityKernel._circuit_for_state_vector")
        x = args[0]
        self.feature_map(x)

        return qml.state()

    def _process_state_vector(self, state_vector: np.ndarray) -> np.ndarray:
        """Process the raw state vector for fidelity kernel computation.

        Args:
            state_vector (np.ndarray): Raw state vector from quantum circuit

        Returns:
            np.ndarray: Raw state vector (no processing needed for fidelity kernel)
        """
        return state_vector

    def _compute_kernel_block(self, block1: np.ndarray, block2: np.ndarray) -> np.ndarray:
        """Compute fidelity kernel values for blocks of states.

        Args:
            block1 (np.ndarray): First block of state vectors
            block2 (np.ndarray): Second block of state vectors

        Returns:
            np.ndarray: Computed fidelity kernel block
        """
        inner_block = np.dot(block1, np.conj(block2.T))
        kernel_block = np.abs(inner_block) ** 2
        return kernel_block

    def _compute_by_sampling(self, x1: np.ndarray, x2: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute the fidelity kernel value between two data points.
        This method is only available in the sampling mode.
        Each kernel value computed by sampling the quantum circuit.

        Args:
            x1 (np.ndarray): numpy array representing the first data point
            x2 (np.ndarray): numpy array representing the second data point

        Returns:
            tuple[float, np.ndarray]: fidelity kernel value and probability distribution
        """
        if not self.is_sampling:
            raise ValueError("_compute_by_sampling method is only available in sampling mode.")

        result = self.qnode(x1, x2)
        probs = self._convert_sampling_results_to_probs(result)

        kernel_value = probs[0]  # get |0..0> state probability

        return kernel_value, probs
