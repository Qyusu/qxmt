from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import pennylane as qml
from pennylane.operation import Operation

from qxmt.devices.base import BaseDevice
from qxmt.exceptions import ModelSettingError
from qxmt.feature_maps.base import BaseFeatureMap
from qxmt.kernels.base import BaseKernel


class ProjectedKernel(BaseKernel):
    """Projected kernel class.
    The projected kernel is a quantum kernel that projects the quantum state to a specific basis
    and computes the kernel value based on the projected measurement results.
    Reference: https://www.nature.com/articles/s41467-021-22539-9

    Args:
        BaseKernel (_type_): base class of kernel

    Examples:
        >>> import numpy as np
        >>> from qxmt.kernels.pennylane.projected_kernel import ProjectedKernel
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
        >>> kernel = ProjectedKernel(device, feature_map)
        >>> x1 = np.random.rand(2)
        >>> x2 = np.random.rand(2)
        >>> kernel.compute(x1, x2)
        0.86
    """

    def __init__(
        self,
        device: BaseDevice,
        feature_map: BaseFeatureMap | Callable[[np.ndarray], None],
        gamma: float = 1.0,
        projection: Literal["x", "y", "z", "xyz", "xyz_sum"] = "xyz",
    ) -> None:
        """Initialize the ProjectedKernel class.

        Args:
            device (BaseDevice): device instance for quantum computation
            feature_map (BaseFeatureMap | Callable[[np.ndarray], None]): feature map instance or function
            gamma (float): gamma parameter for kernel computation
            projection (str): projection method for kernel computation
        """
        super().__init__(device, feature_map)
        self.n_qubits = device.n_qubits
        self.gamma = gamma
        self.projection = projection
        self.projection_ops = self._set_projection_ops(projection)
        self.qnode = qml.QNode(self._circuit, self.device())

    def _set_projection_ops(self, projection: str) -> list[Operation | list[Operation]]:
        """Set the projection operations based on the projection method.
        Projection method determines the Pauli operators to be measured for each qubit.

        Args:
            projection (str): projection method name for kernel computation

        Raises:
            ValueError: Invalid projection method

        Returns:
            list[Operation | list[Operation]]: list of projection operations
        """
        projection_ops = []
        match projection.lower():
            case "x":
                for i in range(self.n_qubits):
                    projection_ops.append(qml.PauliX(wires=i))
            case "y":
                for i in range(self.n_qubits):
                    projection_ops.append(qml.PauliY(wires=i))
            case "z":
                for i in range(self.n_qubits):
                    projection_ops.append(qml.PauliZ(wires=i))
            case "xyz":
                for i in range(self.n_qubits):
                    projection_ops.extend([qml.PauliX(wires=i), qml.PauliY(wires=i), qml.PauliZ(wires=i)])
            case "xyz_sum":
                for i in range(self.n_qubits):
                    projection_ops.append([qml.PauliX(wires=i), qml.PauliY(wires=i), qml.PauliZ(wires=i)])
            case _:
                raise ValueError(f'Invalid projection method: "{projection}"')

        return projection_ops

    def _process_measurement_results(self, results: list[float]) -> np.ndarray:
        """Process the measurement results based on the projection method.

        Args:
            results (list[float]): projected measurement results for each qubit

        Returns:
            np.ndarray: processed measurement results
        """
        if self.projection == "xyz_sum":
            sum_results: list[float] = []
            idx_size = 3  # x, y, z
            for i in range(0, self.n_qubits, idx_size):
                sum_result: float = np.sum([result for result in results[i : i + idx_size]])
                sum_results.append(sum_result)
            projected_results = np.array(sum_results)
        else:
            projected_results = np.array(results)

        return projected_results

    def _circuit(self, x: np.ndarray) -> list[Operation]:
        if self.feature_map is None:
            raise ModelSettingError("Feature map must be provided for FidelityKernel.")

        self.feature_map(x)
        measurement_results = []
        for op in self.projection_ops:
            if isinstance(op, list):
                for single_op in op:
                    measurement_results.append(qml.expval(single_op))
            else:
                measurement_results.append(qml.expval(op))

        return measurement_results

    def compute(self, x1: np.ndarray, x2: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute the projected kernel value between two data points.

        Args:
            x1 (np.ndarray): numpy array representing the first data point
            x2 (np.ndarray): numpy array representing the second data point

        Returns:
            tuple[float, np.ndarray]: projected kernel value and probability distribution
        """
        x1_projected = self._process_measurement_results(self.qnode(x1))
        x2_projected = self._process_measurement_results(self.qnode(x2))
        kernel_value = np.exp(-self.gamma * np.sum((x1_projected - x2_projected) ** 2))

        # [TODO]: Dummy value for now, it is not implemented yet.
        probs = np.array([1.0])

        return kernel_value, probs
