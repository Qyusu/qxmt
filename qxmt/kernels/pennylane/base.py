from abc import abstractmethod
from typing import Any, Callable

import numpy as np
import pennylane as qml
from pennylane.measurements import SampleMP, StateMP

from qxmt.devices.base import BaseDevice
from qxmt.exceptions import ModelSettingError
from qxmt.feature_maps.base import BaseFeatureMap
from qxmt.kernels.base import BaseKernel


class PennyLaneBaseKernel(BaseKernel):
    """PennyLane base kernel class."""

    def __init__(self, device: BaseDevice, feature_map: BaseFeatureMap | Callable[[np.ndarray], None]) -> None:
        super().__init__(device, feature_map)
        self.qnode: qml.QNode | None = None
        self.state_memory: dict[tuple[float, ...], float | np.ndarray] = {}

    @abstractmethod
    def _circuit_for_sampling(self, *args: np.ndarray) -> SampleMP | list[SampleMP]:
        """Circuit for sampling mode.

        Args:
            *args: Variable number of numpy arrays. Can be:
                - Single array (x) for single input circuits
                - Two arrays (x1, x2) for fidelity-based circuits

        Returns:
            SampleMP | list[SampleMP]: Measurement result
        """
        pass

    @abstractmethod
    def _circuit_for_state_vector(self, *args: np.ndarray) -> StateMP:
        """Circuit for state vector mode.

        Args:
            *args: Variable number of numpy arrays. Can be:
                - Single array (x) for single input circuits
                - Two arrays (x1, x2) for fidelity-based circuits

        Returns:
            StateMP: State measurement result
        """
        pass

    def _initialize_qnode(self) -> None:
        if self.feature_map is None:
            raise ModelSettingError("Feature map must be provided for PennyLaneBaseKernel.")

        if (self.qnode is None) and (self.is_sampling):
            self.qnode = qml.QNode(
                self._circuit_for_sampling, device=self.device.get_device(), cache=False, diff_method=None
            )
        elif (self.qnode is None) and (not self.is_sampling):
            self.qnode = qml.QNode(
                self._circuit_for_state_vector, device=self.device.get_device(), cache=False, diff_method=None
            )

    def get_circuit_spec(self, x: np.ndarray) -> dict[str, Any]:
        return qml.specs(self.qnode)(x)  # type: ignore
