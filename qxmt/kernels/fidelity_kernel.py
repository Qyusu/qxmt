import numpy as np
import pennylane as qml

from qxmt.feature_maps.base import BaseFeatureMap
from qxmt.kernels.base import BaseKernel


class FidelityKernel(BaseKernel):
    def __init__(self, device: qml.Device, feature_map: BaseFeatureMap) -> None:
        self.device: qml.Device = device
        self.feature_map: BaseFeatureMap = feature_map
        self.n_qubits: int = self.device.num_wires

    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        self.feature_map(x1, self.n_qubits)
        qml.adjoint(self.feature_map)(x2, self.n_qubits)  # type: ignore

        # get |00> state probability
        kernel_value = qml.probs(wires=range(self.n_qubits))[0]  # type: ignore

        return kernel_value
