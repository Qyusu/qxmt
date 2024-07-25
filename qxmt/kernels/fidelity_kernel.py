import numpy as np
import pennylane as qml
from pennylane.measurements.probs import ProbabilityMP

from qxmt.feature_maps.base import BaseFeatureMap
from qxmt.kernels.base import BaseKernel


class FidelityKernel(BaseKernel):
    def __init__(self, device: qml.Device, feature_map: BaseFeatureMap) -> None:
        super().__init__(device, feature_map)

    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        def circuit(x1: np.ndarray, x2: np.ndarray) -> ProbabilityMP:
            self.feature_map(x1)
            qml.adjoint(self.feature_map)(x2)  # type: ignore

            return qml.probs(wires=range(self.n_qubits))

        qnode = qml.QNode(circuit, self.device)
        probs = qnode(x1, x2)
        kernel_value = probs.numpy()[0]  # get |00> state probability

        return kernel_value
