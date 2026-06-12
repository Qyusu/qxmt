from typing import TYPE_CHECKING, Callable

import numpy as np
from qiskit import QuantumCircuit

from qxmt.feature_maps.qiskit.base import QiskitBaseFeatureMap
from qxmt.kernels.qiskit.base import QiskitBaseKernel

if TYPE_CHECKING:
    from qxmt.devices.qiskit_device import QiskitDevice


class FidelityKernel(QiskitBaseKernel):
    """Fidelity kernel class for Qiskit."""

    def __init__(
        self,
        device: "QiskitDevice",
        feature_map: QiskitBaseFeatureMap | Callable[[np.ndarray], None],
    ) -> None:
        super().__init__(device, feature_map)

    def _process_state_vector(self, state_vector: np.ndarray) -> np.ndarray:
        """Process the raw state vector for fidelity kernel computation."""
        return state_vector

    def _compute_kernel_block(self, block1: np.ndarray, block2: np.ndarray) -> np.ndarray:
        """Compute fidelity kernel values for blocks of states."""
        inner_block = np.dot(block1, np.conj(block2.T))
        return np.abs(inner_block) ** 2

    def _circuit_for_sampling(self, *args: np.ndarray) -> QuantumCircuit:
        """Build a fidelity circuit for sampling mode."""
        self._validate_circuit_args(args, 2, "FidelityKernel._circuit_for_sampling")
        x1, x2 = args

        circuit = self._build_feature_map_circuit(x1)
        circuit.compose(self._build_feature_map_circuit(x2).inverse(), inplace=True)
        return circuit

    def _compute_by_sampling(self, x1: np.ndarray, x2: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute the fidelity kernel value between two data points."""
        if not self.is_sampling:
            raise ValueError("_compute_by_sampling method is only available in sampling mode.")

        circuit = self._circuit_for_sampling(x1, x2)
        counts = self._run_sampling_circuit(circuit)
        probs = self._convert_counts_to_probs(counts)
        kernel_value = float(probs[0])

        return kernel_value, probs
