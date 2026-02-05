from typing import Callable, cast

import numpy as np
from qulacs import QuantumCircuit, QuantumState

from qxmt.devices import QulacsDevice
from qxmt.feature_maps import QulacsBaseFeatureMap
from qxmt.kernels.qulacs.base import QulacsBaseKernel


class FidelityKernel(QulacsBaseKernel):
    """Fidelity kernel class for Qulacs.
    The fidelity kernel is a quantum kernel that computes the kernel value based on the fidelity
    between two quantum states.
    """

    def __init__(
        self,
        device: QulacsDevice,
        feature_map: QulacsBaseFeatureMap | Callable[[np.ndarray], None],
    ) -> None:
        """Initialize the FidelityKernel class.

        Args:
            device (QulacsDevice): qulacs device instance for quantum computation
            feature_map (QulacsBaseFeatureMap | Callable[[np.ndarray], None]): feature map instance or function
        """
        super().__init__(device, feature_map)

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

    def _circuit_for_sampling(self, *args: np.ndarray) -> np.ndarray:
        """Circuit for sampling mode.

        Args:
            *args: Variable number of numpy arrays. Can be:
                - Two arrays (x1, x2) for fidelity-based circuits

        Returns:
            np.ndarray: Measurement result or probability distribution
        """
        self._validate_circuit_args(args, 2, "FidelityKernel._circuit_for_sampling")
        x1, x2 = args

        state = QuantumState(self.n_qubits)
        state.set_zero_state()

        # apply feature map forward
        feature_map = cast(QulacsBaseFeatureMap, self.feature_map)
        feature_map.feature_map(x1)
        feature_map.circuit.update_quantum_state(state)

        # apply feature map backward
        feature_map.feature_map(x2)
        circuit_x2 = feature_map.circuit
        n_gates = circuit_x2.get_gate_count()
        inv_circuit = QuantumCircuit(self.n_qubits)
        for i in range(n_gates - 1, -1, -1):
            gate = circuit_x2.get_gate(i)
            inv_gate = gate.get_inverse()
            inv_circuit.add_gate(inv_gate)
        inv_circuit.update_quantum_state(state)

        # Sample or Measure
        samples = state.sampling(cast(int, self.device.shots))

        return np.array(samples)

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

        # Get samples from circuit
        samples = self._circuit_for_sampling(x1, x2)

        # Convert to probabilities
        probs = self._convert_sampling_results_to_probs(samples)

        kernel_value = probs[0]  # get |0..0> state probability

        return kernel_value, probs
