from pathlib import Path
from typing import Callable

import numpy as np
import pennylane as qml
import pytest

from qxmt import DatasetConfig, Experiment, GenerateDataConfig, SplitConfig
from qxmt.ansatze.pennylane.uccsd import UCCSDAnsatz
from qxmt.datasets import Dataset
from qxmt.devices import BaseDevice
from qxmt.devices.pennylane_device import PennyLaneDevice
from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian
from qxmt.kernels import BaseKernel
from qxmt.models.qkernels import QSVC, BaseMLModel
from qxmt.models.vqe import BasicVQE


def empty_feature_map(x: np.ndarray) -> None:
    qml.Identity(wires=0)


class TestKernel(BaseKernel):
    def __init__(self, device: BaseDevice, feature_map: Callable[[np.ndarray], None]) -> None:
        super().__init__(device, feature_map)

    def _compute_matrix_by_state_vector(
        self, x1: np.ndarray, x2: np.ndarray, bar_label: str = "", show_progress: bool = True
    ) -> np.ndarray:
        kernel_value = np.dot(x1, x2.T)
        return kernel_value

    def _compute_by_sampling(self, x1: np.ndarray, x2: np.ndarray) -> tuple[float, np.ndarray]:
        kernel_value = np.dot(x1, x2)
        probs = np.array([0.2, 0.1, 0.4, 0.3])  # dummy probs
        return kernel_value, probs


@pytest.fixture(scope="function")
def state_vec_qkernel_model() -> BaseMLModel:
    device = PennyLaneDevice(
        platform="pennylane", device_name="default.qubit", backend_name=None, n_qubits=2, shots=None
    )
    kernel = TestKernel(device=device, feature_map=empty_feature_map)
    return QSVC(kernel=kernel, n_jobs=1)


@pytest.fixture(scope="function")
def shots_qkernel_model() -> BaseMLModel:
    device = PennyLaneDevice(platform="pennylane", device_name="default.qubit", backend_name=None, n_qubits=2, shots=5)
    kernel = TestKernel(device=device, feature_map=empty_feature_map)
    return QSVC(kernel=kernel, n_jobs=1)


@pytest.fixture(scope="function")
def hamiltonian() -> MolecularHamiltonian:
    return MolecularHamiltonian(
        symbols=["H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        charge=0,
        multi=1,
        basis_name="STO-3G",
        active_electrons=2,
        active_orbitals=2,
    )


@pytest.fixture(scope="function")
def ansatz(hamiltonian: MolecularHamiltonian) -> UCCSDAnsatz:
    device = PennyLaneDevice(
        platform="pennylane", device_name="default.qubit", backend_name=None, n_qubits=4, shots=None
    )
    return UCCSDAnsatz(device=device, hamiltonian=hamiltonian)


@pytest.fixture(scope="function")
def state_vec_vqe_model(hamiltonian: MolecularHamiltonian, ansatz: UCCSDAnsatz) -> BasicVQE:
    device = PennyLaneDevice(
        platform="pennylane", device_name="default.qubit", backend_name=None, n_qubits=4, shots=None
    )
    return BasicVQE(
        device=device,
        hamiltonian=hamiltonian,
        ansatz=ansatz,
        diff_method="adjoint",
        optimizer_settings=None,
    )


@pytest.fixture(scope="function")
def shots_vqe_model(hamiltonian: MolecularHamiltonian, ansatz: UCCSDAnsatz) -> BasicVQE:
    device = PennyLaneDevice(platform="pennylane", device_name="default.qubit", backend_name=None, n_qubits=4, shots=5)
    return BasicVQE(
        device=device,
        hamiltonian=hamiltonian,
        ansatz=ansatz,
        diff_method="adjoint",
        optimizer_settings=None,
    )


@pytest.fixture(scope="function")
def base_experiment(tmp_path: Path) -> Experiment:
    return Experiment(
        name="test_exp",
        desc="test experiment",
        auto_gen_mode=False,
        root_experiment_dirc=tmp_path,
    )


@pytest.fixture(scope="function")
def create_random_dataset() -> Callable:
    def _create_random_dataset(
        data_num: int, feature_num: int, class_num: int, include_validation: bool = False
    ) -> Dataset:
        return Dataset(
            X_train=np.random.rand(data_num, feature_num),
            y_train=np.random.randint(class_num, size=data_num),
            X_val=np.random.rand(data_num, feature_num),
            y_val=np.random.randint(class_num, size=data_num),
            X_test=np.random.rand(data_num, feature_num),
            y_test=np.random.randint(class_num, size=data_num),
            config=DatasetConfig(
                generate=GenerateDataConfig(generate_method="linear"),
                split=SplitConfig(
                    train_ratio=0.8,
                    validation_ratio=0.1 if include_validation else 0.0,
                    test_ratio=0.1 if include_validation else 0.2,
                    shuffle=True,
                ),
                features=None,
            ),
        )

    return _create_random_dataset
