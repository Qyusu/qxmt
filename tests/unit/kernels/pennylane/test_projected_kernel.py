from typing import Optional, Type

import numpy as np
import pennylane as qml
import pytest

from qxmt.devices.base import BaseDevice
from qxmt.kernels.pennylane import ProjectedKernel


def encode_x_feature_map(x: np.ndarray) -> None:
    for i in range(len(x)):
        if x[i] == 1:
            qml.PauliX(wires=i)


@pytest.fixture(scope="function")
def device() -> BaseDevice:
    return BaseDevice(platform="pennylane", device_name="default.qubit", backend_name=None, n_qubits=2, shots=None)


class TestProjectedKernel:
    @pytest.mark.parametrize(
        ["projection", "x1", "x2", "expected", "expected_error"],
        [
            ("x", np.array([0, 1]), np.array([1, 0]), 1.0, None),
            ("y", np.array([0, 1]), np.array([1, 0]), 1.0, None),
            ("z", np.array([0, 1]), np.array([1, 0]), 0.00034, None),
            ("invalid_projection", np.array([0, 1]), np.array([1, 0]), None, ValueError),
        ],
    )
    def test_compute(
        self,
        device: BaseDevice,
        x1: np.ndarray,
        x2: np.ndarray,
        projection: str,
        expected: Optional[float],
        expected_error: Optional[Type[Exception]],
    ) -> None:
        if expected_error is not None:
            with pytest.raises(expected_error):
                projected_kernel = ProjectedKernel(
                    device,
                    feature_map=encode_x_feature_map,
                    gamma=1.0,
                    projection=projection,  # type: ignore
                )
        else:
            projected_kernel = ProjectedKernel(
                device,
                feature_map=encode_x_feature_map,
                gamma=1.0,
                projection=projection,  # type: ignore
            )
            # [TODO]: currently, projected kernel not support probs
            kernel_value, probs = projected_kernel.compute(x1, x2)
            assert round(kernel_value, 5) == expected
