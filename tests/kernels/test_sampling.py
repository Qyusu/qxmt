import numpy as np
import pytest

from qxmt.kernels import generate_all_observable_states, validate_sampling_values


def test_validate_sampling_values() -> None:
    sampling_result = np.array([[0, 1], [1, 0], [0, 0], [1, 1], [0, 1]])

    # valid pattern
    validate_sampling_values(sampling_result, valid_values=[0, 1])

    # raise error if invalid values
    with pytest.raises(ValueError):
        validate_sampling_values(sampling_result, valid_values=[-1, 1])


def test_generate_all_observable_states() -> None:
    observable_states = generate_all_observable_states(n_qubits=2, state_pattern="01")
    assert observable_states == ["00", "01", "10", "11"]

    observable_states = generate_all_observable_states(n_qubits=5, state_pattern="01")
    # fmt: off
    assert observable_states == [
        "00000", "00001", "00010", "00011", "00100", "00101", "00110", "00111",
        "01000", "01001", "01010", "01011", "01100", "01101", "01110", "01111",
        "10000", "10001", "10010", "10011", "10100", "10101", "10110", "10111",
        "11000", "11001", "11010", "11011", "11100", "11101", "11110", "11111"
    ]
    # fmt: on
