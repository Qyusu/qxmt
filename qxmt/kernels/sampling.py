from collections import Counter
from itertools import product

import numpy as np


def validate_sampling_values(sampling_result: np.ndarray, valid_values: list[int] = [0, 1]) -> None:
    """Validate the sampling resutls of each shots.

    Args:
        sampling_result (np.ndarray): array of sampling results
        valid_values (list[int], optional): valid value of quantum state. Defaults to [0, 1].

    Raises:
        ValueError: invalid values in the sampling results
    """
    if not np.all(np.isin(sampling_result, valid_values)):
        unique_values = np.unique(sampling_result)
        invalid_values = unique_values[~np.isin(unique_values, valid_values)]
        raise ValueError(f"The input array contains values other than 0 and 1. (invalid values: {invalid_values})")


def generate_all_observable_states(n_qubits: int, state_pattern: str = "01") -> list[str]:
    """Generate all possible observable states for the given number of qubits.

    Args:
        n_qubits (int): number of qubits
        state_pattern (str, optional): pattern of the observable state. Defaults to "01".

    Returns:
        list[str]: list of all possible observable states
    """
    return ["".join(bits) for bits in product(state_pattern, repeat=n_qubits)]


def sample_results_to_probs(
    result: np.ndarray | list, n_qubits: int, shots: int, state_pattern: str = "01"
) -> np.ndarray:
    """Convert the sampling results to the probability of each state.
    Handles both integer results (e.g., [0, 3, 2]) and bitstring results (e.g., [[0,0], [1,1]]).

    Args:
        result (np.ndarray | list): sampling results (integers or bit arrays)
        n_qubits (int): number of qubits
        shots (int): number of shots
        state_pattern (str, optional): pattern of the observable state. Defaults to "01".

    Returns:
        np.ndarray: probability of each state
    """
    result_array = np.array(result)

    # If result is already 1D (integers or single-qubit bits), treat as integers
    if result_array.ndim == 1:
        if state_pattern == "01":
            n_states = 2**n_qubits
            counts = np.bincount(result_array.astype(int), minlength=n_states)
            if len(counts) > n_states:
                counts = counts[:n_states]
            return counts / shots

    # If result is 2D (multi-qubit bits), convert to integers or strings
    elif result_array.ndim == 2:
        validate_sampling_values(result_array)
        if state_pattern == "01":
            powers = 1 << np.arange(n_qubits - 1, -1, -1)
            int_results = result_array.dot(powers)
            n_states = 2**n_qubits
            counts = np.bincount(int_results.astype(int), minlength=n_states)
            if len(counts) > n_states:
                counts = counts[:n_states]
            return counts / shots

    # Generic path for custom patterns or unhandled structures
    result_array = np.array([result_array]) if result_array.ndim == 1 else result_array
    bit_strings = ["".join(map(str, sample)) for sample in result_array]
    all_states = generate_all_observable_states(n_qubits, state_pattern=state_pattern)

    count_dict = Counter(bit_strings)
    state_counts = [count_dict.get(state, 0) for state in all_states]
    probs = np.array(state_counts) / shots

    return probs
