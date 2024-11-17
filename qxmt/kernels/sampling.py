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


def sample_results_to_probs(result: np.ndarray, n_qubits: int, shots: int, state_pattern: str = "01") -> np.ndarray:
    """Convert the sampling results to the probability of each

    Args:
        result (np.ndarray): numpy array of sampling results
        n_qubits (int): number of qubits
        shots (int): number of shots
        state_pattern (str, optional): pattern of the observable state. Defaults to "01".

    Returns:
        np.ndarray: numpy array of the probability of each state
    """
    # validate sampleing results for getting the each state probability
    validate_sampling_values(result)

    # convert the sample results to bit strings
    # ex) shots=3, n_qubits=2, [[0, 0], [1, 1], [0, 0]] => ["00", "11", "00"]
    result = np.array([result]) if result.ndim == 1 else result
    bit_strings = ["".join(map(str, sample)) for sample in result]
    all_states = generate_all_observable_states(n_qubits, state_pattern=state_pattern)

    # count the number of each state
    count_dict = Counter(bit_strings)
    state_counts = [count_dict.get(state, 0) for state in all_states]

    # convert the count to the probability
    probs = np.array(state_counts) / shots  # shots must be over 0

    return probs
