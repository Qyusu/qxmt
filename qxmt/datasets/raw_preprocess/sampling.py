import warnings

import numpy as np

from qxmt.types import RAW_DATASET_TYPE


def sample_by_count(X: np.ndarray, y: np.ndarray, n_samples: int, random_seed: int) -> RAW_DATASET_TYPE:
    """Randomly sample a fixed number of rows from the dataset.

    This function samples ``n_samples`` rows from ``X`` and the corresponding
    labels from ``y`` without replacement. Class balance is not considered; use
    ``sample_n_per_class`` when you need the same number of samples from each
    class.

    Args:
        X (np.ndarray): Input feature array. The first dimension must match
            the length of ``y``.
        y (np.ndarray): Label array corresponding to ``X``.
        n_samples (int): Number of rows to sample from the whole dataset.
        random_seed (int): Random seed used for reproducible sampling.

    Returns:
        RAW_DATASET_TYPE: Tuple of sampled features and labels.

    Raises:
        ValueError: If ``n_samples`` is larger than the number of rows in ``X``.
    """
    rng = np.random.default_rng(random_seed)
    indices = rng.choice(X.shape[0], n_samples, replace=False)

    return X[indices], y[indices]


def sampling_by_num(X: np.ndarray, y: np.ndarray, n_samples: int, random_seed: int) -> RAW_DATASET_TYPE:
    """Deprecated alias for sample_by_count."""
    warnings.warn(
        "sampling_by_num is deprecated. Use sample_by_count instead.",
        FutureWarning,
        stacklevel=2,
    )
    return sample_by_count(X, y, n_samples, random_seed)


def sample_n_per_class(
    X: np.ndarray, y: np.ndarray, n_samples: int, labels: list[int], random_seed: int
) -> RAW_DATASET_TYPE:
    """Randomly sample a fixed number of rows from each specified class.

    This function first shuffles the dataset with ``random_seed``, then extracts
    ``n_samples`` rows for every label in ``labels``. The total number of
    returned rows is therefore ``n_samples * len(labels)``. Labels in ``y`` are
    converted to ``int`` before comparison.

    Args:
        X (np.ndarray): Input feature array. The first dimension must match
            the length of ``y``.
        y (np.ndarray): Label array corresponding to ``X``.
        n_samples (int): Number of rows to sample from each label.
        labels (list[int]): Labels to include in the sampled dataset.
        random_seed (int): Random seed used for reproducible sampling.

    Returns:
        RAW_DATASET_TYPE: Tuple of sampled features and labels.

    Raises:
        ValueError: If any label in ``labels`` does not exist in ``y``.
        ValueError: If any requested label has fewer than ``n_samples`` rows.
    """
    not_exist_labels = set(labels) - set(map(int, np.unique(y)))
    if not_exist_labels:
        raise ValueError(f"Labels {not_exist_labels} do not exist in the dataset.")

    # fix random seed and shuffle
    rng = np.random.default_rng(random_seed)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # label convert to int type
    y_shuffled = np.array([int(label) for label in y_shuffled])
    sampled_indices = []
    for label in labels:
        label_indices = np.where(y_shuffled == label)[0]
        if len(label_indices) < n_samples:
            raise ValueError(f"Label {label} has only {len(label_indices)} samples, but {n_samples} are required.")
        sampled_indices.extend(label_indices[:n_samples])

    sampled_indices = np.sort(np.array(sampled_indices))
    X_sampled, y_sampled = X_shuffled[sampled_indices], y_shuffled[sampled_indices]

    return X_sampled, y_sampled


def sampling_by_each_class(
    X: np.ndarray, y: np.ndarray, n_samples: int, labels: list[int], random_seed: int
) -> RAW_DATASET_TYPE:
    """Deprecated alias for sample_n_per_class."""
    warnings.warn(
        "sampling_by_each_class is deprecated. Use sample_n_per_class instead.",
        FutureWarning,
        stacklevel=2,
    )
    return sample_n_per_class(X, y, n_samples, labels, random_seed)
