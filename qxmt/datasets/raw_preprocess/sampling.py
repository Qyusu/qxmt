import numpy as np

from qxmt.types import RAW_DATASET_TYPE


def sampling_by_num(X: np.ndarray, y: np.ndarray, n_samples: int, random_seed: int) -> RAW_DATASET_TYPE:
    """Data sampling by number of samples

    Args:
        X (np.ndarray): input data
        y (np.ndarray): label of input data
        n_samples (int): number of samples to be extracted
        random_seed (int): random seed

    Returns:
        RAW_DATASET_TYPE: sampled data
    """
    rng = np.random.default_rng(random_seed)
    indices = rng.choice(X.shape[0], n_samples, replace=False)

    return X[indices], y[indices]


def sampling_by_each_class(
    X: np.ndarray, y: np.ndarray, n_samples: int, labels: list[int], random_seed: int
) -> RAW_DATASET_TYPE:
    """Data sampling by each class

    Args:
        X (np.ndarray): input data
        y (np.ndarray): label of input data
        n_samples (int): number of samples to be extracted
        labels (list[int]): labels to be extracted
        random_seed (int): random seed

    Returns:
        RAW_DATASET_TYPE: sampled data
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
    indices = np.where(np.isin(y_shuffled, labels))[0]
    X_sampled, y_sampled = X_shuffled[indices][:n_samples], y_shuffled[indices][:n_samples]

    return X_sampled, y_sampled
