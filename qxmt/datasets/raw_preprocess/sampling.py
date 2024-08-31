import numpy as np

from qxmt.datasets.builder import RAW_DATASET_TYPE


def sampling_by_each_class(X: np.ndarray, y: np.ndarray, n_samples: int) -> RAW_DATASET_TYPE:
    """Data sampling by each class

    Args:
        X (np.ndarray): input data
        y (np.ndarray): label of input data

    Returns:
        RAW_DATASET_TYPE: sampled data
    """
    labels = [0, 1]
    # n_sample = 100
    y = np.array([int(label) for label in y])
    indices = np.where(np.isin(y, labels))[0]
    X, y = X[indices][:n_samples], y[indices][:n_samples]

    return X, y
