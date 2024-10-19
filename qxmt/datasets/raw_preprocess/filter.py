import numpy as np

from qxmt.types import RAW_DATASET_TYPE


def filter_by_categorical(X: np.ndarray, y: np.ndarray, drop_na: bool = True) -> RAW_DATASET_TYPE:
    """Filter the dataset by categorical features.

    Args:
        X (np.ndarray): input data
        y (np.ndarray): label of input data
        drop_na (bool, optional): drop NaN values. Defaults to True.

    Returns:
        RAW_DATASET_TYPE: filtered data
    """
    numerical_indices = []
    for i in range(X.shape[1]):
        try:
            X[:, i].astype(float)
            numerical_indices.append(i)
        except ValueError:
            # if cannot convert to float, it regards as categorical feature
            pass

    X_filtered = X[:, numerical_indices]

    if drop_na:
        X_filtered = np.array(X_filtered, dtype=float)
        mask = ~np.isnan(X_filtered).any(axis=1)
        X_filtered = X_filtered[mask]
        y_filtered = y[mask]

    return X_filtered, y_filtered
