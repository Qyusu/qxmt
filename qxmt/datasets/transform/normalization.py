from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from qxmt.types import PROCESSCED_DATASET_TYPE


def normalization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> PROCESSCED_DATASET_TYPE:
    """Normalization of dataset by StandardScaler

    Args:
        X_train (np.ndarray): numpy array of training data
        y_train (np.ndarray): numpy array of training label
        X_val (Optional[np.ndarray]): numpy array of validation data. None if validation set is not used
        y_val (Optional[np.ndarray]): numpy array of validation data. None if validation set is not used
        X_test (np.ndarray): numpy array of testing data
        y_test (np.ndarray): numpy array of testing label

    Returns:
        PROCESSCED_DATASET_TYPE: tuple of normalized dataset
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
