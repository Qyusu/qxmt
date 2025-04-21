from typing import Literal, Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from qxmt.types import PROCESSCED_DATASET_TYPE

SCALER_TYPE = Literal["StandardScaler", "MinMaxScaler"]


def normalization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_type: SCALER_TYPE = "StandardScaler",
) -> PROCESSCED_DATASET_TYPE:
    """Normalization of dataset by StandardScaler or MinMaxScaler

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
    match scaler_type:
        case "StandardScaler":
            scaler = StandardScaler()
        case "MinMaxScaler":
            scaler = MinMaxScaler()
        case _:
            raise ValueError(f"Invalid scaler type: {scaler_type}")

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
