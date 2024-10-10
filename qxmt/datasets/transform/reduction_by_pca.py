from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from qxmt.types import PROCESSCED_DATASET_TYPE


def dimension_reduction_by_pca(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: int,
) -> PROCESSCED_DATASET_TYPE:
    """Dimension reduction by PCA

    Args:
        X_train (np.ndarray): numpy array of training data
        y_train (np.ndarray): numpy array of training label
        X_val (Optional[np.ndarray]): numpy array of validation data. None if validation set is not used
        y_val (Optional[np.ndarray]): numpy array of validation data. None if validation set is not used
        X_test (np.ndarray): numpy array of testing data
        y_test (np.ndarray): numpy array of testing label
        n_components (int): number of components to keep

    Returns:
        PROCESSCED_DATASET_TYPE: tuple of processed dataset
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    pca.fit(X_train_scaled)
    X_train_pca = pca.transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled) if X_val_scaled is not None else None
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test
